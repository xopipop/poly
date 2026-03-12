"""
analysis.py — Analysis Module (LLM-based probability estimation).

Sends collected news texts to OpenAI GPT-4o / GPT-4-turbo with a
Superforecaster-calibrated system prompt and returns a structured
JSON analysis with probability, confidence and reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import structlog
from openai import AsyncOpenAI

from data_ingestion import NewsItem

logger = structlog.get_logger(__name__)

# ── Superforecaster System Prompt ───────────────────────────

SYSTEM_PROMPT = """\
You are a world-class probabilistic forecaster modelled after Philip Tetlock's \
Superforecasters — the top 2 % of forecasters who consistently outperform \
prediction markets and intelligence analysts.

### Your cognitive toolkit
1. **Outside view first (base rates).**  Before examining the evidence, \
anchor on the historical base rate of similar events.  Ask yourself: \
"Of all situations structurally similar to this one, how often did the \
event occur?"
2. **Inside view update.**  Read every provided news article carefully.  \
Adjust your base-rate estimate up or down based on specific, diagnostic \
evidence.  Weight recent, high-quality, first-hand reporting more heavily \
than opinion pieces or speculation.
3. **Consider multiple scenarios.**  Enumerate at least two plausible \
scenarios (one where the event happens, one where it does not) and assign \
rough probabilities to each before synthesising a final number.
4. **Avoid cognitive biases.**  Guard against anchoring on salient \
headlines, availability bias, confirmation bias, and the narrative fallacy.  \
If the evidence is mixed, reflect that in a moderate probability — do NOT \
default to 50 % out of laziness; instead calibrate precisely.
5. **Granularity.**  Use precise numbers (e.g. 23 %, 67 %, 82 %) rather \
than round multiples of 5 or 10.  Well-calibrated forecasters differentiate \
between 60 % and 65 %.
6. **Confidence assessment.**  Separately rate how confident you are in \
your own forecast (0.0 = pure guess, 1.0 = certainty) based on evidence \
quality and quantity.

### Output rules
Return your answer as a **single JSON object** with exactly these keys:

```
{
  "reasoning": "<2–3 sentence explanation referencing specific evidence>",
  "probability": <integer 0–100>,
  "confidence": <float 0.0–1.0>
}
```

- `probability` — your best estimate that the event resolves **YES** \
(integer, 0 to 100).
- `confidence` — your meta-confidence in the estimate (float, 0.0 to 1.0).  \
Low confidence means the evidence is thin or contradictory.
- `reasoning` — concise but specific explanation grounded in the news provided.

Do NOT include any text outside the JSON object.
"""


# ── Result dataclass ────────────────────────────────────────


@dataclass(frozen=True)
class AnalysisResult:
    """Structured output from the LLM analysis."""

    probability: int          # 0–100
    confidence: float         # 0.0–1.0
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @property
    def probability_float(self) -> float:
        """Probability normalised to 0.0–1.0 (for downstream modules)."""
        return self.probability / 100.0


# ── Default / fallback result ───────────────────────────────

_FALLBACK = AnalysisResult(
    probability=50,
    confidence=0.0,
    reasoning="Analysis unavailable — defaulting to maximum uncertainty.",
)


# ── Analyzer class ──────────────────────────────────────────


class LLMAnalyzer:
    """Wrapper around OpenAI async client for probability estimation."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    # ── public API ──────────────────────────────────────────

    async def estimate_probability(
        self,
        market_question: str,
        news_items: Sequence[NewsItem],
    ) -> AnalysisResult:
        """Analyse *news_items* against *market_question* via LLM.

        Returns
        -------
        AnalysisResult
            Structured result with ``probability``, ``confidence``,
            and ``reasoning``.  On any error a safe fallback (50 / 0.0)
            is returned so the pipeline never crashes.
        """

        # Guard: no news at all
        if not news_items:
            logger.warning(
                "analysis_no_news",
                question=market_question[:80],
            )
            return AnalysisResult(
                probability=50,
                confidence=0.0,
                reasoning="No news articles available — cannot form an estimate.",
            )

        # ── build user message ──────────────────────────────
        news_block = "\n\n".join(
            f"### {i + 1}. {item.title} ({item.source})\n{item.summary}"
            for i, item in enumerate(news_items[:30])  # cap to avoid token overflow
        )
        user_msg = (
            f"QUESTION: {market_question}\n\n"
            f"NEWS ARTICLES ({len(news_items)} total, showing up to 30):\n\n"
            f"{news_block}"
        )

        logger.info(
            "llm_request",
            model=self._model,
            question=market_question[:80],
            articles=len(news_items),
        )

        # ── call OpenAI ─────────────────────────────────────
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )

            raw = response.choices[0].message.content or ""
            result = self._parse_response(raw)

            logger.info(
                "llm_estimate",
                probability=result.probability,
                confidence=result.confidence,
                reasoning=result.reasoning[:120],
            )
            return result

        except Exception as exc:
            logger.error("llm_request_failed", error=str(exc))
            return AnalysisResult(
                probability=50,
                confidence=0.0,
                reasoning=f"LLM error — defaulting to 50 %: {exc}",
            )

    # ── response parser ─────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> AnalysisResult:
        """Parse the LLM JSON response into an ``AnalysisResult``.

        Handles edge cases: missing keys, out-of-range values, and
        malformed JSON (falls back to ``_FALLBACK``).
        """
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("json_parse_error", raw=raw[:300])
            return AnalysisResult(
                probability=50,
                confidence=0.0,
                reasoning=f"Could not parse LLM response: {raw[:200]}",
            )

        # ── probability ─────────────────────────────────────
        prob_raw = data.get("probability", 50)
        try:
            prob = int(float(prob_raw))
        except (TypeError, ValueError):
            prob = 50
        prob = max(0, min(100, prob))

        # ── confidence ──────────────────────────────────────
        conf_raw = data.get("confidence", 0.5)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        # ── reasoning ───────────────────────────────────────
        reasoning = str(data.get("reasoning", "No reasoning provided."))

        return AnalysisResult(
            probability=prob,
            confidence=conf,
            reasoning=reasoning,
        )
