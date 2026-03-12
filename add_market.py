import sys
import json
import urllib.request
from pathlib import Path

def add_market(url_or_slug: str, out_file: str = "markets.json"):
    """
    Fetches market details from Polymarket's Gamma API by event URL or slug,
    and appends them to the specified output JSON file.
    """
    # Extract slug safely
    # Handle both "polymarket.com/event/" and localized like "polymarket.com/ru/event/"
    if "/event/" in url_or_slug:
        # Avoid query params
        clean_url = url_or_slug.split("?")[0].rstrip("/")
        slug = clean_url.split("/")[-1]
    else:
        slug = url_or_slug.split("?")[0].rstrip("/")

    print(f"Fetching data for slug: {slug}")
    
    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return

    if not data:
        print(f"No event found for slug '{slug}'.")
        return

    event = data[0]
    markets = event.get("markets", [])
    if not markets:
        print("No markets found in this event.")
        return

    print(f"Found event: '{event.get('title', slug)}' with {len(markets)} market(s).\n")

    # Load existing markets.json
    markets_file = Path(out_file)
    existing_markets = []
    if markets_file.exists():
        try:
            with open(markets_file, "r", encoding="utf-8") as f:
                existing_markets = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {out_file} is not a valid JSON. Starting fresh.")
            existing_markets = []

    # Map existing condition IDs to avoid duplicates
    existing_ids = {m.get("condition_id") for m in existing_markets if m.get("condition_id")}
    
    added_count = 0
    for market in markets:
        condition_id = market.get("conditionId")
        if not condition_id:
            continue
            
        if condition_id in existing_ids:
            print(f"  [Skipped] Market already exists: {market.get('question', condition_id)}")
            continue

        raw_tokens = market.get("clobTokenIds", "[]")
        try:
            token_ids = json.loads(raw_tokens)
        except json.JSONDecodeError:
            token_ids = []

        if len(token_ids) < 2:
            print(f"  [Warning] Market '{market.get('question')}' has less than 2 token IDs. Adding anyway.")

        token_yes = token_ids[0] if len(token_ids) > 0 else ""
        token_no = token_ids[1] if len(token_ids) > 1 else ""

        new_entry = {
            "condition_id": condition_id,
            "token_id_yes": token_yes,
            "token_id_no": token_no,
            "question": market.get("question", "Unknown question")
        }
        
        existing_markets.append(new_entry)
        existing_ids.add(condition_id)
        added_count += 1
        print(f"  [Added] {new_entry['question']}")

    if added_count > 0:
        with open(markets_file, "w", encoding="utf-8") as f:
            json.dump(existing_markets, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Add a newline at the end of the file
        print(f"\nSuccessfully saved {added_count} new market(s) to '{out_file}'.")
    else:
        print("\nNo new markets added.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_market.py <polymarket_url_or_slug> [output_file]")
        sys.exit(1)
        
    target = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "markets.json"
    add_market(target, output)
