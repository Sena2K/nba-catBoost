import os
import time
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.getenv("RAPIDAPI_KEY", "27244be62dmshb2410b2c636a6a7p18e011jsn345988bae2b4")

HEADERS = {
    "x-rapidapi-host": "api-nba-v1.p.rapidapi.com",
    "x-rapidapi-key": API_KEY,
}

SEASON = 2025
DATE_START = datetime(2025, 10, 21)
DATE_END = datetime(2025, 12, 15)

def fetch_season_games(season: int, max_retries: int = 5, backoff: int = 30):
    url = f"https://api-nba-v1.p.rapidapi.com/games?season={season}"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("response", [])
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    return []

def to_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def main():
    raw = fetch_season_games(SEASON)
    rows = []
    for g in raw:
        dt = to_dt(g.get("date", {}).get("start", ""))
        if not dt:
            continue
        if not (DATE_START <= dt.replace(tzinfo=None) <= DATE_END):
            continue
        rows.append({
            "game_id": g.get("id"),
            "date": g.get("date", {}).get("start"),
            "home_team_code": g.get("teams", {}).get("home", {}).get("code"),
            "away_team_code": g.get("teams", {}).get("visitors", {}).get("code"),
            "home_team_name": g.get("teams", {}).get("home", {}).get("name"),
            "away_team_name": g.get("teams", {}).get("visitors", {}).get("name"),
            "home_score": (g.get("scores", {}).get("home", {}) or {}).get("points"),
            "away_score": (g.get("scores", {}).get("visitors", {}) or {}).get("points"),
            "arena": (g.get("arena") or {}).get("name"),
            "city": (g.get("arena") or {}).get("city"),
            "season": g.get("season"),
            "status": (g.get("status") or {}).get("long"),
        })

    cols = ["game_id","date","home_team_code","away_team_code","home_team_name","away_team_name","home_score","away_score","arena","city","season","status"]
    df = pd.DataFrame(rows, columns=cols)

    base = f"nba_games_season_{SEASON}_2025-10-21_to_2025-12-15"
    df.to_excel(f"{base}.xlsx", index=False)

    print(f"Linhas: {len(df)}")
    print(f"Salvo: {base}.xlsx")

if __name__ == "__main__":
    main()
