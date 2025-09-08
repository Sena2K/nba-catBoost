import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional

BASE_URL = "https://www.espn.com"
API_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}"

# TEAMS = [
#     "ATL","BOS","NJN","CHA","CHI","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM",
#     "MIA","MIL","MIN","NOH","NYK","OKC","ORL","PHI","PHO","POR","SAC","SAS","UTA","WAS"
# ]

TEAMS = [
    "ATL"
]

HEADERS_HTML = {"User-Agent": "Mozilla/5.0"}
HEADERS_API = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.espn.com/"
}

def extract_game_id(href: str) -> Optional[str]:
    m = re.search(r"/gameId/(\d+)", href)
    return m.group(1) if m else None

def get_team_year_game_ids(team: str, year: int) -> List[str]:
    url = f"{BASE_URL}/nba/team/schedule/_/name/{team}/season/{year}/seasontype/2"
    r = requests.get(url, headers=HEADERS_HTML, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    links = [
        a["href"] for a in soup.find_all("a", class_="AnchorLink", href=True)
        if "/nba/game/_/gameId/" in a["href"]
    ]
    gids = []
    for lk in links:
        gid = extract_game_id(lk)
        if gid:
            gids.append(gid)
    return gids

def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "—":
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].replace(",", ".")) / 100.0
        except:
            return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return a / b if b else None
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None

def snake(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def get_summary(gid: str) -> Dict[str, Any]:
    url = API_SUMMARY.format(gid=gid)
    r = requests.get(url, headers=HEADERS_API, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_home_away(js: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    teams = js.get("boxscore", {}).get("teams", [])
    out = {}
    for t in teams:
        side = t.get("homeAway")
        if side in ("home", "away"):
            out[side] = t
    if not out:
        comp = js.get("competitions", [{}])[0]
        for c in comp.get("competitors", []):
            key = c.get("homeAway")
            if key in ("home", "away"):
                out[key] = {"team": c.get("team", {}), "score": c.get("score")}
    return out

def flatten_stat_list(stats_list: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for st in stats_list or []:
        name = st.get("name") or st.get("abbreviation") or st.get("displayName")
        if not name:
            continue
        key = snake(name)
        val = st.get("value")
        disp = st.get("displayValue")
        num = to_float(val if val not in (None, "", "0-0") else disp)
        out[key] = num
    return out

def flatten_team(team_obj: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    info = team_obj.get("team", {})
    out[f"{prefix}_team_id"] = info.get("id")
    out[f"{prefix}_team_abbr"] = info.get("abbreviation")
    out[f"{prefix}_team_name"] = info.get("displayName")
    out[f"{prefix}_score"] = to_float(team_obj.get("score"))

    stats1 = flatten_stat_list(team_obj.get("statistics", []))
    for k, v in stats1.items():
        out[f"{prefix}_{k}"] = v

    for grp in team_obj.get("groups", []) or []:
        gname = snake(grp.get("name") or grp.get("displayName") or "")
        gstats = flatten_stat_list(grp.get("statistics", []))
        for k, v in gstats.items():
            key = f"{prefix}_{gname}_{k}" if gname else f"{prefix}_{k}"
            if key not in out:
                out[key] = v

    totals = team_obj.get("totals", {})
    for k, v in totals.items():
        key = f"{prefix}_{snake(k)}"
        if key not in out:
            out[key] = to_float(v)

    return out

def build_row(gid: str) -> Dict[str, Any]:
    js = get_summary(gid)
    comp = js.get("competitions", [{}])[0]
    date = comp.get("date")
    status = comp.get("status", {}).get("type", {}).get("name")

    sides = pick_home_away(js)
    if "home" not in sides or "away" not in sides:
        raise ValueError(f"jogo {gid} sem blocos home e away")

    home = flatten_team(sides["home"], "home")
    away = flatten_team(sides["away"], "away")

    row: Dict[str, Any] = {"game_id": gid, "date": date, "status": status}
    row.update(home)
    row.update(away)

    hs = row.get("home_score")
    as_ = row.get("away_score")
    if hs is not None and as_ is not None:
        row["home_win"] = 1 if float(hs) > float(as_) else 0
        row["point_diff"] = float(hs) - float(as_)
    else:
        row["home_win"] = None
        row["point_diff"] = None
    return row

def main():
    years = list(range(2015, 2025))
    all_gids: List[str] = []
    seen = set()

    for team in TEAMS:
        for year in years:
            try:
                gids = get_team_year_game_ids(team, year)
                for gid in gids:
                    if gid not in seen:
                        seen.add(gid)
                        all_gids.append(gid)
                print(f"time {team} ano {year} ids coletados {len(gids)} total acumulado {len(all_gids)}")
                time.sleep(0.4)
            except Exception as e:
                print(f"falha schedule {team} {year} {e}")

    rows: List[Dict[str, Any]] = []
    for i, gid in enumerate(all_gids, 1):
        try:
            rows.append(build_row(gid))
        except Exception as e:
            print(f"falha resumo {gid} {e}")
        if i % 20 == 0:
            time.sleep(0.6)

    df = pd.DataFrame(rows)
    meta = ["game_id", "date", "status"]
    homes = sorted([c for c in df.columns if c.startswith("home_") and c not in ("home_win",)])
    aways = sorted([c for c in df.columns if c.startswith("away_")])
    tail = [c for c in ["point_diff", "home_win"] if c in df.columns]
    cols = [c for c in meta if c in df.columns] + homes + aways + tail
    df = df[cols]

    df.to_csv("espn_nba_team_stats_dataset.csv", index=False, encoding="utf-8")
    print("salvo espn_nba_team_stats_dataset")

if __name__ == "__main__":
    main()