import requests
import pandas as pd
from datetime import datetime, timedelta
import time

headers = {
    'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com',
    'x-rapidapi-key': '27244be62dmshb2410b2c636a6a7p18e011jsn345988bae2b4'
}

team_codes_of_interest = {'BOS', 'CLE', 'HOU', 'IND', 'LAL', 'PHI', 'LAC', 'MIN', 'NOP', 'OKC'}


def get_games_by_date(date, max_retries=5):  # Aumentei as retentativas
    url = f"https://api-nba-v1.p.rapidapi.com/games?date={date}"
    backoff = 30  # Aumentei o backoff inicial
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"  ⚠️  Too Many Requests (429). Aguardando {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"  ⚠️  Erro na API: Status {r.status_code}")
                return None
            return r.json()
        except Exception as e:
            print(f"  ⚠️  Erro na requisição: {e}")
            if attempt < max_retries - 1:
                wait = backoff * (attempt + 1)
                print(f"  Tentando novamente em {wait}s...")
                time.sleep(wait)
    return None


def get_player_stats(game_id, max_retries=5):
    url = f"https://api-nba-v1.p.rapidapi.com/players/statistics?game={game_id}"
    backoff = 30
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = backoff * (attempt + 1)
                print(f"    ⚠️  429 em player stats. Aguardando {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"    ⚠️  Erro ao buscar stats: Status {r.status_code}")
                return None
            return r.json()
        except Exception as e:
            print(f"    ⚠️  Erro ao buscar stats: {e}")
            if attempt < max_retries - 1:
                wait = backoff * (attempt + 1)
                print(f"    Tentando novamente em {wait}s...")
                time.sleep(wait)
    return None


all_games, all_player_stats = [], []
processed_game_ids = set()

# ✅ DATAS CORRIGIDAS: Temporada 2023-24 começou em 24 de outubro
start_date = datetime(2023, 10, 24)  # Corrigido: primeiro dia da temporada
end_date = datetime(2024, 4, 16)

current_date = start_date
games_found = {code: 0 for code in team_codes_of_interest}

print("Buscando jogos da temporada 2023-24 (outubro/novembro)...")
print("=" * 50)

consecutive_empty_days = 0
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n📅 Verificando data: {date_str}")

    # ✅ Delay maior entre requisições de datas
    time.sleep(3)  # Aumentei para 3 segundos

    games_data = get_games_by_date(date_str)

    if games_data and games_data.get('results', 0) > 0:
        print(f"  → Encontrados {games_data['results']} jogos nesta data")
        consecutive_empty_days = 0

        for game in games_data['response']:
            if game['id'] in processed_game_ids:
                continue

            home_code = game['teams']['home']['code']
            away_code = game['teams']['visitors']['code']

            # Filtra apenas jogos com times de interesse
            if not ({home_code, away_code} & team_codes_of_interest):
                continue

            game_info = {
                'game_id': game['id'],
                'date': game['date']['start'],
                'home_team_code': home_code,
                'away_team_code': away_code,
                'home_team_name': game['teams']['home']['name'],
                'away_team_name': game['teams']['visitors']['name'],
                'home_score': game['scores']['home']['points'],
                'away_score': game['scores']['visitors']['points'],
                'arena': (game.get('arena') or {}).get('name'),
                'city': (game.get('arena') or {}).get('city'),
                'season': game.get('season')
            }
            all_games.append(game_info)
            processed_game_ids.add(game['id'])

            for code in ({home_code, away_code} & team_codes_of_interest):
                games_found[code] += 1

            print(f"  ✅ Jogo: {away_code} @ {home_code}")

            # ✅ Delay maior entre requisições de estatísticas
            time.sleep(3)

            print(f"     Buscando estatísticas dos jogadores...")
            player_stats = get_player_stats(game['id'])

            if player_stats and player_stats.get('results', 0) > 0:
                print(f"     → {player_stats['results']} entradas de jogadores")
                for p in player_stats['response']:
                    all_player_stats.append({
                        'game_id': game['id'],
                        'date': game['date']['start'],
                        'team_code': p['team']['code'],
                        'player_id': p['player']['id'],
                        'player_name': f"{p['player']['firstname']} {p['player']['lastname']}",
                        'pos': p.get('pos', ''),
                        'min': p.get('min', ''),
                        'points': p.get('points', 0),
                        'fgm': p.get('fgm', 0), 'fga': p.get('fga', 0), 'fgp': p.get('fgp', ''),
                        'ftm': p.get('ftm', 0), 'fta': p.get('fta', 0), 'ftp': p.get('ftp', ''),
                        'tpm': p.get('tpm', 0), 'tpa': p.get('tpa', 0), 'tpp': p.get('tpp', ''),
                        'offReb': p.get('offReb', 0), 'defReb': p.get('defReb', 0), 'totReb': p.get('totReb', 0),
                        'assists': p.get('assists', 0), 'steals': p.get('steals', 0), 'blocks': p.get('blocks', 0),
                        'turnovers': p.get('turnovers', 0), 'pFouls': p.get('pFouls', 0),
                        'plusMinus': p.get('plusMinus', '')
                    })
            else:
                print("     ❌ Não foi possível obter estatísticas")
    else:
        print("  ❌ Sem jogos ou erro na API")
        consecutive_empty_days += 1

    # Status parcial
    print("\n📊 Status atual:")
    for code in sorted(team_codes_of_interest):
        print(f"  {code}: {games_found[code]} jogos")

    current_date += timedelta(days=1)

# Salvar resultados
if all_games:
    df_games = pd.DataFrame(all_games)
    df_players = pd.DataFrame(all_player_stats)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df_games.to_excel(f"nba_games_{ts}.xlsx", index=False)
    df_players.to_excel(f"nba_player_stats_{ts}.xlsx", index=False)

    print("\n✅ Concluído.")
    print(f"📋 Total de jogos únicos: {len(df_games)}")
    print(f"👥 Total de linhas de estatísticas de jogadores: {len(df_players)}")
else:
    print("\n❌ Nenhum jogo foi encontrado. Verifique as datas e a conexão com a API.")