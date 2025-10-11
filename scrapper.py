import requests
import pandas as pd
from datetime import datetime, timedelta
import time

headers = {
    'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com',
    'x-rapidapi-key': '27244be62dmshb2410b2c636a6a7p18e011jsn345988bae2b4'
}


# Removemos o filtro de times específicos para buscar todos os times
def get_games_by_date(date, max_retries=5):
    url = f"https://api-nba-v1.p.rapidapi.com/games?date={date}"
    backoff = 30
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


# ✅ PERÍODO AMPLIADO: De 2021 até agora
start_date = datetime(2021, 10, 18)  # Começa em 2021
end_date = datetime.now()  # Até a data atual

all_games, all_player_stats = [], []
processed_game_ids = set()
total_games_found = 0

current_date = start_date

print(
    f"Buscando jogos de TODOS os times da NBA de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
print("=" * 60)

while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n📅 Verificando data: {date_str}")

    time.sleep(3)  # Delay entre requisições

    games_data = get_games_by_date(date_str)

    if games_data and games_data.get('results', 0) > 0:
        games_count = games_data['results']
        print(f"  → Encontrados {games_count} jogos nesta data")

        for game in games_data['response']:
            if game['id'] in processed_game_ids:
                continue

            home_code = game['teams']['home']['code']
            away_code = game['teams']['visitors']['code']

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
                'season': game.get('season'),
                'status': game.get('status', {}).get('long')
            }
            all_games.append(game_info)
            processed_game_ids.add(game['id'])
            total_games_found += 1

            print(f"  ✅ Jogo {total_games_found}: {away_code} @ {home_code}")

            time.sleep(3)  # Delay entre requisições de estatísticas

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

    # Status parcial a cada 30 dias
    if (current_date - start_date).days % 30 == 0:
        print(f"\n📊 Status após {((current_date - start_date).days)} dias:")
        print(f"  Total de jogos encontrados: {total_games_found}")
        print(f"  Total de estatísticas de jogadores: {len(all_player_stats)}")

    current_date += timedelta(days=1)

# Salvar resultados
if all_games:
    df_games = pd.DataFrame(all_games)
    df_players = pd.DataFrame(all_player_stats)

    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Salvar em Excel
    df_games.to_excel(f"nba_all_games_2021_present_{ts}.xlsx", index=False)
    df_players.to_excel(f"nba_all_player_stats_2021_present_{ts}.xlsx", index=False)

    # Salvar em CSV também (opcional - arquivos menores)
    df_games.to_csv(f"nba_all_games_2021_present_{ts}.csv", index=False)
    df_players.to_csv(f"nba_all_player_stats_2021_present_{ts}.csv", index=False)

    print("\n✅ Concluído!")
    print(f"📋 Total de jogos únicos: {len(df_games)}")
    print(f"👥 Total de linhas de estatísticas de jogadores: {len(df_players)}")
    print(f"💾 Arquivos salvos com timestamp: {ts}")
else:
    print("\n❌ Nenhum jogo foi encontrado. Verifique as datas e a conexão com a API.")

print(f"\n🎯 Período coberto: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")