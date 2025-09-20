import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

# Configuração da API
headers = {
    'x-rapidapi-host': 'api-nba-v1.p.rapidapi.com',
    'x-rapidapi-key': '27244be62dmshb2410b2c636a6a7p18e011jsn345988bae2b4'
}

# Times que queremos buscar
teams_of_interest = [
    'Oklahoma City Thunder',
    'Minnesota Timberwolves',
    'Indiana Pacers',
    'New York Knicks',
    'Toronto Raptors'
]

# Função para buscar jogos por data com retry
def get_games_by_date(date, max_retries=3):
    url = f"https://api-nba-v1.p.rapidapi.com/games?date={date}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print(f"  ⚠️  Erro na API: Status {response.status_code}")
                if response.status_code == 429:
                    print("  ⚠️  Too Many Requests! Aguardando 60 segundos...")
                    time.sleep(60)
                    continue

            data = response.json()

            if 'errors' in data and data['errors']:
                print(f"  ⚠️  Erro na resposta: {data['errors']}")

            return data

        except Exception as e:
            print(f"  ⚠️  Erro na requisição: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"  Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)

    return None

# Função para buscar estatísticas dos jogadores por jogo
def get_player_stats(game_id, max_retries=3):
    url = f"https://api-nba-v1.p.rapidapi.com/players/statistics?game={game_id}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                print(f"    ⚠️  Erro ao buscar stats: Status {response.status_code}")
                if response.status_code == 429:
                    print("    ⚠️  Too Many Requests! Aguardando 60 segundos...")
                    time.sleep(60)
                    continue

            return response.json()

        except Exception as e:
            print(f"    ⚠️  Erro ao buscar stats: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"    Tentando novamente em {wait_time} segundos...")
                time.sleep(wait_time)

    return None

# Coletar dados
all_games = []
all_player_stats = []
processed_game_ids = set()

# Começar a buscar jogos a partir de outubro de 2024 até junho 2025 (fim da temporada)
start_date = datetime(2024, 10, 22)
end_date = datetime(2025, 6, 30)  # Limite para evitar loop infinito; ajuste se quiser
current_date = start_date

games_found = {}
for team in teams_of_interest:
    games_found[team] = 0

print("Buscando TODOS os jogos dos times de interesse na temporada 2024-2025...")
print("=" * 50)

consecutive_empty_days = 0
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n📅 Verificando data: {date_str}")

    # Delay entre requisições
    time.sleep(2)

    # Buscar jogos da data
    games_data = get_games_by_date(date_str)

    if games_data and 'results' in games_data:
        print(f"  → Encontrados {games_data['results']} jogos nesta data")

        if games_data['results'] > 0:
            consecutive_empty_days = 0

            for game in games_data['response']:
                # Verificar se é da temporada 2024-2025
                if game.get('season') == 2024 and game['id'] not in processed_game_ids:
                    home_team = game['teams']['home']['name']
                    visitor_team = game['teams']['visitors']['name']

                    # Verificar se pelo menos um time de interesse está no jogo
                    teams_in_game = []
                    if home_team in teams_of_interest:
                        teams_in_game.append(home_team)
                    if visitor_team in teams_of_interest:
                        teams_in_game.append(visitor_team)

                    if teams_in_game:
                        # Adicionar informações do jogo
                        game_info = {
                            'game_id': game['id'],
                            'date': game['date']['start'],
                            'home_team': home_team,
                            'visitor_team': visitor_team,
                            'home_score': game['scores']['home']['points'],
                            'visitor_score': game['scores']['visitors']['points'],
                            'arena': game['arena']['name'],
                            'city': game['arena']['city']
                        }
                        all_games.append(game_info)
                        processed_game_ids.add(game['id'])

                        # Atualizar contadores (sem limite)
                        for team in teams_in_game:
                            games_found[team] += 1

                        print(f"  ✅ Jogo encontrado: {visitor_team} @ {home_team}")
                        print(f"     Contando para: {', '.join(teams_in_game)}")

                        # Buscar estatísticas dos jogadores
                        print(f"     Buscando estatísticas dos jogadores...")
                        time.sleep(3)  # Delay para não sobrecarregar

                        player_stats = get_player_stats(game['id'])

                        if player_stats and player_stats['results'] > 0:
                            print(f"     → {player_stats['results']} jogadores encontrados")

                            for player in player_stats['response']:
                                player_info = {
                                    'game_id': game['id'],
                                    'date': game['date']['start'],
                                    'player_id': player['player']['id'],
                                    'player_name': f"{player['player']['firstname']} {player['player']['lastname']}",
                                    'team': player['team']['name'],
                                    'position': player.get('pos', ''),
                                    'minutes': player.get('min', ''),
                                    'points': player.get('points', 0),
                                    'fgm': player.get('fgm', 0),
                                    'fga': player.get('fga', 0),
                                    'fgp': player.get('fgp', ''),
                                    'ftm': player.get('ftm', 0),
                                    'fta': player.get('fta', 0),
                                    'ftp': player.get('ftp', ''),
                                    'tpm': player.get('tpm', 0),
                                    'tpa': player.get('tpa', 0),
                                    'tpp': player.get('tpp', ''),
                                    'offReb': player.get('offReb', 0),
                                    'defReb': player.get('defReb', 0),
                                    'totReb': player.get('totReb', 0),
                                    'assists': player.get('assists', 0),
                                    'steals': player.get('steals', 0),
                                    'blocks': player.get('blocks', 0),
                                    'turnovers': player.get('turnovers', 0),
                                    'pFouls': player.get('pFouls', 0),
                                    'plusMinus': player.get('plusMinus', '')
                                }
                                all_player_stats.append(player_info)
        else:
            consecutive_empty_days += 1
    else:
        print("  ❌ Nenhuma resposta da API ou erro")
        consecutive_empty_days += 1

    # Handling de dias vazios
    if consecutive_empty_days > 7:
        print("\n⚠️  Muitos dias consecutivos sem jogos. Possível problema na API ou fim da temporada.")
        print("Aguardando 30 segundos antes de continuar...")
        time.sleep(30)
        consecutive_empty_days = 0

    # Avançar para o próximo dia
    current_date += timedelta(days=1)

    # Status atual (tracking sem limite)
    print(f"\n📊 Status atual:")
    for team, count in games_found.items():
        print(f"  {team}: {count} jogos")

print("\n" + "=" * 50)
print("RESUMO FINAL:")
print("=" * 50)
for team, count in games_found.items():
    print(f"{team}: {count} jogos")

# Criar DataFrames
df_games = pd.DataFrame(all_games)
df_players = pd.DataFrame(all_player_stats)

# Salvar em Excel com timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
games_filename = f'nba_games_2024_2025_full_{timestamp}.xlsx'
players_filename = f'nba_player_stats_2024_2025_full_{timestamp}.xlsx'

print("\n💾 Salvando arquivos Excel...")
df_games.to_excel(games_filename, index=False)
df_players.to_excel(players_filename, index=False)

print(f"\n✅ Arquivos salvos com sucesso!")
print(f"📋 Arquivo de jogos: {games_filename}")
print(f"👥 Arquivo de stats: {players_filename}")
print(f"📋 Total de jogos únicos: {len(all_games)}")
print(f"👥 Total de estatísticas de jogadores: {len(all_player_stats)}")