import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
import os


class ESPNScraper:
    def __init__(self):
        self.base_url = "https://www.espn.com/nba/team/depth/_/name/"
        self.teams = {
            'atl': 'Atlanta Hawks', 'bos': 'Boston Celtics', 'bkn': 'Brooklyn Nets',
            'cha': 'Charlotte Hornets', 'chi': 'Chicago Bulls', 'cle': 'Cleveland Cavaliers',
            'dal': 'Dallas Mavericks', 'den': 'Denver Nuggets', 'det': 'Detroit Pistons',
            'gs': 'Golden State Warriors', 'hou': 'Houston Rockets', 'ind': 'Indiana Pacers',
            'lac': 'LA Clippers', 'lal': 'Los Angeles Lakers', 'mem': 'Memphis Grizzlies',
            'mia': 'Miami Heat', 'mil': 'Milwaukee Bucks', 'min': 'Minnesota Timberwolves',
            'no': 'New Orleans Pelicans', 'ny': 'New York Knicks', 'okc': 'Oklahoma City Thunder',
            'orl': 'Orlando Magic', 'phi': 'Philadelphia 76ers', 'phx': 'Phoenix Suns',
            'por': 'Portland Trail Blazers', 'sac': 'Sacramento Kings', 'sa': 'San Antonio Spurs',
            'tor': 'Toronto Raptors', 'utah': 'Utah Jazz', 'wsh': 'Washington Wizards'
        }

    def get_team_depth_chart(self, team_abbr):
        """Pega o depth chart completo de um time"""
        url = f"{self.base_url}{team_abbr}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"❌ Erro HTTP {team_abbr}: {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Estratégia de parsing
            tables = soup.find_all('table', class_='Table')

            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 5:
                    continue

                first_row = rows[0]
                header_cells = first_row.find_all('th')
                header_texts = [cell.get_text(strip=True) for cell in header_cells]

                if any(header in header_texts for header in ['Starter', '2nd', '3rd', '4th', '5th']):
                    return self._parse_depth_from_table(table, team_abbr)

            return None

        except Exception as e:
            print(f"❌ Erro em {team_abbr}: {e}")
            return None

    def _parse_depth_from_table(self, table, team_abbr):
        """Parseia dados de depth de uma tabela identificada"""
        depth_data = {}
        positions = ['PG', 'SG', 'SF', 'PF', 'C']

        rows = table.find_all('tr')[1:]  # Pular linha de cabeçalho

        for i, row in enumerate(rows):
            if i >= len(positions):
                break

            current_pos = positions[i]
            cells = row.find_all('td')
            players = []

            for j, cell in enumerate(cells):
                player_links = cell.find_all('a')
                for player_link in player_links:
                    player_name = player_link.get_text(strip=True)
                    if player_name and player_name not in ['', '-']:
                        # Verificar lesão
                        injury_span = cell.find('span', class_=re.compile(r'injur', re.IGNORECASE))
                        injury_status = injury_span.get_text(strip=True) if injury_span else ''

                        # Normalizar
                        if 'Day-to-Day' in injury_status:
                            injury_status = 'DD'
                        elif 'Out' in injury_status:
                            injury_status = 'O'
                        elif 'Doubtful' in injury_status:
                            injury_status = 'D'

                        players.append({
                            'name': player_name,
                            'status': injury_status,
                            'depth': j + 1
                        })

            if players:
                depth_data[current_pos] = players

        return depth_data

    def get_all_teams_depth_charts(self):
        """Pega depth charts de todos os times"""
        all_teams_data = {}
        successful = 0
        failed = 0

        print("🏀 INICIANDO COLETA DE TODOS OS TIMES NBA")
        print("=" * 60)

        for team_abbr, team_name in self.teams.items():
            print(f"🎯 Coletando {team_name}...")
            depth_chart = self.get_team_depth_chart(team_abbr)

            if depth_chart:
                all_teams_data[team_abbr] = {
                    'team_name': team_name,
                    'depth_chart': depth_chart,
                    'last_updated': datetime.now()
                }
                total_players = sum(len(players) for players in depth_chart.values())
                print(f"✅ {team_name} - {total_players} jogadores")
                successful += 1
            else:
                print(f"❌ Falha em {team_name}")
                failed += 1

            time.sleep(1)  # Respeitar rate limiting

        print(f"\n📊 RESUMO: {successful} sucessos, {failed} falhas")
        return all_teams_data

    def save_to_excel(self, depth_charts_data, filename=None):
        """Salva todos os depth charts em Excel"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nba_depth_charts_{timestamp}.xlsx"

        print(f"\n💾 Salvando em Excel: {filename}")

        # Criar DataFrame principal
        all_data = []

        for team_abbr, team_info in depth_charts_data.items():
            team_name = team_info['team_name']
            depth_chart = team_info['depth_chart']

            for position, players in depth_chart.items():
                for player in players:
                    all_data.append({
                        'Team_Abbr': team_abbr.upper(),
                        'Team_Name': team_name,
                        'Position': position,
                        'Player': player['name'],
                        'Status': player['status'],
                        'Depth': player['depth'],
                        'Scraped_At': team_info['last_updated']
                    })

        # DataFrame principal
        main_df = pd.DataFrame(all_data)

        # Criar abas especiais
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # ABA 1: Todos os dados
            main_df.to_excel(writer, sheet_name='All_Teams', index=False)

            # ABA 2: Starters apenas
            starters_df = main_df[main_df['Depth'] == 1].copy()
            starters_df.to_excel(writer, sheet_name='Starters_Only', index=False)

            # ABA 3: Jogadores lesionados
            injured_df = main_df[main_df['Status'].isin(['DD', 'O', 'D'])].copy()
            injured_df.to_excel(writer, sheet_name='Injured_Players', index=False)

            # ABA 4: Resumo por time
            team_summary = []
            for team_abbr, team_info in depth_charts_data.items():
                depth_chart = team_info['depth_chart']
                total_players = sum(len(players) for players in depth_chart.values())
                injured_players = sum(1 for players in depth_chart.values()
                                      for player in players if player['status'])

                team_summary.append({
                    'Team_Abbr': team_abbr.upper(),
                    'Team_Name': team_info['team_name'],
                    'Total_Players': total_players,
                    'Injured_Players': injured_players,
                    'Starter_Quality': calculate_starter_quality(depth_chart),
                    'Team_Strength': calculate_team_strength(depth_chart)
                })

            summary_df = pd.DataFrame(team_summary)
            summary_df.to_excel(writer, sheet_name='Team_Summary', index=False)

        print(f"✅ Excel salvo com sucesso: {filename}")
        print(f"   📊 {len(main_df)} registros totais")
        print(f"   🏀 {len(starters_df)} starters")
        print(f"   🤕 {len(injured_df)} jogadores lesionados")
        print(f"   📋 {len(summary_df)} times resumidos")

        return filename


# FUNÇÕES AUXILIARES (já testadas e funcionando)
def count_injured_players(depth_chart):
    injured_count = 0
    for pos, players in depth_chart.items():
        for player in players:
            if player['status'] and player['status'] in ['DD', 'O', 'D']:
                injured_count += 1
    return injured_count


def get_player_status(depth_chart, position):
    if position in depth_chart and len(depth_chart[position]) > 0:
        starter = depth_chart[position][0]
        return starter['status']
    return ''


def calculate_starter_quality(depth_chart):
    quality_score = 0
    starter_count = 0
    for pos, players in depth_chart.items():
        if players and len(players) > 0:
            starter = players[0]
            base_score = 2 if not starter['status'] else 1
            quality_score += base_score
            starter_count += 1
    if starter_count > 0:
        return quality_score / starter_count
    return 0


def calculate_team_strength(depth_chart):
    strength_score = 0
    position_weights = {'PG': 1.2, 'SG': 1.0, 'SF': 1.0, 'PF': 1.1, 'C': 1.1}
    for pos, players in depth_chart.items():
        weight = position_weights.get(pos, 1.0)
        for i, player in enumerate(players):
            player_score = (3 - i) * weight
            if player['status']:
                if player['status'] == 'O':
                    player_score *= 0.1
                elif player['status'] == 'DD':
                    player_score *= 0.5
                elif player['status'] == 'D':
                    player_score *= 0.3
            strength_score += player_score
    return strength_score


# SISTEMA DE EXECUÇÃO DIÁRIA
def daily_nba_scraping():
    """Função principal para execução diária"""
    print("🚀 INICIANDO SCRAPING DIÁRIO NBA")
    print("=" * 60)

    # Iniciar scraper
    scraper = ESPNScraper()

    # Coletar todos os depth charts
    depth_charts = scraper.get_all_teams_depth_charts()

    if not depth_charts:
        print("❌ Nenhum dado coletado - verifique conexão")
        return None

    # Salvar em Excel
    filename = scraper.save_to_excel(depth_charts)

    # Gerar relatório rápido
    generate_quick_report(depth_charts)

    return filename


def generate_quick_report(depth_charts):
    """Gera um relatório rápido no console"""
    print("\n📈 RELATÓRIO RÁPIDO NBA")
    print("=" * 50)

    total_teams = len(depth_charts)
    total_players = sum(sum(len(players) for players in team_data['depth_chart'].values())
                        for team_data in depth_charts.values())
    total_injured = sum(count_injured_players(team_data['depth_chart'])
                        for team_data in depth_charts.values())

    print(f"🏀 Times processados: {total_teams}/30")
    print(f"👤 Total de jogadores: {total_players}")
    print(f"🤕 Jogadores lesionados: {total_injured}")
    print(f"📊 Taxa de lesões: {(total_injured / total_players) * 100:.1f}%")

    # Top 5 times mais fortes
    team_strengths = []
    for team_abbr, team_data in depth_charts.items():
        strength = calculate_team_strength(team_data['depth_chart'])
        team_strengths.append((team_data['team_name'], strength))

    team_strengths.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 TOP 5 TIMES MAIS FORTES:")
    for i, (team, strength) in enumerate(team_strengths[:5], 1):
        print(f"   {i}. {team}: {strength:.1f}")

    # Times com mais lesões
    team_injuries = []
    for team_abbr, team_data in depth_charts.items():
        injuries = count_injured_players(team_data['depth_chart'])
        team_injuries.append((team_data['team_name'], injuries))

    team_injuries.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🤕 TOP 5 TIMES COM MAIS LESÕES:")
    for i, (team, injuries) in enumerate(team_injuries[:5], 1):
        print(f"   {i}. {team}: {injuries} lesionados")


# EXECUÇÃO AUTOMÁTICA
if __name__ == "__main__":
    print("🎯 SISTEMA DE SCRAPING NBA - EXECUÇÃO DIÁRIA")
    print("⭐ Coleção automática de Depth Charts da ESPN")
    print("=" * 60)

    # Executar scraping completo
    result_file = daily_nba_scraping()

    if result_file:
        print(f"\n✅ SCRAPING CONCLUÍDO!")
        print(f"📁 Arquivo salvo: {result_file}")
        print(f"🕐 Próxima execução: Amanhã mesmo horário")
        print(f"💡 Dica: Agende este script para rodar automaticamente todo dia!")
    else:
        print(f"\n❌ SCRAPING FALHOU - Tente novamente mais tarde")