import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def get_nba_injuries_simple():

    url = "https://www.espn.com/nba/injuries"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    try:
        time.sleep(1)

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"Status Code: {response.status_code}")
        print(f"Encoding: {response.encoding}")

        if not response.content:
            print("No content received")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, 'html.parser')
        injuries = []

        tables = soup.find_all('div', class_='ResponsiveTable')
        print(f"Found {len(tables)} ResponsiveTable divs")

        injury_tables = soup.find_all('div', class_='Table__league-injuries')
        print(f"Found {len(injury_tables)} Table__league-injuries divs")

        all_tables = soup.find_all('table')
        print(f"Found {len(all_tables)} total tables")

        target_tables = injury_tables if injury_tables else tables

        for i, table in enumerate(target_tables):
            print(f"Processing table {i + 1}")

            team_name = None
            team_selectors = [
                'span.injuries__teamName',
                '.Table__Title span',
                '.injuries__teamName',
                '.flex-row span'
            ]

            for selector in team_selectors:
                team_element = table.select_one(selector)
                if team_element:
                    team_name = team_element.text.strip()
                    break

            if not team_name:
                team_name = f"Unknown_Team_{i}"

            print(f"Team: {team_name}")

            tbody = table.find('tbody')
            if not tbody:
                tbody = table.find('tbody', class_='Table__TBODY')
                if not tbody:
                    tbody = table.find('tbody')

            if tbody:
                rows = tbody.find_all('tr')
                print(f"Found {len(rows)} rows for {team_name}")

                for row_idx, row in enumerate(rows):
                    cols = row.find_all('td')
                    if len(cols) < 5:
                        cols = row.find_all('td', class_='Table__TD')

                    if len(cols) >= 5:
                        try:
                            player_name = cols[0].text.strip()
                            position = cols[1].text.strip()
                            return_date = cols[2].text.strip()
                            status = cols[3].text.strip()
                            comment = cols[4].text.strip()

                            injury_data = {
                                'Team': team_name,
                                'Player': player_name,
                                'Position': position,
                                'Return_Date': return_date,
                                'Status': status,
                                'Comment': comment,
                                'Scraped_At': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            injuries.append(injury_data)
                            print(f"  - Added: {player_name}")

                        except Exception as e:
                            print(f"  - Error processing row {row_idx}: {e}")
                            continue
                    else:
                        print(f"  - Row {row_idx} has only {len(cols)} columns")
            else:
                print(f"  - No tbody found for {team_name}")

        print(f"\nTotal injuries collected: {len(injuries)}")
        return pd.DataFrame(injuries)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


def get_nba_injuries_with_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    })

    try:
        response = session.get("https://www.espn.com/nba/injuries", timeout=10)
        response.raise_for_status()

        with open('debug_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Saved page content to debug_page.html for inspection")

        return get_nba_injuries_simple()
    except Exception as e:
        print(f"Session Error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Attempting to scrape NBA injuries...")

    df = get_nba_injuries_simple()

    if df.empty:
        print("\nTrying with session...")
        df = get_nba_injuries_with_session()

    if not df.empty:
        print("\nNBA Injuries Data:")
        print(f"Total records: {len(df)}")
        print("\nFirst 10 records:")
        print(df.head(10))

        # Salvar dados
        df.to_csv('nba_injuries_quick.csv', index=False)
        print(f"\nData saved to 'nba_injuries_quick.csv'")

        # Estatísticas
        print(f"\nInjuries by team:")
        print(df['Team'].value_counts())
        print(f"\nInjuries by status:")
        print(df['Status'].value_counts())

    else:
        print("Failed to retrieve injury data")
        print("Please check:")
        print("1. Your internet connection")
        print("2. If ESPN website is accessible")
        print("3. The debug_page.html file for content inspection")