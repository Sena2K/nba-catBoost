import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup

BASE_URL = "https://www.espn.com"

TEAMS = [
    'ATL','BOS','NJN','CHA','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM',
    'MIA','MIL','MIN','NOH','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','UTA','WAS'
]

def main():
    headers = {"User-Agent": "Mozilla/5.0"}
    year = 2015
    team = "ATL"

    url = f"{BASE_URL}/nba/team/schedule/_/name/{team}/season/{year}/seasontype/2"
    print("Buscando:", url)

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    # Usar BeautifulSoup para capturar os <a>
    soup = BeautifulSoup(resp.text, "lxml")

    # todos os links que levam para "game/_/gameId/"
    links = [
        a["href"] for a in soup.find_all("a", class_="AnchorLink", href=True)
        if "/nba/game/_/gameId/" in a["href"]
    ]

    print("Links de jogos encontrados:")
    for link in links:
        print(link)
        
        df_lista = pd.read_html(link)
        print(f"Tabelas encontradas: {len(df_lista)}")
        print(df_lista[0].head())

    


if __name__ == "__main__":
    main()
