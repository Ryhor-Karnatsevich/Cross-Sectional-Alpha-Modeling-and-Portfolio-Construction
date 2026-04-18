import requests
from bs4 import BeautifulSoup


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})

    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip().replace('.', '-')
        tickers.append(ticker)

    return tickers


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} tickers")
    print(tickers[:10])
