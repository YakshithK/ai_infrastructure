# scripts / scrape_articles.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
from rich import print

urls = [
    "https://finextra.com/newsarticle/12345/example1",
    "https://finextra.com/newsarticle/67890/example2",
    "https://medium.com/@parsaahm/i-failed-ae8be3087ada"
]

def scrape(url):
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    text = soup.get_text(separator=" ", strip=True)
    return text

data = []
for url in urls:
    print(f"[cyan]Scraping[/cyan] {url}")
    text = scrape(url)
    data.append({"url": url, "text": text})

df = pd.DataFrame(data)
df.to_csv("data/raw/raw_articles.csv", index=False)
print("[green]Scraping completed! Articles saved to data/raw/raw_articles.csv[/green]")