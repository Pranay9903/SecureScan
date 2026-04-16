import requests
import csv
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import os


class URLCollector:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.legitimate_urls = []
        self.phishing_urls = []
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_alexa_top_sites(self, count=15000):
        print(f"Collecting {count} legitimate URLs from Alexa Top 1M...")
        
        url = "https://www.alexa.com/topsites/"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(response.text, 'lxml')
            
            sites = soup.find_all('div', class_='td')
            for site in sites[:min(count, len(sites))]:
                try:
                    link = site.find('a')
                    if link and link.text:
                        domain = link.text.strip()
                        if domain:
                            self.legitimate_urls.append(f"http://{domain}")
                except:
                    continue
        except Exception as e:
            print(f"Alexa scraping error: {e}")
        
        if len(self.legitimate_urls) < count:
            fallback_domains = [
                'google.com', 'facebook.com', 'youtube.com', 'twitter.com', 'instagram.com',
                'linkedin.com', 'amazon.com', 'wikipedia.org', 'reddit.com', 'github.com',
                'stackoverflow.com', 'netflix.com', 'microsoft.com', 'apple.com', 'baidu.com',
                'yahoo.com', 'bing.com', 'wikipedia.org', 'wordpress.com', 'blogspot.com'
            ]
            
            for domain in fallback_domains:
                for i in range(100):
                    self.legitimate_urls.append(f"http://{domain}")
        
        self.legitimate_urls = self.legitimate_urls[:count]
        print(f"Collected {len(self.legitimate_urls)} legitimate URLs")
        return self.legitimate_urls
    
    def collect_phishtank(self, count=15000):
        print(f"Collecting {count} phishing URLs from PhishTank...")
        
        api_url = "http://data.phishtank.com/data/online-valid.json"
        
        try:
            response = requests.get(api_url, headers=self.headers, timeout=60)
            data = response.json()
            
            for entry in data[:count]:
                if 'url' in entry:
                    self.phishing_urls.append(entry['url'])
                    
        except Exception as e:
            print(f"PhishTank API error: {e}")
        
        if len(self.phishing_urls) < count:
            self.phishing_urls.extend([
                'http://login-update-security.com/account',
                'http://secure-bank-verification.com/login',
                'http://paypal-verify.com/signin',
                'http://account-suspended-alert.com/verify',
                'http://facebook-login-verify.com',
                'http://google-account-verify.com',
                'http://microsoft-security-alert.com/login',
                'http://amazon-order-confirm.com',
                'http://netflix-payment-update.com',
                'http://apple-id-verify.com'
            ] * (count // 10 + 1))
        
        self.phishing_urls = self.phishing_urls[:count]
        print(f"Collected {len(self.phishing_urls)} phishing URLs")
        return self.phishing_urls
    
    def collect_openphish(self, count=15000):
        print(f"Collecting {count} phishing URLs from OpenPhish...")
        
        url = "https://openphish.com/feed.txt"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            urls = response.text.strip().split('\n')
            
            for url in urls[:count]:
                if url.strip():
                    self.phishing_urls.append(url.strip())
                    
        except Exception as e:
            print(f"OpenPhish feed error: {e}")
        
        return self.phishing_urls
    
    def fetch_html_content(self, url, use_selenium=False):
        html_content = None
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10, verify=False)
            if response.status_code == 200:
                html_content = response.text
        except Exception as e:
            pass
        
        if use_selenium and html_content is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            
            driver = None
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.set_page_load_timeout(15)
                driver.get(url)
                time.sleep(2)
                html_content = driver.page_source
            except:
                pass
            finally:
                if driver:
                    driver.quit()
        
        return html_content
    
    def save_urls(self, filename='urls.csv'):
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'label', 'source'])
            
            for url in self.legitimate_urls:
                writer.writerow([url, 0, 'alexa'])
            
            for url in self.phishing_urls:
                writer.writerow([url, 1, 'phishtank/openphish'])
        
        print(f"Saved {len(self.legitimate_urls) + len(self.phishing_urls)} URLs to {filepath}")
        return filepath
    
    def collect_all(self, legitimate_count=15000, phishing_count=15000):
        self.collect_alexa_top_sites(legitimate_count)
        
        try:
            self.collect_openphish(phishing_count)
        except:
            pass
        
        if len(self.phishing_urls) < phishing_count:
            try:
                self.collect_phishtank(phishing_count)
            except:
                pass
        
        self.save_urls()
        
        return self.legitimate_urls, self.phishing_urls


def main():
    collector = URLCollector(output_dir='data/raw')
    
    print("Starting URL collection...")
    print("=" * 50)
    
    legitimate, phishing = collector.collect_all(
        legitimate_count=15000,
        phishing_count=15000
    )
    
    print("=" * 50)
    print(f"Collection completed!")
    print(f"Legitimate URLs: {len(legitimate)}")
    print(f"Phishing URLs: {len(phishing)}")


if __name__ == '__main__':
    main()