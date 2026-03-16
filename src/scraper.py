import pandas as pd
import newspaper
from newspaper import Config
import time

# Standard config to bypass basic bot-blockers
config = Config()
config.request_timeout = 10     
config.fetch_images = False      
config.memoize_articles = False  
config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'

def scrape_single_article(url):
    """Scrapes a single article URL. Perfect for the Streamlit UI input box."""
    try:
        article = newspaper.Article(url, config=config)
        article.download()
        article.parse()
        article.nlp()
        
        headline = article.title
        short_description = article.summary
        if not short_description:
            short_description = " ".join(article.text.split('.')[:2]) + "."
            
        return pd.DataFrame([{"headline": headline, "short_description": short_description.strip()}])
    except Exception as e:
        print(f"Error scraping article: {e}")
        return pd.DataFrame()

def scrape_news_homepage(url, max_articles=5):
    """Scrapes multiple articles from a homepage for the Analytics Dashboard."""
    print(f"Scanning homepage: {url} ...")
    
    try:
        news_source = newspaper.build(url, config=config)
    except Exception as e:
        print(f"Error building source from {url}: {e}")
        return pd.DataFrame()
        
    articles_data = []
    count = 0
    
    print(f"Found {len(news_source.articles)} potential article links. Starting extraction...")
    
    for article in news_source.articles:
        if count >= max_articles:
            break
            
        try:
            time.sleep(1) # Be polite to the server
            article.download()
            article.parse()
            article.nlp()
            
            headline = article.title
            short_description = article.summary
            
            if not short_description:
                short_description = " ".join(article.text.split('.')[:2]) + "."
                
            if headline and short_description and len(short_description) > 10:
                articles_data.append({
                    "headline": headline,
                    "short_description": short_description.strip()
                })
                count += 1
                print(f"Scraped [{count}/{max_articles}]: {headline}")
                
        except Exception:
            continue
            
    return pd.DataFrame(articles_data, columns=["headline", "short_description"])

if __name__ == "__main__":
    # 1. Test Single Article Scrape
    single_url = "https://www.theguardian.com/technology/2024/mar/14/apple-buys-ai-startup-darwinai"
    print(f"--- Testing Single Article: {single_url} ---")
    single_df = scrape_single_article(single_url)
    print(single_df)
    print("\n")

    # 2. Test Homepage Scrape
    home_url = "https://www.theguardian.com/international" 
    print(f"--- Testing Homepage Scraper: {home_url} ---")
    scraped_df = scrape_news_homepage(home_url, max_articles=3)
    
    print("\n--- Scraping Complete ---")
    if not scraped_df.empty:
        print(scraped_df)
    else:
        print("No articles were successfully scraped.")