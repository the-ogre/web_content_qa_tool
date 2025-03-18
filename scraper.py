import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from html import unescape

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAsyncScraper:
    """Simple asynchronous web scraper that doesn't use readability."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def scrape_url(self, url):
        """
        Scrape content from a URL asynchronously using BeautifulSoup only
        
        Args:
            url (str): URL to scrape
            
        Returns:
            str: Extracted, cleaned text content
        """
        # Validate URL format
        if not self._validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return None
        
        try:
            logger.info(f"Fetching content from: {url}")
            
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html_content = await response.text()
                    
                    # Extract title and content with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Get page title
                    title = soup.title.string if soup.title else "No Title"
                    
                    # Remove script, style, header, footer, nav elements
                    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'iframe', 'form']):
                        element.decompose()
                    
                    # Extract main content (simple heuristic to get main content)
                    main_content = ""
                    
                    # Try to find main content container
                    main_candidates = soup.find_all(['main', 'article', 'div', 'section'], 
                                                    class_=lambda c: c and ('content' in c.lower() or 
                                                                          'main' in c.lower() or 
                                                                          'article' in c.lower()))
                    
                    if main_candidates:
                        # Use the largest content block
                        main_content = max(main_candidates, key=lambda x: len(x.get_text())).get_text(separator=' ', strip=True)
                    else:
                        # If no main content found, use the body
                        main_content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
                    
                    # Clean the extracted text
                    main_content = self._clean_text(main_content)
                    
                    # Add title at the beginning
                    text = f"Title: {title}\n\n{main_content}"
                    
                    logger.info(f"Successfully extracted {len(text)} characters from {url}")
                    return text
                    
        except aiohttp.ClientError as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    async def scrape_multiple_urls(self, urls):
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls (list): List of URLs to scrape
            
        Returns:
            dict: Dictionary mapping URLs to their content
        """
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        return {url: content for url, content in zip(urls, results) if content is not None}
    
    def _validate_url(self, url):
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _clean_text(self, text):
        """Extract and clean text from HTML content"""
        if not text:
            return ""
            
        # Clean up the text
        text = unescape(text)  # Convert HTML entities
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'(\n\s*)+', '\n\n', text)  # Normalize line breaks
        
        return text.strip()