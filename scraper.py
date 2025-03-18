import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_url(url):
    """
    Scrape content from a given URL
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        str: The extracted text content from the URL
    """
    # Validate URL format
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            logger.error(f"Invalid URL format: {url}")
            return None
    except Exception as e:
        logger.error(f"URL parsing error: {str(e)}")
        return None
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        # Send HTTP request
        logger.info(f"Fetching content from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
        
        # Extract text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text: remove extra whitespace
        text = ' '.join(text.split())
        
        logger.info(f"Successfully extracted {len(text)} characters from {url}")
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None