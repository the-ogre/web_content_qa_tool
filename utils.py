import os
import re
import logging
import aiohttp
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def validate_ollama_connection():
    """
    Validate connection to Ollama server
    
    Returns:
        bool: True if connected, False otherwise
    """
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_base_url}/api/tags", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    if not models:
                        logger.warning("No models found in Ollama")
                        return False
                    
                    # Check if mistral is available
                    mistral_available = any(model.get("name", "").startswith("mistral") for model in models)
                    if not mistral_available:
                        logger.warning("Mistral model not found in Ollama. Run 'ollama pull mistral'")
                        return False
                    
                    logger.info("Successfully connected to Ollama with Mistral model")
                    return True
                else:
                    logger.error(f"Failed to connect to Ollama: HTTP {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama at {ollama_base_url}: {str(e)}")
        return False

def check_stella_model():
    """
    Check if the Stella embedding model is available
    
    Returns:
        bool: True if available, False otherwise
    """
    try:
        # First check if the library is installed
        import sentence_transformers
        
        # Check if model is available in huggingface
        from huggingface_hub import model_info
        
        try:
            # Just check if model info can be retrieved
            _ = model_info("infosumm/stella-en-400m-v5")
            return True
        except Exception as e:
            logger.warning(f"Could not verify Stella model: {str(e)}")
            return False
    except ImportError as e:
        logger.error(f"Missing required packages for Stella model: {str(e)}")
        return False

def validate_url(url):
    """
    Validate URL format
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return re.match(regex, url) is not None

def clean_text(text):
    """
    Clean text by removing excessive whitespace and special characters
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    return text.strip()