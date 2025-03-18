import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from prompts import get_query_expansion_prompt, get_hypothetical_document_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Query expansion for improved retrieval using various techniques
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the query expander
        
        Args:
            ollama_base_url (str): Base URL for Ollama API
        """
        self.ollama_base_url = ollama_base_url
        self.expansion_prompt = get_query_expansion_prompt()
        self.hypothetical_doc_prompt = get_hypothetical_document_prompt()
        
    async def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate multiple variations of the query to improve recall
        
        Args:
            query (str): Original user query
            
        Returns:
            List[str]: List of query variations
        """
        try:
            # Initialize LLM
            llm = Ollama(model="mistral", base_url=self.ollama_base_url)
            
            # Get query expansions
            response = await llm.ainvoke(self.expansion_prompt.format(query=query))
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if not json_match:
                logger.warning("Could not find JSON in query expansion response. Using original query only.")
                return [query]
                
            json_str = json_match.group(0)
            
            # Parse the JSON
            try:
                expansion_results = json.loads(json_str)
                variations = expansion_results.get("expansions", [])
                
                # Add original query if not already present
                if query not in variations:
                    variations = [query] + variations
                    
                logger.info(f"Generated {len(variations)} query variations")
                return variations
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing query expansion JSON: {str(e)}")
                return [query]
                
        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}")
            return [query]
    
    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query
        
        Args:
            query (str): User query
            
        Returns:
            str: Hypothetical document text
        """
        try:
            # Initialize LLM
            llm = Ollama(model="mistral", base_url=self.ollama_base_url)
            
            # Generate hypothetical document
            hypothetical_doc = await llm.ainvoke(self.hypothetical_doc_prompt.format(query=query))
            
            logger.info(f"Generated hypothetical document of length {len(hypothetical_doc)}")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            return ""