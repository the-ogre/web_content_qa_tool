import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from prompts import get_reranking_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Semantic reranker for retrieved documents using LLM
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the semantic reranker
        
        Args:
            ollama_base_url (str): Base URL for Ollama API
        """
        self.ollama_base_url = ollama_base_url
        self.prompt = get_reranking_prompt()
        
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on semantic relevance to query
        
        Args:
            query (str): User query
            documents (List[Document]): Documents to rerank
            
        Returns:
            List[Document]: Reranked documents
        """
        if not documents:
            return []
            
        try:
            # Create prompt with documents
            docs_with_ids = []
            for i, doc in enumerate(documents):
                # Truncate document content for the prompt
                content = doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else "")
                docs_with_ids.append({
                    "chunk_id": i,
                    "content": content,
                    "source": doc.metadata.get("source", "Unknown")
                })
                
            # Format documents for the prompt
            formatted_docs = "\n\n".join([
                f"Document {doc['chunk_id']}:\nSource: {doc['source']}\n{doc['content']}"
                for doc in docs_with_ids
            ])
            
            # Initialize LLM
            llm = Ollama(model="mistral", base_url=self.ollama_base_url)
            
            # Call LLM to rerank documents
            reranking_input = {
                "query": query,
                "documents": formatted_docs
            }
            
            # Get reranking scores
            rerank_response = await llm.ainvoke(self.prompt.format(query=query))
            
            # Extract JSON from response
            # Find JSON in the response which might be embedded in other text
            import re
            json_match = re.search(r'\[.*\]', rerank_response, re.DOTALL)
            
            if not json_match:
                logger.warning("Could not find JSON in reranker response. Using original order.")
                return documents
                
            json_str = json_match.group(0)
            
            # Parse the JSON
            try:
                rerank_results = json.loads(json_str)
                
                # Sort documents by relevance score
                id_to_score = {item["chunk_id"]: item["relevance_score"] for item in rerank_results}
                
                # Create a list of (document, score) pairs
                doc_score_pairs = [(doc, id_to_score.get(i, 0)) 
                                  for i, doc in enumerate(documents)]
                
                # Sort by score in descending order
                sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
                
                # Extract just the documents
                reranked_docs = [doc for doc, _ in sorted_pairs]
                
                logger.info(f"Reranked {len(documents)} documents successfully")
                return reranked_docs
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing reranker JSON response: {str(e)}")
                logger.error(f"Raw response: {rerank_response}")
                return documents
                
        except Exception as e:
            logger.error(f"Error in semantic reranking: {str(e)}")
            # Fall back to original order
            return documents