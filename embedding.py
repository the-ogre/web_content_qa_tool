import logging
import asyncio
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from langchain_core.embeddings import Embeddings

# Add these imports for local embedding model
from config import SearchConfig, setup_logger
from background_task_manager import BackgroundTaskManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalEmbeddingModel:
    """Local embedding model using the model structure from embedding_generator.py"""
    
    def __init__(self):
        """Initialize the local embedding model"""
        self.search_config = SearchConfig()
        self.logger = setup_logger(__name__, 'embedding.log')
        self.device = torch.device('cpu')
        
        self.model = None
        self.tokenizer = None
        self.vector_linear = None
        
        self.embedding_cache = {}
        self.cache_lock = asyncio.Lock()
        self.background_manager = BackgroundTaskManager()
        
        self.query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
        
        # Initialize model components
        self._initialize_model()
        self.logger.info("Running on CPU device")
        
    def _initialize_model(self):
        """Initialize the model with CPU-specific settings."""
        try:
            # Check if model directory exists
            if not os.path.exists(self.search_config.model_dir):
                self.logger.error(f"Model directory not found: {self.search_config.model_dir}")
                raise FileNotFoundError(f"Model directory not found: {self.search_config.model_dir}")
                
            # Initialize model
            self.model = AutoModel.from_pretrained(
                self.search_config.model_dir,
                trust_remote_code=True,
                use_memory_efficient_attention=False,
                unpad_inputs=False
            ).to(self.device).eval()
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.search_config.model_dir,
                trust_remote_code=True
            )
            
            # Initialize vector linear layer
            vector_linear_directory = f"2_Dense_{self.search_config.vector_dim}"
            self.vector_linear = torch.nn.Linear(
                in_features=self.model.config.hidden_size,
                out_features=self.search_config.vector_dim
            )
            
            vector_linear_path = os.path.join(
                self.search_config.model_dir,
                f"{vector_linear_directory}/pytorch_model.bin"
            )
            
            vector_linear_dict = {
                k.replace("linear.", ""): v 
                for k, v in torch.load(
                    vector_linear_path, 
                    map_location=self.device
                ).items()
            }
            self.vector_linear.load_state_dict(vector_linear_dict)
            self.vector_linear.to(self.device)
            
            self.logger.info("Model initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model initialization: {str(e)}")
            raise
            
    async def generate_embeddings_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            self.logger.debug(f"Starting batch embedding for {len(texts)} texts")
            
            if not texts:
                return np.array([])
                
            if is_query:
                texts = [self.query_prompt + text for text in texts]
                
            with torch.no_grad():
                # Log tokenization input
                self.logger.debug(f"Text sample for tokenization: {texts[0][:100]}...")
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                self.logger.debug(f"Tokenized input shape: {inputs['input_ids'].shape}")
                
                attention_mask = inputs["attention_mask"]
                last_hidden_state = self.model(**inputs)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                vectors = self.vector_linear(vectors)
                
                vectors_np = vectors.numpy()
                normalized_vectors = normalize(vectors_np)
                self.logger.debug(f"Final vectors shape: {normalized_vectors.shape}")
                
                return normalized_vectors
                    
        except Exception as e:
            self.logger.error(f"Batch embedding error: {str(e)}", exc_info=True)
            raise
            
    async def generate_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            self.logger.debug(f"Starting embedding generation for text length: {len(text)}")
            vectors = await self.generate_embeddings_batch([text], is_query=is_query)
            return vectors[0]
        except Exception as e:
            self.logger.error(f"Error generating single embedding: {str(e)}", exc_info=True)
            raise

class LocalEmbeddings(Embeddings):
    """
    Wrapper for local embedding model to use with LangChain
    """
    
    def __init__(self):
        """Initialize the local embedding model"""
        self.model = LocalEmbeddingModel()
        self.lock = asyncio.Lock()
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents asynchronously"""
        try:
            if not texts:
                return []
                
            vectors = await self.model.generate_embeddings_batch(texts, is_query=False)
            return vectors.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    async def aembed_query(self, text: str) -> List[float]:
        """Embed a query asynchronously"""
        try:
            if not text:
                return []
                
            vector = await self.model.generate_embedding(text, is_query=True)
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    # Synchronous methods required by LangChain
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous version of embed_documents (required by LangChain)
        """
        return asyncio.run(self.aembed_documents(texts))
    
    def embed_query(self, text: str) -> List[float]:
        """
        Synchronous version of embed_query (required by LangChain)
        """
        return asyncio.run(self.aembed_query(text))