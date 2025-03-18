import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from embedding import LocalEmbeddings
from reranker import SemanticReranker
from query_expansion import QueryExpander
from prompts import get_answer_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncAgent:
    """Asynchronous RAG agent using advanced retrieval techniques"""
    
    def __init__(self, contents: Dict[str, str], rag_config: Dict[str, Any] = None):
        """
        Initialize the agent with content and configuration
        
        Args:
            contents (Dict[str, str]): Dictionary mapping URLs to their text content
            rag_config (Dict[str, Any], optional): RAG configuration parameters
        """
        self.contents = contents
        self.rag_config = rag_config or {
            "retrieval_k": 5,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "rerank_enabled": True
        }
        
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embeddings = None
        self.vector_store = None
        self.reranker = None
        self.query_expander = None
        
        # Asyncio locks to prevent concurrent access to shared resources
        self.init_lock = asyncio.Lock()
        
        # Track initialization status
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize agent components asynchronously"""
        if self.is_initialized:
            return
            
        async with self.init_lock:
            if self.is_initialized:
                return
                
            try:
                logger.info("Initializing embeddings model...")
                self.embeddings = LocalEmbeddings()
                
                logger.info("Initializing vector store...")
                await self._setup_vector_store()
                
                logger.info("Initializing reranker...")
                self.reranker = SemanticReranker(ollama_base_url=self.ollama_base_url)
                
                logger.info("Initializing query expander...")
                self.query_expander = QueryExpander(ollama_base_url=self.ollama_base_url)
                
                self.is_initialized = True
                logger.info("Agent initialization complete")
                
            except Exception as e:
                logger.error(f"Error during agent initialization: {str(e)}")
                raise
    
    async def _setup_vector_store(self):
        """Set up vector store with content"""
        try:
            # Text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.rag_config["chunk_size"],
                chunk_overlap=self.rag_config["chunk_overlap"]
            )
            
            # Process all documents
            documents = []
            for url, content in self.contents.items():
                chunks = text_splitter.create_documents([content], metadatas=[{"source": url}])
                documents.extend(chunks)
            
            logger.info(f"Created {len(documents)} chunks from {len(self.contents)} URLs")
            
            # Get embeddings for all documents
            texts = [doc.page_content for doc in documents]
            embeddings_list = await self.embeddings.aembed_documents(texts)
            
            # Create FAISS index
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings_list)),
                embedding=self.embeddings,
                metadatas=[doc.metadata for doc in documents]
            )
            
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    async def update_content(self, contents: Dict[str, str], rag_config: Dict[str, Any] = None):
        """
        Update content and reinitialize vector store
        
        Args:
            contents (Dict[str, str]): Dictionary mapping URLs to their text content
            rag_config (Dict[str, Any], optional): RAG configuration parameters
        """
        self.contents = contents
        if rag_config:
            self.rag_config = rag_config
            
        self.is_initialized = False
        await self.initialize()
    
    async def _retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query (str): User query
            
        Returns:
            List[Document]: Retrieved documents
        """
        try:
            # Embed the query
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Retrieve documents
            k = self.rag_config["retrieval_k"]
            docs = self.vector_store.similarity_search_by_vector(
                query_embedding, k=k
            )
            
            # Extract just the documents
            docs = [doc for doc, score in docs_and_scores]
            
            # Rerank if enabled
            if self.rag_config["rerank_enabled"] and self.reranker is not None:
                docs = await self.reranker.rerank(query, docs)
            
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def answer_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Answer a question using advanced RAG techniques
        
        Args:
            question (str): The question to answer
            
        Returns:
            Tuple[str, List[Document]]: The answer and retrieved documents
        """
        # Make sure the agent is initialized
        await self.initialize()
        
        try:
            # Expand the query if query expander is available
            if self.query_expander:
                query_variations = await self.query_expander.generate_query_variations(question)
                logger.info(f"Generated {len(query_variations)} query variations")
            else:
                query_variations = [question]
            
            # Retrieve documents for each query variation
            all_docs = []
            for query in query_variations:
                docs = await self._retrieve_documents(query)
                all_docs.extend(docs)
            
            # Remove duplicates while preserving order
            unique_docs = []
            seen_contents = set()
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
            
            # Limit to top K documents to avoid context overflow
            top_docs = unique_docs[:self.rag_config["retrieval_k"]]
            
            logger.info(f"Retrieved {len(top_docs)} unique documents")
            
            # Initialize LLM
            llm = Ollama(model="mistral", base_url=self.ollama_base_url)
            
            # Get answer prompt
            prompt = get_answer_prompt()
            
            # Create LCEL chain
            rag_chain = (
                {"context": lambda x: x, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = await rag_chain.ainvoke(top_docs)
            
            logger.info(f"Generated answer of length {len(answer)}")
            return answer, top_docs
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            error_msg = f"An error occurred while generating the answer: {str(e)}"
            return error_msg, []