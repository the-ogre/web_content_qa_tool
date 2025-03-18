import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, contents):
        """
        Initialize the Agent with web content.
        
        Args:
            contents (dict): Dictionary mapping URLs to their text content
        """
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.contents = contents
        self.vector_store = None
        self.retriever = None
        self.chain = None
        
        # Initialize the agent components
        self._setup_vector_store()
        self._setup_chain()
        
    def _setup_vector_store(self):
        """Set up the vector store with all content"""
        try:
            # Text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Process all documents
            documents = []
            for url, content in self.contents.items():
                chunks = text_splitter.create_documents([content], metadatas=[{"source": url}])
                documents.extend(chunks)
            
            logger.info(f"Created {len(documents)} chunks from {len(self.contents)} URLs")
            
            # Set up embeddings and vector store
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=self.openai_api_key
            )
            
            self.vector_store = FAISS.from_documents(documents, embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            logger.info("Vector store and retriever successfully initialized")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    def _setup_chain(self):
        """Set up the QA chain"""
        try:
            # Create LLM
            llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo-16k",
                openai_api_key=self.openai_api_key
            )
            
            # Set up prompt template
            prompt = ChatPromptTemplate.from_template("""
            You are a helpful question-answering assistant. Your task is to answer the user's question based ONLY on 
            the provided context. If the answer cannot be found in the context, say "I don't have enough information 
            to answer this question based on the provided content." Do not use any external knowledge.
            
            Context:
            {context}
            
            Question:
            {input}
            
            Answer:
            """)
            
            # Create document chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            
            # Create retrieval chain
            self.chain = create_retrieval_chain(self.retriever, question_answer_chain)
            
            logger.info("QA chain successfully initialized")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise
    
    def answer_question(self, question):
        """
        Answer a question based on the web content
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The answer based on the web content
        """
        try:
            if not self.chain:
                raise ValueError("QA chain not initialized. Please check your API key and initialization.")
                
            logger.info(f"Processing question: {question}")
            
            response = self.chain.invoke({"input": question})
            answer = response.get("answer", "Unable to generate an answer.")
            
            logger.info(f"Generated answer for question: {question}")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise