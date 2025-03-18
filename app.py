import streamlit as st
import asyncio
from scraper import SimpleAsyncScraper
from agent import AsyncAgent
from utils import validate_ollama_connection, validate_url
import os
import json
import requests
import time

# Set up page configuration
st.set_page_config(
    page_title="Advanced RAG Content Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "urls" not in st.session_state:
    st.session_state.urls = []
if "contents" not in st.session_state:
    st.session_state.contents = {}
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ollama_validated" not in st.session_state:
    st.session_state.ollama_validated = asyncio.run(validate_ollama_connection())

# Sidebar for configuration and settings
with st.sidebar:
    st.title("Configuration")
    
    # Ollama configuration
    ollama_url = st.text_input("Ollama Base URL", 
                            value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), 
                            help="URL for your Ollama server")
    
    if st.button("Check Ollama Connection"):
        with st.spinner("Checking Ollama connection..."):
            os.environ["OLLAMA_BASE_URL"] = ollama_url
            ollama_status = asyncio.run(validate_ollama_connection())
            st.session_state.ollama_validated = ollama_status
            
            if ollama_status:
                st.success("✅ Connected to Ollama successfully!")
            else:
                st.error("❌ Failed to connect to Ollama. Check if it's running.")
    
    # Check model dependencies
    if st.button("Check Model Dependencies"):
        with st.spinner("Checking dependencies..."):
            # Validate Ollama models
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                response = requests.get(f"{ollama_base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if models:
                        mistral_available = any(model.get("name", "").startswith("mistral") for model in models)
                        if mistral_available:
                            st.success("✅ Mistral model is available in Ollama")
                        else:
                            st.error("❌ Mistral model not found. Please run: ollama pull mistral")
                    else:
                        st.error("❌ No models found in Ollama")
            except Exception as e:
                st.error(f"❌ Could not connect to Ollama: {str(e)}")
            
            # Check local model availability
            from model_check import check_model_directory
            if check_model_directory():
                st.success("✅ Local embedding model is available")
            else:
                st.error("❌ Local embedding model not properly set up. Please check the model directory.")
    
    st.divider()
    
    # RAG Configuration options
    st.subheader("RAG Options")
    
    retrieval_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=10, value=5, 
                            help="Higher values increase context but may add noise")
    
    chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1000,
                          help="Size of text chunks for indexing")
    
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=200,
                             help="Overlap between chunks for better context")
    
    rerank_enabled = st.checkbox("Enable Semantic Reranking", value=True,
                                help="Rerank chunks based on semantic similarity")
    
    st.divider()
    
    # Chat history management
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main content area
st.title("Advanced RAG Content Q&A with Mistral + Local Embeddings")

# URL input section
with st.container():
    st.subheader("Step 1: Enter URLs to analyze")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("Enter a URL:", placeholder="https://example.com")
    with col2:
        add_url = st.button("Add URL", use_container_width=True, disabled=not st.session_state.ollama_validated)
    
    if add_url and url_input:
        if not validate_url(url_input):
            st.error("Invalid URL format. Please enter a valid URL.")
        else:
            with st.spinner(f"Scraping content from {url_input}..."):
                try:
                    scraper = SimpleAsyncScraper()
                    content = asyncio.run(scraper.scrape_url(url_input))
                    
                    if content:
                        if url_input not in st.session_state.urls:
                            st.session_state.urls.append(url_input)
                            st.session_state.contents[url_input] = content
                            
                            # Initialize or reinitialize the agent with updated content
                            rag_config = {
                                "retrieval_k": retrieval_k,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "rerank_enabled": rerank_enabled
                            }
                            
                            # Initialize agent
                            if st.session_state.agent is None:
                                st.session_state.agent = AsyncAgent(st.session_state.contents, rag_config)
                            else:
                                # Update agent with new content and config
                                asyncio.run(st.session_state.agent.update_content(st.session_state.contents, rag_config))
                                
                            st.success(f"Successfully added: {url_input}")
                        else:
                            st.warning(f"URL already added: {url_input}")
                    else:
                        st.error(f"Failed to extract content from {url_input}")
                except Exception as e:
                    st.error(f"Error scraping URL: {str(e)}")

# Display added URLs with option to remove
if st.session_state.urls:
    st.subheader("Added URLs:")
    for i, url in enumerate(st.session_state.urls):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(f"{i+1}. {url}")
        with col2:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.urls.pop(i)
                removed_url = url
                st.session_state.contents.pop(url)
                
                if st.session_state.contents:
                    # Update agent with remaining content
                    rag_config = {
                        "retrieval_k": retrieval_k,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "rerank_enabled": rerank_enabled
                    }
                    asyncio.run(st.session_state.agent.update_content(st.session_state.contents, rag_config))
                else:
                    st.session_state.agent = None
                    
                st.success(f"Removed: {removed_url}")
                st.rerun()

# Q&A Section
with st.container():
    st.subheader("Step 2: Ask questions about the content")
    
    if not st.session_state.urls:
        st.warning("Please add at least one URL before asking questions.")
    else:
        question = st.text_area("Enter your question:", placeholder="What is the main topic of these pages?")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            ask_button = st.button("Ask Question", disabled=not st.session_state.agent, use_container_width=True)
        with col2:
            show_details = st.checkbox("Show retrieval details", value=False)
        
        if ask_button and question:
            with st.spinner("Generating answer..."):
                try:
                    start_time = time.time()
                    answer, retrieved_chunks = asyncio.run(st.session_state.agent.answer_question(question))
                    end_time = time.time()
                    processing_time = round(end_time - start_time, 2)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": question, "answer": answer})
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(answer)
                    st.caption(f"Processing time: {processing_time} seconds")
                    
                    # Display retrieval details if enabled
                    if show_details and retrieved_chunks:
                        with st.expander("Retrieved Chunks", expanded=True):
                            st.write(f"Number of chunks retrieved: {len(retrieved_chunks)}")
                            for i, chunk in enumerate(retrieved_chunks):
                                st.markdown(f"**Chunk {i+1}** (Source: {chunk.metadata.get('source', 'Unknown')})")
                                st.markdown(f"<div style='border:1px solid #ddd; padding:10px; margin-bottom:10px; font-size:0.9em; max-height:200px; overflow-y:auto;'>{chunk.page_content}</div>", unsafe_allow_html=True)
                    
                    # Display sources
                    st.subheader("Sources:")
                    for url in st.session_state.urls:
                        st.write(f"- {url}")
                    
                    # Add feedback buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 This answer was helpful", key="helpful"):
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 This answer needs improvement", key="not_helpful"):
                            st.info("Thank you for your feedback! We'll work to improve.")
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            with st.expander("Chat History", expanded=False):
                for i, chat in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q: {chat['question']}**")
                    st.markdown(f"A: {chat['answer']}")
                    if i < len(st.session_state.chat_history) - 1:
                        st.divider()

# Display footer with instructions
st.markdown("---")
st.markdown("""
**Instructions:**
1. Enter one or more URLs to extract content from websites
2. Configure RAG options in the sidebar for optimal retrieval
3. Ask questions about the content of those websites
4. Get answers based ONLY on the information from the provided URLs
""")