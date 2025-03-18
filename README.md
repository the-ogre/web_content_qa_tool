# Web Content Q&A Tool

This web-based tool allows users to:
1. Input one or more URLs to ingest their content
2. Ask questions about that content
3. View concise, accurate answers using only the ingested information

## Features

- Simple, clean UI built with Streamlit
- URL content scraping with BeautifulSoup
- Semantic search and retrieval using LangChain and FAISS
- OpenAI-powered Q&A system that answers based only on the provided content

## Setup Instructions

### Prerequisites
- Python 3.9+ installed
- OpenAI API key

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/web-content-qa-tool.git
cd web-content-qa-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in your terminal (typically http://localhost:8501)

## Usage

1. Enter a URL in the input field and click "Add URL"
2. Add as many URLs as needed
3. Type your question in the text area
4. Click "Ask Question" to get an answer based only on the content from the provided URLs

## Project Structure

- `app.py`: Main Streamlit application
- `scraper.py`: Web content scraping functionality
- `agent.py`: Q&A agent using LangChain and OpenAI
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file locally)

## Limitations

- The tool only answers based on the content from the provided URLs
- Some websites may block scraping attempts
- Complex or dynamic web pages may not be scraped correctly
- Large amounts of content may require more processing time

## License

MIT