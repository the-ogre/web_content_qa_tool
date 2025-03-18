# Deployment Guide

This guide outlines different deployment options for the Web Content Q&A Tool.

## Local Deployment

To run the application locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

3. Run the application:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

You can deploy this application to Streamlit Cloud for free:

1. Push your code to a GitHub repository.

2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in.

3. Create a new app, connect to your GitHub repository.

4. Add your API key as a secret:
   - Name: `OPENAI_API_KEY`
   - Value: `your-api-key-here`

5. Deploy the app.

## Docker Deployment

1. Create a `Dockerfile`:
```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run the Docker container:
```bash
docker build -t web-content-qa .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-api-key-here web-content-qa
```

## Heroku Deployment

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

2. Add a `runtime.txt`:
```
python-3.9.16
```

3. Deploy to Heroku:
```bash
heroku create web-content-qa
heroku config:set OPENAI_API_KEY=your-api-key-here
git push heroku main
```

## Environment Variables

- `OPENAI_API_KEY`: Required for the OpenAI API
- `DEBUG`: Set to "True" to enable debug logging (optional)

## Security Considerations

- Store API keys securely, never commit them to version control
- For production, use a proper secrets management solution
- Consider implementing rate limiting for public deployments
- Add authentication if deploying to a public environment