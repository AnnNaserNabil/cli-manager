# Research AI Agent

A simple research agent that uses RAG (Retrieval Augmented Generation) to answer questions based on ingested documents.

## Features

- Document ingestion and processing
- Vector embeddings using OpenAI
- Semantic search and retrieval
- Natural language responses
- Incremental document addition

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials:
- Create a service account key in Google Cloud Console
- Save the JSON key file as `credentials.json` in the project directory
- The agent will automatically use the credentials from this file

3. Set the model to use in a `.env` file:
```
GOOGLE_PALM_MODEL=gemini-pro
```

## Usage

```python
from research_agent import ResearchAgent

# Initialize agent
agent = ResearchAgent()

# Ingest documents
agent.ingest_documents("path/to/documents")

# Perform research
query = "What information do you have about [topic]?"
result = agent.research(query)
print(result)
```

## Adding Documents

You can add documents incrementally:
```python
agent.add_document("path/to/new/document.txt")
```

## Document Formats Supported

- Text files (.txt)
- PDF files (.pdf)
- Other formats supported by unstructured library

## Example

The agent comes with a simple example in the `data` directory. You can run the script directly to see it in action:
```bash
python research_agent.py
```
