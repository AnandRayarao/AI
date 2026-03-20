# Healthcare Document AI Assistant

A RAG (Retrieval Augmented Generation) application 
that answers questions about healthcare documents 
using Claude AI.

## What it does
- Loads any healthcare PDF document
- Splits it into searchable chunks
- Finds the most relevant sections for your question
- Answers using only information from the document
- Refuses to make up answers not found in the document

## Tech stack
- Python 3.10+
- Anthropic Claude API
- PyPDF2 for document reading
- Custom RAG pipeline built from scratch

## How to run

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies:
   pip install anthropic python-dotenv pypdf2
4. Add your API key to .env:
   ANTHROPIC_API_KEY=your-key-here
5. Add your PDF as document.pdf
6. Run:
   python app.py

## Example questions
- What are my rights under HIPAA?
- Can I see my medical records?
- How do I file a complaint?

## Built by
Java developer transitioning into AI engineering.
