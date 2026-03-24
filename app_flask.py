from flask import Flask, request, jsonify, render_template
import anthropic
import os
from dotenv import load_dotenv
from pdf_reader import read_pdf
from chunker import split_into_chunks
from multi_doc_store import load_all_documents, search_all

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load documents at startup
print("Loading documents...")
load_all_documents()
print("Ready!")

def ask_document(question):
    results = search_all(question)
    
    context = ""
    for r in results:
        context += f"From {r['source']}:\n{r['text']}\n\n"
    
    prompt = f"""You are a helpful healthcare assistant.
Answer the question based ONLY on the context below.
Always mention which document the answer came from.
If the answer is not in the context say 'I could not find that in the documents.'

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    answer = ask_document(question)
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)