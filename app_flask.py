from flask import Flask, request, jsonify, render_template
import anthropic
import os
from dotenv import load_dotenv
from pdf_reader import read_pdf
from chunker import split_into_chunks
from multi_doc_store import load_all_documents, search_all
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Upload settings
UPLOAD_FOLDER = 'docs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load documents at startup
print("Loading documents...")
load_all_documents()
print("Ready!")

def ask_document(question):
    # Guard: empty question
    if not question or not question.strip():
        return "Please ask a valid question."

    try:
        results = search_all(question)
    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return "Sorry, I could not search the documents. Please try again."

    # Guard: no documents loaded
    if not results:
        return "No documents are loaded yet. Please upload a PDF first."

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

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except anthropic.APIConnectionError:
        return "Could not connect to Claude API. Please check your internet connection."
    except anthropic.AuthenticationError:
        return "Invalid API key. Please check your .env file."
    except anthropic.RateLimitError:
        return "Too many requests. Please wait a moment and try again."
    except Exception as e:
        print(f"Claude API error: {e}")
        return "Something went wrong while generating the answer. Please try again."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = ask_document(question)
    return jsonify({'answer': answer})


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
    except Exception as e:
        print(f"File save error: {e}")
        return jsonify({'error': 'Failed to save the file. Please try again.'}), 500

    try:
        pdf_text = read_pdf(file_path)
    except Exception as e:
        print(f"PDF read error: {e}")
        # Clean up the saved file if we can't read it
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': 'Could not read the PDF. It may be corrupted or password-protected.'}), 400

    if not pdf_text or not pdf_text.strip():
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': 'The PDF appears to be empty or has no readable text.'}), 400

    try:
        chunks = split_into_chunks(pdf_text)
        from multi_doc_store import add_single_document
        add_single_document(filename, chunks)
    except Exception as e:
        print(f"Embedding/storage error: {e}")
        return jsonify({'error': 'Failed to process the PDF. Please try again.'}), 500

    return jsonify({
        'message': f'Successfully loaded {filename}',
        'chunks': len(chunks)
    })


@app.route('/documents', methods=['GET'])
def list_documents():
    try:
        files = os.listdir('docs')
        pdfs = [f for f in files if f.endswith('.pdf')]
        return jsonify({'documents': pdfs})
    except Exception as e:
        print(f"Error listing documents: {e}")
        return jsonify({'error': 'Could not list documents.'}), 500


if __name__ == '__main__':
    app.run(debug=True)