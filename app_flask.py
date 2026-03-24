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


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files allowed'}), 400
    
    # Save file to docs folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Load into vector database
    pdf_text = read_pdf(file_path)
    chunks = split_into_chunks(pdf_text)
    
    from multi_doc_store import add_single_document
    add_single_document(filename, chunks)
    
    return jsonify({
        'message': f'Successfully loaded {filename}',
        'chunks': len(chunks)
    })


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all loaded documents"""
    files = os.listdir('docs')
    pdfs = [f for f in files if f.endswith('.pdf')]
    return jsonify({'documents': pdfs})


if __name__ == '__main__':
    app.run(debug=True)