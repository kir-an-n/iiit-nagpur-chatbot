import os
import requests

# Download RAG file from Google Drive if not exists
if not os.path.exists("college_rag_complete.pkl"):
    print("Downloading RAG system from Google Drive...")
    file_id = "1DAMe7U6MXaxRpZtJRCDlIzmKfCTxK8zk"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(url, stream=True)
    with open("college_rag_complete.pkl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✓ Downloaded!")


import os
import gdown

# Local path to save the file
file_path = "college_rag_complete.pkl"

# Google Drive file ID
file_id = "1DAMe7U6MXaxRpZtJRCDlIzmKfCTxK8zk"

# Construct direct download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download if it doesn't exist locally
if not os.path.exists(file_path):
    print("Downloading college_rag_complete.pkl from Google Drive...")
    gdown.download(url, file_path, quiet=False)


from flask import Flask, request, jsonify
from flask_cors import CORS
from college_rag import CollegeRAGSystem

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Initialize RAG system
GROQ_API_KEY = "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"
print("Loading RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)
rag.load("college_rag_complete.pkl")
print(f"✓ RAG system loaded! Documents: {len(rag.documents)}")

@app.route('/')
def home():
    return "IIIT Nagpur AI Chatbot API is running!"

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get answer from RAG
        result = rag.generate_answer(question)
        
        return jsonify({
            "question": question,
            "answer": result['answer'],
            "sources": result['sources'][:2] if result['sources'] else [],
            "images": result['images']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
