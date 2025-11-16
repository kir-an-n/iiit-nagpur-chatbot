import os

# Check if RAG exists, if not rebuild it
if not os.path.exists("college_rag_complete.pkl"):
    print("Building RAG system from data files...")
    import glob
    for txt_file in glob.glob("college_data/text/*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            rag.add_text(f.read(), {"source": txt_file})
    rag.save("college_rag_complete.pkl")


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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)