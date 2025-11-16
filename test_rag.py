from college_rag import CollegeRAGSystem

# Your Groq API key
GROQ_API_KEY = "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"

# Initialize and load saved system
print("Loading RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)
rag.load("college_rag_complete.pkl")

print(f"✓ Loaded! Total documents: {len(rag.documents)}\n")

# Test questions
test_questions = [
    "What are the hostel mess timings?",
    "Tell me about the academic building",
    "What is the fee for 2nd year BTech?",
    "Who is the director of IIIT Nagpur?",
    "What sports facilities are available?"
]

for question in test_questions:
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print(f"{'='*70}")
    
    result = rag.generate_answer(question)
    print(f"\nA: {result['answer']}\n")