from college_rag import CollegeRAGSystem

rag = CollegeRAGSystem("your_key")
rag.load("college_rag_complete.pkl")
print(f"Total documents: {len(rag.documents)}")