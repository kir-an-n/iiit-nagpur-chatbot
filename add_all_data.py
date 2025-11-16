import os
from college_rag import CollegeRAGSystem

# Your Groq API key
GROQ_API_KEY = "your_actual_groq_key_here"

# Initialize RAG system
print("Initializing RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)

# Add all text files
text_folder = "college_data/text"
text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

print(f"\nFound {len(text_files)} text files to process\n")

for idx, filename in enumerate(text_files, 1):
    file_path = os.path.join(text_folder, filename)
    print(f"[{idx}/{len(text_files)}] Processing: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add to RAG
        rag.add_text(content, {
            "title": filename.replace('.txt', '').replace('_', ' ').title(),
            "source": filename,
            "type": "text"
        })
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        continue

# Save the system
print("\nSaving RAG system...")
rag.save("college_rag_complete.pkl")

print(f"\n✓ Done! Total documents in system: {len(rag.documents)}")
print("RAG system saved as: college_rag_complete.pkl")