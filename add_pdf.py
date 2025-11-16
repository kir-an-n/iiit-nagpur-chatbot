import os
from college_rag import CollegeRAGSystem

# Your Groq API key
GROQ_API_KEY = "your_groq_key_here"

# Initialize RAG system
print("Initializing RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)

# Path to PDF folder
pdf_folder = "college_data/pdfs"

# Add all PDFs
print(f"\nAdding PDFs from {pdf_folder}...\n")

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files\n")

for idx, filename in enumerate(pdf_files, 1):
    pdf_path = os.path.join(pdf_folder, filename)
    print(f"[{idx}/{len(pdf_files)}] Processing: {filename}")
    
    try:
        rag.add_pdf(pdf_path, doc_type="college_policy")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        continue

# Save the system
print("\nSaving RAG system...")
rag.save("college_rag_full.pkl")

print("\n✓ Done! All PDFs processed and saved.")
print(f"Total documents in system: {len(rag.documents)}")