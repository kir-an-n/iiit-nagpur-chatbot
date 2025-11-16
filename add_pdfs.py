import os
from college_rag import CollegeRAGSystem

GROQ_API_KEY = "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"

# Load existing RAG system
print("Loading RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)
rag.load("college_rag_complete.pkl")

print(f"Current documents: {len(rag.documents)}")

# Add all PDFs
pdf_folder = "college_data/pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

print(f"\nFound {len(pdf_files)} PDF files\n")

for idx, filename in enumerate(pdf_files, 1):
    pdf_path = os.path.join(pdf_folder, filename)
    print(f"[{idx}/{len(pdf_files)}] Processing: {filename}")
    
    try:
        rag.add_pdf(pdf_path, doc_type="official_document")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# Save
print("\nSaving updated RAG system...")
rag.save("college_rag_complete.pkl")
print(f"✓ Done! Total documents: {len(rag.documents)}")