import os
from college_rag import CollegeRAGSystem
import PyPDF2
from pdf2image import convert_from_path
import pytesseract

# Set tesseract path (change this to YOUR installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

GROQ_API_KEY = "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"

print("Loading RAG system...")
rag = CollegeRAGSystem(GROQ_API_KEY)
rag.load("college_rag_complete.pkl")
print(f"Starting documents: {len(rag.documents)}")

pdf_folder = "college_data/pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

print(f"\nProcessing {len(pdf_files)} PDFs with OCR...\n")
print("⚠️ This will take 20-30 minutes for 77 PDFs!\n")

for idx, filename in enumerate(pdf_files, 1):
    pdf_path = os.path.join(pdf_folder, filename)
    print(f"[{idx}/{len(pdf_files)}] {filename}")
    
    try:
        # Try normal text extraction first
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            
            # If no text, try OCR
            if len(text.strip()) < 100:
                print(f"  Using OCR...")
                images = convert_from_path(pdf_path, first_page=1, last_page=3)  # Only first 3 pages
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            
            # Add to RAG if we got text
            if len(text.strip()) > 100:
                rag.add_text(text, {
                    "title": filename.replace('.pdf', ''),
                    "source": filename,
                    "type": "pdf_document"
                })
                print(f"  ✓ Added ({len(text)} chars)")
            else:
                print(f"  ✗ No readable text")
                
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

print("\nSaving...")
rag.save("college_rag_complete.pkl")
print(f"✓ Done! Total documents: {len(rag.documents)}")