
import os
import json
import pickle
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import PyPDF2
from PIL import Image

class CollegeRAGSystem:
    def __init__(self, groq_api_key: str):
        """Initialize the RAG system with Groq API"""
        self.client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        # Initialize embedding model (converts text to numbers)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        
        # Initialize FAISS index (vector database)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store document metadata
        self.documents = []  # List of document texts
        self.metadata = []   # List of metadata for each document
        
        print("✓ RAG System initialized!")
    
    def add_pdf(self, pdf_path: str, doc_type: str = "general"):
        """Add a PDF document to the knowledge base"""
        print(f"Processing PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():  # Only process non-empty pages
                        # Split into chunks
                        chunks = self._chunk_text(text, chunk_size=500)
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            if len(chunk.strip()) > 50:  # Skip very small chunks
                                # Generate embedding
                                embedding = self.embedding_model.encode(chunk)
                                
                                # Add to FAISS index
                                self.index.add(np.array([embedding], dtype=np.float32))
                                
                                # Store document and metadata
                                self.documents.append(chunk)
                                self.metadata.append({
                                    "source": os.path.basename(pdf_path),
                                    "type": doc_type,
                                    "page": page_num + 1,
                                    "chunk": chunk_idx
                                })
            
            print(f"✓ Added {pdf_path} - Total documents: {len(self.documents)}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {str(e)}")
    
    def add_text(self, text: str, metadata: Dict):
        """Add plain text to the knowledge base"""
        chunks = self._chunk_text(text, chunk_size=500)
        
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                embedding = self.embedding_model.encode(chunk)
                self.index.add(np.array([embedding], dtype=np.float32))
                
                self.documents.append(chunk)
                metadata_copy = metadata.copy()
                metadata_copy['chunk'] = chunk_idx
                self.metadata.append(metadata_copy)
        
        print(f"✓ Added text: {metadata.get('title', 'Untitled')}")
    
    def add_image_info(self, image_path: str, description: str, metadata: Dict):
        """Add image metadata (for question papers, college photos)"""
        full_text = f"Image: {os.path.basename(image_path)}\nDescription: {description}"
        
        embedding = self.embedding_model.encode(full_text)
        self.index.add(np.array([embedding], dtype=np.float32))
        
        self.documents.append(description)
        metadata.update({
            "type": "image",
            "image_path": image_path,
            "description": description
        })
        self.metadata.append(metadata)
        
        print(f"✓ Added image metadata: {os.path.basename(image_path)}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
        """Search for relevant documents"""
        if len(self.documents) == 0:
            return []
        
        # Convert query to embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search in FAISS
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(top_k, len(self.documents))
        )
        
        # Retrieve documents and metadata
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append((self.documents[idx], self.metadata[idx]))
        
        return results
    
    def generate_answer(self, query: str, top_k: int = 3) -> Dict:
        """Main RAG pipeline: Search + Generate answer"""
        
        # Step 1: Retrieve relevant context
        search_results = self.search(query, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I don't have enough information to answer this question. Please add more data to the knowledge base.",
                "sources": [],
                "images": []
            }
        
        # Step 2: Prepare context
        context_parts = []
        sources = []
        images = []
        
        for doc, meta in search_results:
            context_parts.append(doc)
            sources.append(meta)
            
            # Collect images if any
            if meta.get('type') == 'image':
                images.append({
                    'path': meta.get('image_path'),
                    'description': meta.get('description')
                })
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Build prompt
        system_prompt = """You are a helpful AI assistant for a college. You provide accurate information about:
- Academic programs, courses, syllabus, and exams
- Hostel facilities, rules, and timings
- Campus facilities and locations
- Question papers and exam patterns
- College events and activities

Use the provided context to answer questions. Be concise, friendly, and student-focused.
If the context doesn't contain relevant information, say so politely."""

        user_prompt = f"""Context from college database:
{context}

Student question: {query}

Answer the question based on the context above. Be helpful and concise."""
        
        # Step 4: Generate response using Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "sources": sources,
            "images": images
        }
    
    def save(self, filename: str = "rag_system.pkl"):
        """Save the RAG system to disk"""
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'index': faiss.serialize_index(self.index)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved RAG system to {filename}")
    
    def load(self, filename: str = "rag_system.pkl"):
        """Load a saved RAG system"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.index = faiss.deserialize_index(data['index'])
        
        print(f"✓ Loaded RAG system from {filename}")


# ==================== USAGE EXAMPLE ====================

def main():
    """Example usage of the RAG system"""
    
    # Initialize with your Groq API key
    GROQ_API_KEY ="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"
    rag = CollegeRAGSystem(GROQ_API_KEY)
    
    # ===== ADD DATA (Do this once) =====
    
    print("\n=== Adding Data to Knowledge Base ===\n")
    
    # 1. Add college information as text
    hostel_info = """
    College Hostel Information:
    
    We have 4 hostels:
    - Boys Hostel A: 200 capacity, AC rooms, located near gate 1
    - Boys Hostel B: 150 capacity, Non-AC rooms, located near library
    - Girls Hostel A: 180 capacity, AC rooms, 24/7 security
    - Girls Hostel B: 120 capacity, AC rooms, near sports complex
    
    Hostel Timings:
    - Entry allowed till 11 PM on weekdays
    - Entry allowed till 12 AM on weekends
    - Gates close at 6 AM for maintenance
    
    Mess Timings:
    - Breakfast: 7:00 AM to 9:00 AM
    - Lunch: 12:00 PM to 2:00 PM
    - Dinner: 7:00 PM to 9:00 PM
    
    Hostel Fees: Rs 80,000 per year (includes mess charges)
    
    Facilities: Wifi, laundry, common room, gym, study room
    """
    
    rag.add_text(hostel_info, {
        "title": "Hostel Information",
        "category": "hostel",
        "type": "text"
    })
    
    # 2. Add academic information
    academic_info = """
    BTech Computer Science Course Structure:
    
    Third Year (Semester 5 & 6):
    - Data Structures and Algorithms
    - Database Management Systems
    - Operating Systems
    - Computer Networks
    - Software Engineering
    - Machine Learning
    - Web Technologies
    - Elective 1
    
    Fourth Year (Semester 7 & 8):
    - Artificial Intelligence
    - Cloud Computing
    - Cyber Security
    - Big Data Analytics
    - Mobile App Development
    - Elective 2
    - Final Year Project
    - Internship (mandatory)
    
    Exam Pattern:
    - Mid-term: 30 marks (objective + subjective)
    - End-term: 70 marks (subjective)
    - Internal assessment: continuous evaluation
    
    Total Credits Required: 160 credits for graduation
    """
    
    rag.add_text(academic_info, {
        "title": "Academic Information",
        "category": "academics",
        "type": "text"
    })
    
    # 3. Add PDF (if you have any)
    # Uncomment when you have actual PDFs
    # rag.add_pdf("path/to/question_paper.pdf", doc_type="question_paper")
    
    # 4. Add image metadata
    # Uncomment when you have actual images
    # rag.add_image_info(
    #     image_path="path/to/campus_library.jpg",
    #     description="College central library building, 4 floors, open 8 AM to 10 PM, has computer lab",
    #     metadata={"location": "central_campus", "facility": "library"}
    # )
    
    # Optional: Save the system for later use
    # rag.save("college_rag.pkl")
    
    # ===== TEST QUERIES =====
    
    print("\n=== Testing RAG System ===\n")
    
    test_questions = [
        "What are the hostel timings?",
        "Tell me about Boys Hostel A",
        "What subjects are there in 3rd year CSE?",
        "What are the mess timings?",
        "How many credits do I need to graduate?",
        "Tell me about the exam pattern"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        
        result = rag.generate_answer(question)
        
        print(f"\nA: {result['answer']}")
        
        if result['sources']:
            print(f"\nSources:")
            for source in result['sources'][:2]:  # Show top 2 sources
                print(f"  - {source.get('title', source.get('source', 'Unknown'))}")
        
        if result['images']:
            print(f"\nRelevant Images:")
            for img in result['images']:
                print(f"  - {img['path']}")


if __name__ == "__main__":
    main()