import os
import pickle
import numpy as np
import faiss
import PyPDF2
from typing import Dict, List, Tuple
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- RAG SYSTEM CLASS ---

class CollegeRAGSystem:
    def __init__(self, groq_api_key: str):
        """Initialize the RAG system components and Groq client."""
        try:
            self.client = Groq(api_key=groq_api_key)
            self.model_name = "llama-3.1-8b-instant"
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.documents = []
            self.metadata = []
            self.api_ready = True
        except Exception as e:
            self.api_ready = False
            print(f"ERROR: Failed to initialize RAG system components: {e}")
            
    # --- Data Persistence ---
    def load(self, filename: str = "college_rag.pkl"):
        """Load a saved RAG system from disk."""
        if not os.path.exists(filename):
            print(f"Warning: RAG index file '{filename}' not found. Please run the data indexing process first.")
            return

        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.index = faiss.deserialize_index(data['index'])
            print(f"✓ Loaded RAG system with {len(self.documents)} chunks from {filename}")
        except Exception as e:
            print(f"ERROR loading index: {e}")
    
    def save(self, filename: str = "college_rag.pkl"):
        """Save the RAG system to disk."""
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'index': faiss.serialize_index(self.index)
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved RAG system to {filename}")

    # --- Data Ingestion (Simplified for the final code) ---
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    def add_text(self, text: str, metadata: Dict):
        """Adds text chunks to the knowledge base."""
        chunks = self._chunk_text(text, chunk_size=500)
        
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                embedding = self.embedding_model.encode(chunk)
                self.index.add(np.array([embedding], dtype=np.float32))
                self.documents.append(chunk)
                metadata_copy = metadata.copy()
                metadata_copy['chunk'] = chunk_idx
                self.metadata.append(metadata_copy)

    # --- Search and Generation ---
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict]]:
        """Search for relevant documents in FAISS."""
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(top_k, len(self.documents))
        )
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append((self.documents[idx], self.metadata[idx]))
        return results
    
    def generate_answer(self, query: str, user_role: str, user_name: str, top_k: int = 3) -> Dict:
        """Main RAG pipeline: Search + Generate answer with RBAC context."""
        
        if not self.api_ready:
             return {"answer": "Error: Groq API client failed to initialize. Please check the API key.", "sources": []}
        
        # Step 1: Retrieve relevant context
        search_results = self.search(query, top_k=top_k)
        
        # Step 2: Prepare context
        if not search_results:
            context = "No relevant information found in the knowledge base."
            sources = []
        else:
            context_parts = [doc for doc, meta in search_results]
            context = "\n\n".join(context_parts)
            sources = [meta for doc, meta in search_results]
        
        # Step 3: Build RAG Prompt with RBAC and Personalization
        system_prompt = f"""You are a helpful AI assistant for a college. The user is {user_name}, and they have logged in as a {user_role}. 
        Use the 'Context from college database' below to answer the user's question accurately. Be concise and polite.
        STRICTLY ADHERE to the user's role: If the answer requires information beyond {user_role}'s access rights (e.g., restricted faculty memos), politely state that you cannot provide that information due to their current role."""

        user_prompt = f"""Context from college database:
{context}

Student question (from {user_name}, role {user_role}): {query}

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
            answer = f"Error generating response (Groq API): {str(e)}"
        
        return {
            "answer": answer,
            "sources": sources,
        }

# --- DATA BUILDER FUNCTION ---

def run_data_builder(groq_api_key: str):
    """Function to run once to build the RAG index."""
    
    rag = CollegeRAGSystem(groq_api_key)
    if not rag.api_ready:
        print("Error: Could not initialize RAG builder due to API key failure.")
        return
    
    print("\n=== Adding Data to Knowledge Base ===\n")
    
    # 1. Add sample academic information
    academic_info = """
    BTech Computer Science Course Structure: Third Year (Semester 5 & 6) includes Data Structures, Operating Systems, Computer Networks, and Machine Learning. Fourth Year includes Artificial Intelligence, Cloud Computing, Cyber Security, and the Final Year Project. Total Credits Required: 160 credits for graduation. Exam Pattern: Mid-term is 30 marks, End-term is 70 marks.
    """
    rag.add_text(academic_info, {"title": "Academic Information", "category": "academics", "type": "text"})
    
    # 2. Add sample hostel information
    hostel_info = """
    College Hostel Information: We have 4 hostels. Boys Hostel A has 200 capacity, AC rooms. Girls Hostel A has 180 capacity, 24/7 security. Hostel Timings: Entry allowed till 11 PM on weekdays and 12 AM on weekends. Hostel Fees: Rs 80,000 per year (includes mess charges).
    """
    rag.add_text(hostel_info, {"title": "Hostel Information", "category": "hostel", "type": "text"})
    
    # Save the system
    rag.save("college_rag.pkl")
    print("\n--- RAG Indexing Complete. Ready to run Streamlit app. ---\n")


if __name__ == "__main__":
    # WARNING: Replace this with your actual Groq key for building the index
    # Use your actual key here, or set it as an environment variable
    API_KEY = "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" 
    run_data_builder(API_KEY)