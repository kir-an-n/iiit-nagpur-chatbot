import streamlit as st
from college_rag import CollegeRAGSystem
import os

# Page config
st.set_page_config(
    page_title="IIIT Nagpur AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG system
@st.cache_resource
def load_rag():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    rag = CollegeRAGSystem(api_key)
    rag.load("college_rag_complete.pkl")
    return rag

# Header
st.title("🎓 IIIT Nagpur AI Assistant")
st.markdown("Ask me anything about IIIT Nagpur!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"][:3]:
                    st.write(f"- {source.get('title', 'Unknown')}")

# Chat input
if prompt := st.chat_input("Ask about hostel, fees, facilities, programs..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag = load_rag()
                result = rag.generate_answer(prompt)
                response = result['answer']
                sources = result.get('sources', [])
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources[:3]:
                            st.write(f"- {source.get('title', 'Unknown')}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("AI-powered chatbot for IIIT Nagpur")
    st.write(f"📚 Knowledge Base: 534 documents")
    
    st.header("Quick Questions")
    if st.button("Hostel timings?"):
        st.session_state.messages.append({"role": "user", "content": "What are hostel mess timings?"})
        st.rerun()
    
    if st.button("Annual fees?"):
        st.session_state.messages.append({"role": "user", "content": "What is the annual fee?"})
        st.rerun()
    
    if st.button("Sports facilities?"):
        st.session_state.messages.append({"role": "user", "content": "What sports facilities are available?"})
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()