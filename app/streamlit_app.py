"""
DocBot - Streamlit Frontend
AI Document Chatbot with RAG & Hallucination Control
"""
import streamlit as st
from datetime import datetime
import requests

# Page configuration
st.set_page_config(
    page_title="DocBot - AI Document Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .user-msg { background-color: #E3F2FD; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #1E88E5; }
    .ai-msg { background-color: #F5F5F5; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #4CAF50; }
    .not-found { background-color: #FFEBEE; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #F44336; color: #C62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def init_state():
    """Initialize session state."""
    if "messages" not in st.session_state: st.session_state.messages = []
    if "session_id" not in st.session_state: st.session_state.session_id = None
    if "documents" not in st.session_state: st.session_state.documents = []
    if "embedding" not in st.session_state: st.session_state.embedding = "free"
    if "llm" not in st.session_state: st.session_state.llm = "gemini"


def check_api():
    """Check API availability."""
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=2).status_code == 200
    except:
        return False


def chat_api(question, session_id, emb, llm):
    """Call chat API."""
    try:
        resp = requests.post(f"{API_BASE_URL}/chat", json={
            "question": question,
            "session_id": session_id,
            "embedding_provider": emb,
            "llm_provider": llm
        }, timeout=60)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def upload_file(file):
    """Upload document."""
    try:
        resp = requests.post(f"{API_BASE_URL}/upload-doc", 
            files={"file": (file.name, file.getvalue(), file.type)}, timeout=120)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None


def get_docs():
    """Get document list."""
    try:
        resp = requests.get(f"{API_BASE_URL}/list-docs", timeout=5)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []


def del_doc(file_id):
    """Delete document."""
    try:
        return requests.post(f"{API_BASE_URL}/delete-doc", params={"file_id": file_id}, timeout=10).status_code == 200
    except:
        return False


def render_msg(role, content, sources=None, model=None):
    """Render chat message."""
    if role == "user":
        st.markdown(f'<div class="user-msg"><strong>👤 You:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        if content == "This information is not present in the provided document.":
            st.markdown(f'<div class="not-found"><strong>🤖 DocBot:</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg"><strong>🤖 DocBot:</strong><br>{content}</div>', unsafe_allow_html=True)
        
        if sources:
            with st.expander("📚 Sources & Scores"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** (Score: `{s['score']:.4f}`) - `{s['source']}`")
                    st.markdown(f"> {s['content'][:150]}...")
        
        if model:
            st.caption(f"🧠 {model}")


def main():
    """Main app."""
    init_state()
    
    st.markdown('<h1 class="main-header">📄 DocBot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">RAG Chatbot with Hallucination Control</p>', unsafe_allow_html=True)
    
    if not check_api():
        st.error("⚠️ API not running. Start with: `cd api && uvicorn main:app --reload --port 8000`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.session_state.embedding = st.selectbox(
            "Embedding", 
            ["free_huggingface", "free_watsonx", "paid_openai", "paid_gemini"],
            format_func=lambda x: {
                "free_huggingface": "🆓 HuggingFace (local, FREE)",
                "free_watsonx": "🆓 IBM WatsonX (FREE tier)",
                "paid_openai": "💰 OpenAI (PAID)",
                "paid_gemini": "💰 Google Gemini (PAID)"
            }[x]
        )
        
        st.session_state.llm = st.selectbox(
            "LLM", ["gemini", "openai"],
            format_func=lambda x: {"gemini": "🤖 Gemini (free)", "openai": "🤖 GPT-4 (paid)"}[x]
        )
        
        st.markdown("### 📤 Upload")
        uploaded = st.file_uploader("PDF/DOCX", type=["pdf", "docx"])
        
        if uploaded and st.button("Upload", type="primary"):
            with st.spinner("Processing..."):
                result = upload_file(uploaded)
                if result:
                    st.success(f"✅ Uploaded: {result['chunks_created']} chunks")
                    st.session_state.documents = get_docs()
        
        st.markdown("### 📚 Documents")
        if st.button("🔄 Refresh"):
            st.session_state.documents = get_docs()
        
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.write(f"Chunks: {doc['chunks_count']}")
                if st.button(f"Delete", key=f"del_{doc['file_id']}"):
                    if del_doc(doc['file_id']):
                        st.success("Deleted")
                        st.session_state.documents = get_docs()
    
    # Chat
    st.markdown("### 💬 Conversation")
    
    for msg in st.session_state.messages:
        render_msg(msg['role'], msg['content'], msg.get('sources'), msg.get('model'))
    
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("Ask about your documents...", key="input")
    with col2:
        send = st.button("Send", type="primary", use_container_width=True)
    
    if send and question.strip():
        with st.spinner("Thinking..."):
            resp = chat_api(question, st.session_state.session_id, 
                          st.session_state.embedding, st.session_state.llm)
            if resp:
                st.session_state.session_id = resp['session_id']
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": resp['answer'],
                    "sources": resp.get('sources', []),
                    "model": resp.get('model_used')
                })
                st.rerun()
    
    # Info
    st.markdown("""
    ---
    **How to use:**
    1. Upload PDF/DOCX from sidebar
    2. Ask questions about the document
    3. Get answers ONLY from document content
    4. "Not found" response when info unavailable
    """)


if __name__ == "__main__":
    main()