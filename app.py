import streamlit as st
import os
import logging
import json
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, utility
import time

MILVUS_COLLEC_NAME = "gmesh_doc_collec"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K_RESULTS = 5

class RAGSystem:
    def __init__(self):
        self.setup_logging()
        self.model = None
        self.collection = None
        
    def setup_logging(self):
        """Set up logging for the RAG system"""
        os.makedirs('logs', exist_ok=True)
        log_filename = f'logs/rag_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename)
            ]
        )
    
    @st.cache_resource
    def load_sentence_transformer(_self):
        """Load the sentence transformer model (cached)"""
        return SentenceTransformer('all-MiniLM-L6-v2')
        
    def connect_to_milvus(self):
        """Connect to Milvus and load the collection"""
        try:
            connections.connect("default", host="localhost", port="19530")
            
            if utility.has_collection(MILVUS_COLLEC_NAME):
                self.collection = Collection(MILVUS_COLLEC_NAME)
                self.collection.load()
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def retrieve_similar_documents(self, query, top_k=TOP_K_RESULTS):
        """Retrieve similar documents from Milvus based on query"""
        try:
            if self.model is None:
                self.model = self.load_sentence_transformer()
                
            query_embedding = self.model.encode([query]).tolist()
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=query_embedding,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )
            
            retrieved_docs = []
            for hits in results:
                for hit in hits:
                    retrieved_docs.append({
                        'text': hit.entity.get('text'),
                        'score': hit.score
                    })
            
            return retrieved_docs
            
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            return []
    
    def query_ollama(self, prompt):
        """Send a query to Ollama and get the response"""
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama server - {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error occurred - {str(e)}"
    
    def create_rag_prompt(self, query, retrieved_docs):
        """Create a prompt for the LLM using retrieved documents"""
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. Use the context below to answer the user's question. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def ask_question(self, query):
        """Main method to ask a question using RAG"""
        retrieved_docs = self.retrieve_similar_documents(query)
        
        if not retrieved_docs:
            return "Sorry, I couldn't find any relevant information to answer your question.", []
        
        prompt = self.create_rag_prompt(query, retrieved_docs)
        
        response = self.query_ollama(prompt)
        
        return response, retrieved_docs

@st.cache_resource
def initialize_rag():
    return RAGSystem()

def main():
    st.set_page_config(
        page_title="RAG System - Gmsh Documentation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 RAG System - Gmsh Documentation Assistant</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔧 System Status")
        
        rag = initialize_rag()
        
        milvus_status = rag.connect_to_milvus()
        if milvus_status:
            st.success("✅ Milvus Connected")
        else:
            st.error("❌ Milvus Connection Failed")
        
        try:
            test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ Ollama Connected")
                
                models = test_response.json().get('models', [])
                if models:
                    st.info(f"📋 Model: {OLLAMA_MODEL}")
            else:
                st.error("❌ Ollama Connection Failed")
        except:
            st.error("❌ Ollama Not Available")
        
        st.markdown("---")
        
        st.subheader("🎛️ Settings")
        top_k = st.slider("Number of retrieved documents", 1, 10, TOP_K_RESULTS)
        show_sources = st.checkbox("Show source documents", value=True)
        
        st.markdown("---")
        
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Ask a question** about Gmsh in the text area
        2. **Click 'Get Answer'** to process your query
        3. **View the response** and source documents
        4. **Ask follow-up questions** as needed
        
        **Example questions:**
        - How to create a square mesh in Gmsh?
        - What are the element types in Gmsh?
        - How to set mesh size in Gmsh?
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("❓ Ask Your Question")
        question = st.text_area(
            "Enter your question about Gmsh:",
            placeholder="e.g., Give me a Gmsh code to generate a square with 1m sides and element size of 0.05m",
            height=100
        )
        
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                if milvus_status:
                    with st.spinner("🔍 Searching for relevant information..."):
                        answer, retrieved_docs = rag.ask_question(question)
                        
                        st.session_state.last_question = question
                        st.session_state.last_answer = answer
                        st.session_state.last_sources = retrieved_docs
                else:
                    st.error("❌ Cannot process question: Milvus connection failed")
            else:
                st.warning("⚠️ Please enter a question first")
        
        if hasattr(st.session_state, 'last_question'):
            st.markdown("---")
            
            st.markdown('<div class="question-box">', unsafe_allow_html=True)
            st.markdown(f"**Your Question:** {st.session_state.last_question}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown("**Answer:**")
            st.markdown(st.session_state.last_answer)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if show_sources and hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
                with st.expander(f"📚 Source Documents ({len(st.session_state.last_sources)} found)", expanded=False):
                    for i, doc in enumerate(st.session_state.last_sources, 1):
                        st.markdown(f"**Source {i}** (Similarity Score: {doc['score']:.4f})")
                        st.markdown('<div class="source-box">', unsafe_allow_html=True)
                        st.markdown(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 Quick Examples")
        
        example_questions = [
            "Generate a square mesh with uniform elements",
            "How to create a 3D cylinder mesh?",
            "What mesh algorithms are available?",
            "How to refine mesh locally?",
            "Export mesh to different formats"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_question = example
                st.rerun()
        
        st.markdown("---")
        
        if hasattr(st.session_state, 'last_sources'):
            st.metric("Documents Retrieved", len(st.session_state.last_sources))
            if st.session_state.last_sources:
                avg_score = sum(doc['score'] for doc in st.session_state.last_sources) / len(st.session_state.last_sources)
                st.metric("Avg. Similarity Score", f"{avg_score:.4f}")

if __name__ == "__main__":
    main()