import streamlit as st
import arxiv
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(
    page_title="ArXiv RAG Bot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Genel arka plan */
    .stApp {
        background-color: #0f1117;
    }

    /* Başlık alanı */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        border: 1px solid #2d3561;
    }

    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .hero p {
        color: #8892b0;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    /* Kart stili */
    .paper-card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .paper-card h3 {
        color: #e94560;
        margin: 0 0 8px 0;
        font-size: 1rem;
    }

    .paper-card p {
        color: #ccd6f6;
        margin: 0;
        font-size: 0.9rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3561;
    }

    /* Input */
    .stTextInput input {
        background-color: #16213e !important;
        border: 1px solid #2d3561 !important;
        border-radius: 8px !important;
        color: #ccd6f6 !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(90deg, #e94560, #c23152) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 12px !important;
    }

    /* Chat mesajları */
    [data-testid="stChatMessage"] {
        background-color: #1a1f2e !important;
        border: 1px solid #2d3561 !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: #1a1f2e !important;
        border: 1px solid #2d3561 !important;
        border-radius: 12px !important;
    }

    /* Divider */
    hr {
        border-color: #2d3561 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Yardımcı Fonksiyonlar ---

def download_and_process_paper(arxiv_id: str):
    if os.path.exists("papers"):
        shutil.rmtree("papers")
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
    os.makedirs("papers", exist_ok=True)
    paper.download_pdf(dirpath="papers")

    pdf_paths = [
        os.path.join("papers", f)
        for f in os.listdir("papers")
        if f.endswith(".pdf")
    ]

    all_documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_documents.extend(loader.load())

    all_documents = [
        doc for doc in all_documents
        if "The Law will never be perfect" not in doc.page_content
        and "Input-Input Layer" not in doc.page_content
        and len(doc.page_content.strip()) > 200
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("vector_db")

    return paper.title, paper.authors, paper.published.year, len(chunks)


def load_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = OllamaLLM(model="mistral")
    retriever = vector_db.as_retriever(search_kwargs={"k": 6})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an academic paper assistant.
Answer the following question based only on the provided context.
If the answer is not in the context, say "This information is not available in the provided papers."
Be detailed and informative in your response.

Context:
{context}

Question: {question}

Answer:"""
    )

    def combine_chunks(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | combine_chunks, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🔬 ArXiv RAG Bot")
    st.markdown("---")
    st.markdown("### 📥 Load a Paper")
    arxiv_id = st.text_input("ArXiv Paper ID", value="1706.03762", label_visibility="collapsed", placeholder="e.g. 1706.03762")
    st.caption("💡 Find IDs at [arxiv.org](https://arxiv.org)")

    if st.button("🚀 Load Paper"):
        with st.spinner("Downloading and processing..."):
            try:
                title, authors, year, chunk_count = download_and_process_paper(arxiv_id)
                st.session_state["paper_loaded"] = True
                st.session_state["paper_title"] = title
                st.session_state["paper_authors"] = authors
                st.session_state["paper_year"] = year
                st.session_state["chunk_count"] = chunk_count
                st.session_state["chat_history"] = []
                st.success("✅ Paper loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.get("paper_loaded"):
        st.markdown("---")
        st.markdown("### 📄 Current Paper")
        st.markdown(f"**{st.session_state['paper_title']}**")
        st.caption(f"📅 {st.session_state['paper_year']} · 📦 {st.session_state['chunk_count']} chunks")
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.caption("• What is the main contribution?")
        st.caption("• How does the model architecture work?")
        st.caption("• What are the experimental results?")
        st.caption("• What problem does this paper solve?")


# --- Ana Panel ---
if not st.session_state.get("paper_loaded"):
    # Hero ekranı
    st.markdown("""
    <div class="hero">
        <h1>🔬 ArXiv RAG Bot</h1>
        <p>Load any ArXiv paper and ask questions about it using AI-powered retrieval.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="paper-card">
            <h3>📥 Load Any Paper</h3>
            <p>Enter an ArXiv ID to instantly download and index any research paper.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="paper-card">
            <h3>🔍 Smart Retrieval</h3>
            <p>FAISS vector search finds the most relevant sections for your question.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="paper-card">
            <h3>🤖 AI Answers</h3>
            <p>Local LLM generates detailed answers grounded in the paper's content.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"### 💬 {st.session_state['paper_title']}")
    st.markdown("---")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the paper...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state["chat_history"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain, retriever = load_rag_chain()
                answer = chain.invoke(question)
                sources = retriever.invoke(question)

            st.write(answer)

            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1} — Page {doc.metadata.get('page', '?')}**")
                    st.text(doc.page_content[:300])
                    st.divider()

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})