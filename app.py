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
    .stApp { background-color: #0f1117; }
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
    .hero p { color: #8892b0; font-size: 1.1rem; margin-top: 8px; }
    .paper-card {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .paper-card h3 { color: #e94560; margin: 0 0 8px 0; font-size: 1rem; }
    .paper-card p { color: #ccd6f6; margin: 0; font-size: 0.9rem; }
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        border-right: 1px solid #2d3561;
    }
    .stTextInput input {
        background-color: #16213e !important;
        border: 1px solid #2d3561 !important;
        border-radius: 8px !important;
        color: #ccd6f6 !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #e94560, #c23152) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 12px !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1a1f2e !important;
        border: 1px solid #2d3561 !important;
        border-radius: 12px !important;
        margin-bottom: 12px !important;
    }
    hr { border-color: #2d3561 !important; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def download_paper(arxiv_id: str):
    """Tek bir makaleyi indir, döküman listesi döndür."""
    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
    os.makedirs("papers", exist_ok=True)

    # Her makaleyi ayrı klasöre indir
    paper_dir = os.path.join("papers", arxiv_id)
    os.makedirs(paper_dir, exist_ok=True)
    paper.download_pdf(dirpath=paper_dir)

    documents = []
    for f in os.listdir(paper_dir):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(paper_dir, f))
            documents.extend(loader.load())
            # Her dökümana hangi makaleye ait olduğunu ekle
            for doc in documents:
                doc.metadata["arxiv_id"] = arxiv_id
                doc.metadata["title"] = paper.title

    return paper, documents


def build_vector_store(all_documents: list):
    """Tüm dökümanlardan vektör DB oluştur."""
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    filtered = [
        doc for doc in all_documents
        if "The Law will never be perfect" not in doc.page_content
        and "Input-Input Layer" not in doc.page_content
        and len(doc.page_content.strip()) > 200
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(filtered)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("vector_db")

    return len(chunks)


def load_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.load_local(
        "vector_db", embeddings, allow_dangerous_deserialization=True
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


def summarize_paper(title: str, arxiv_id: str):
    """Belirli bir makaleyi özetle."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.load_local(
        "vector_db", embeddings, allow_dangerous_deserialization=True
    )
    llm = OllamaLLM(model="mistral")

    # Sadece bu makaleye ait chunk'ları getir
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(f"main contribution methodology results of {title}")

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are an academic paper assistant. 
Based on the following excerpts from the paper "{title}", write a comprehensive summary covering:
1. The problem being solved
2. The proposed approach/methodology
3. Key results and contributions

Context:
{context}

Summary:"""

    return llm.invoke(prompt)


# --- Session State Init ---
if "papers" not in st.session_state:
    st.session_state["papers"] = {}  # {arxiv_id: {title, year, authors}}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "all_documents" not in st.session_state:
    st.session_state["all_documents"] = []


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🔬 ArXiv RAG Bot")
    st.markdown("---")
    st.markdown("### 📥 Add a Paper")

    arxiv_id = st.text_input("ArXiv Paper ID", placeholder="e.g. 1706.03762", label_visibility="collapsed")
    st.caption("💡 Find IDs at [arxiv.org](https://arxiv.org)")

    if st.button("➕ Add Paper"):
        if not arxiv_id.strip():
            st.error("Please enter a paper ID.")
        elif arxiv_id in st.session_state["papers"]:
            st.warning("This paper is already loaded.")
        else:
            with st.spinner(f"Downloading {arxiv_id}..."):
                try:
                    paper, documents = download_paper(arxiv_id)
                    st.session_state["papers"][arxiv_id] = {
                        "title": paper.title,
                        "year": paper.published.year,
                        "authors": [a.name for a in paper.authors[:3]]
                    }
                    st.session_state["all_documents"].extend(documents)

                    # Vektör DB'yi yeniden oluştur
                    chunk_count = build_vector_store(st.session_state["all_documents"])
                    st.success(f"✅ Added! {chunk_count} total chunks.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Yüklü makaleler listesi
    if st.session_state["papers"]:
        st.markdown("---")
        st.markdown("### 📚 Loaded Papers")
        for pid, info in st.session_state["papers"].items():
            with st.expander(f"📄 {info['title'][:40]}..."):
                st.caption(f"ID: {pid} · {info['year']}")
                st.caption(", ".join(info["authors"]))
                if st.button(f"🗑️ Remove", key=f"remove_{pid}"):
                    del st.session_state["papers"][pid]
                    # Dökümanları filtrele
                    st.session_state["all_documents"] = [
                        doc for doc in st.session_state["all_documents"]
                        if doc.metadata.get("arxiv_id") != pid
                    ]
                    if st.session_state["all_documents"]:
                        build_vector_store(st.session_state["all_documents"])
                    st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.caption("• What is the main contribution?")
        st.caption("• How does the model architecture work?")
        st.caption("• What are the experimental results?")
        st.caption("• Compare the methods in these papers.")


# --- Ana Panel ---
if not st.session_state["papers"]:
    st.markdown("""
    <div class="hero">
        <h1>🔬 ArXiv RAG Bot</h1>
        <p>Load any ArXiv paper and ask questions about it using AI-powered retrieval.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="paper-card">
            <h3>📥 Load Multiple Papers</h3>
            <p>Add multiple ArXiv papers and ask questions across all of them at once.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="paper-card">
            <h3>🔍 Smart Retrieval</h3>
            <p>FAISS vector search finds the most relevant sections for your question.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="paper-card">
            <h3>📝 Auto Summarize</h3>
            <p>Generate a comprehensive summary of any loaded paper with one click.</p>
        </div>""", unsafe_allow_html=True)

else:
    paper_count = len(st.session_state["papers"])
    st.markdown(f"### 💬 Chatting across {paper_count} paper{'s' if paper_count > 1 else ''}")

    # Özet butonları
    if st.session_state["papers"]:
        st.markdown("#### 📝 Summarize a Paper")
        cols = st.columns(min(paper_count, 3))
        for i, (pid, info) in enumerate(st.session_state["papers"].items()):
            with cols[i % 3]:
                if st.button(f"Summarize: {info['title'][:30]}...", key=f"sum_{pid}"):
                    with st.spinner("Generating summary..."):
                        summary = summarize_paper(info["title"], pid)
                    st.session_state["chat_history"].append({
                        "role": "user",
                        "content": f"📝 Summarize: {info['title']}"
                    })
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": summary
                    })
                    st.rerun()

    st.markdown("---")

    # Sohbet geçmişi
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Soru input
    question = st.chat_input("Ask a question about the papers...")

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
                    title = doc.metadata.get("title", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i+1} — {title} · Page {page}**")
                    st.text(doc.page_content[:300])
                    st.divider()

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})