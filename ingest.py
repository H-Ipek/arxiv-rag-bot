import arxiv
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def load_and_split_pdfs(pdf_paths: list):
    all_documents = []

    for path in pdf_paths:
        print(f"Reading: {path}")
        loader = PyPDFLoader(path)
        pages = loader.load()
        all_documents.extend(pages)

    print(f"Total pages loaded: {len(all_documents)}")

    # Filter out noisy pages
    all_documents = [
        doc for doc in all_documents
        if "The Law will never be perfect" not in doc.page_content
        and "Input-Input Layer" not in doc.page_content
        and len(doc.page_content.strip()) > 200
    ]

    print(f"Pages after filtering: {len(all_documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )

    chunks = splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vector_store(chunks: list):
    print("Creating embeddings... (this may take a while)")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("vector_db")
    print("Vector store saved to 'vector_db'!")
    return vector_db


if __name__ == "__main__":
    if os.path.exists("papers"):
        shutil.rmtree("papers")
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")

    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=["1706.03762"])))
    os.makedirs("papers", exist_ok=True)
    paper.download_pdf(dirpath="papers")
    print(f"Downloaded: {paper.title}")

    pdf_paths = []
    for file in os.listdir("papers"):
        if file.endswith(".pdf"):
            pdf_paths.append(os.path.join("papers", file))

    chunks = load_and_split_pdfs(pdf_paths)
    create_vector_store(chunks)