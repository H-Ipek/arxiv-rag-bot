from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_vector_db():
    print("Loading vector store...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Loaded!")
    return vector_db


def create_rag_chain(vector_db):
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


def ask_question(chain, retriever, question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    answer = chain.invoke(question)
    print(f"Answer:\n{answer}")

    sources = retriever.invoke(question)
    print("\n--- Source Chunks ---")
    for i, doc in enumerate(sources):
        print(f"\n[Source {i+1}] Page {doc.metadata.get('page', '?')}")
        print(doc.page_content[:200] + "...")


if __name__ == "__main__":
    db = load_vector_db()
    chain, retriever = create_rag_chain(db)

    ask_question(chain, retriever, "What is the purpose of the attention mechanism?")
    ask_question(chain, retriever, "What is the main contribution of this paper?")