import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# Import our config and QA pairs
from config import LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, EMBEDDING_MODEL
from qa_pairs import QA_PAIRS

# ── 1. Environment setup ────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

# ── 2. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# ── 3. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    print("Building vector store...")
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError("knowledge_base.txt not found in data/ directory.")
    
    text = kb_path.read_text(encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"   Split into {len(chunks)} chunks")
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 4. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# ── 5. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 6. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# ── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # Build everything
    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    # Run all 50 questions
    print(f"\nRunning {len(QA_PAIRS)} questions...")
    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        answer = ask(chain, question)
        print(f"[{i:02d}/50] Q: {question[:60]}...")
        # print(f"       A: {answer[:100]}\n") # Keeping output clean

    print(f"\nAll {len(QA_PAIRS)} traces sent to LangSmith project '{LANGCHAIN_PROJECT}'")
    print("   Open https://smith.langchain.com to view traces.")

if __name__ == "__main__":
    main()
