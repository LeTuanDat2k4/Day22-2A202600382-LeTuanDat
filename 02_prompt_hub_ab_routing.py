import os
import hashlib
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# Import our config and QA pairs
from config import LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, EMBEDDING_MODEL
from qa_pairs import QA_PAIRS

# ── 1. Environment / imports ────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

# Unique prompt names for the Hub
PROMPT_V1_NAME = "letuandat-rag-v1"
PROMPT_V2_NAME = "letuandat-rag-v2"

# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    print("Pushing prompts to LangSmith Prompt Hub...")
    try:
        url1 = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 - concise answers")
        print(f"   Pushed V1 -> {url1}")
    except Exception as e:
        print(f"   V1 push failed: {e}")

    try:
        url2 = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 - structured answers")
        print(f"   Pushed V2 -> {url2}")
    except Exception as e:
        print(f"   V2 push failed: {e}")

# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    print("Pulling prompts from Prompt Hub...")
    prompts = {}
    
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"   Pulled '{PROMPT_V1_NAME}'")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"   Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"   Pulled '{PROMPT_V2_NAME}'")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"   Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts

# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# ── 6. Build vectorstore ────────────────────────────────────────────────────
def build_vectorstore():
    print("Building vector store...")
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    chain = (prompt | llm | StrOutputParser())
    answer = chain.invoke({"context": context, "question": question})
    
    return {"question": question, "answer": answer, "version": version}

# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=LANGCHAIN_API_KEY)
    
    # 1. Push
    push_prompts_to_hub(client)
    
    # 2. Pull
    prompts = pull_prompts_from_hub(client)

    # 3. Setup RAG
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    # 4. Run A/B test
    print(f"\nRunning {len(QA_PAIRS)} questions with A/B routing...")
    v1_count = 0
    v2_count = 0
    
    for i, qa in enumerate(QA_PAIRS):
        question = qa["question"]
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        
        if version_tag == "v1": v1_count += 1
        else: v2_count += 1
        
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print(f"\nRouting Summary:")
    print(f"   V1: {v1_count} requests")
    print(f"   V2: {v2_count} requests")
    print(f"\nAll traces sent to LangSmith project '{LANGCHAIN_PROJECT}'")

if __name__ == "__main__":
    main()
