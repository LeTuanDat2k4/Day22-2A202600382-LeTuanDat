import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "day22-lab-rag")

# OpenAI / LLM Configuration
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE      = os.getenv("OPENAI_API_BASE")  # Optional: for custom endpoints
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def check_config():
    """Verify that all required environment variables are set."""
    missing = []
    if not LANGCHAIN_API_KEY: missing.append("LANGCHAIN_API_KEY")
    if not OPENAI_API_KEY:    missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGCHAIN_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_API_BASE or 'default'}")
    print(f"   Default LLM model : {LLM_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")
    return True

if __name__ == "__main__":
    check_config()
