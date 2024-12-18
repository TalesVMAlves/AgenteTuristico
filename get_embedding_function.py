from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
    return embeddings
