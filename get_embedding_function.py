from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings