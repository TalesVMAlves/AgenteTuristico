import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import time
from datetime import datetime
from get_embedding_function import get_embedding_function
import os
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Responda a pergunta em português baseado apenas no seguinte contexto:
{context}

Responda a pergunta em português baseado apenas no contexto acima: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Texto da pesquisa.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Check if the input is a file path and read the content if true
    if os.path.isfile(query_text):
        with open(query_text, 'r') as file:
            query_text = file.read()
    # Prepare the DB.
    start_time = time.time()
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="llama3.1")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    formatted_response = (
        f"Resposta: {response_text}\nFontes: {sources}"
        f"\nTempo de Execução: {time.time() - start_time:.2f} segundos"
    )
    print(formatted_response)
    # Create a filename based on the current date and time.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resposta/resposta_{timestamp}.txt"

    with open(filename, "w") as file:
        file.write(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

