import argparse
from dotenv import load_dotenv
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import wandb

load_dotenv()

CHROMA_ROOT_PATH = "chroma"
DATA_ROOT_PATH = "pdf"
WANDB_API = os.getenv("WANDB_API")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the databases.")
    parser.add_argument("--wandb_project", type=str, default="chroma_db_artifacts", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default="process_chroma_dbs", help="WandB run name.")
    args = parser.parse_args()

    if not WANDB_API:
        print("Erro: Variável de ambiente WANDB_API não foi definida.")
        return

    wandb.login(key=WANDB_API)
    with wandb.init(project=args.wandb_project, name=args.wandb_run_name) as run:
        if args.reset:
            print("Clearing all Chromas")
            clear_all_databases()

        for city_folder in os.listdir(DATA_ROOT_PATH):
            city_path = os.path.join(DATA_ROOT_PATH, city_folder)

            if os.path.isdir(city_path):
                print(f"🔄 Processando a cidade: {city_folder}")
                process_city(city_folder, city_path)

        # Save the entire Chroma directory as an artifact
        artifact = wandb.Artifact(
            "chroma_databases",
            type="dataset",
            description="Chroma databases for different cities.",
        )
        artifact.add_dir(CHROMA_ROOT_PATH, "chroma")
        run.log_artifact(artifact)
        artifact.wait()
        print(f"💾 Chroma databases saved as WandB artifact: {artifact.name}:{artifact.version}")

def process_city(city_name: str, city_path: str):
    """
    Processa uma subpasta de cidade, criando ou atualizando o Chroma correspondente.
    """
    chroma_city_path = os.path.join(CHROMA_ROOT_PATH, f"{city_name}")

    documents = load_documents(city_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks, chroma_city_path)

def load_documents(city_path: str):
    """
    Carrega documentos PDF da subpasta de uma cidade.
    """
    document_loader = PyPDFDirectoryLoader(city_path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """
    Divide os documentos em chunks menores.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], chroma_path: str):
    """
    Adiciona ou atualiza os documentos no Chroma específico da cidade.
    """
    db = Chroma(
        persist_directory=chroma_path, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Número de documentos no banco de dados '{chroma_path}': {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids
    ]

    if len(new_chunks):
        print(
            f"👉 Adicionando {len(new_chunks)} novo(s) documento(s) ao banco '{chroma_path}'"
        )
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    else:
        print(f"✅ Nenhum novo documento para adicionar ao banco '{chroma_path}'")

def calculate_chunk_ids(chunks):
    """
    Calcula IDs únicos para cada chunk com base na fonte e na página.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_all_databases():
    """
    Remove todos os bancos de dados Chroma existentes.
    """
    if os.path.exists(CHROMA_ROOT_PATH):
        shutil.rmtree(CHROMA_ROOT_PATH)

if __name__ == "__main__":
    main()