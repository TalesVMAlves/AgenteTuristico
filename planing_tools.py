from dotenv import load_dotenv
import os
import requests
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import wandb

load_dotenv()

WEATHER_API = os.getenv('WEATHER_API')
BASE_URL = "http://api.weatherapi.com/v1/forecast.json"
WANDB_PROJECT = "tales-v-m-alves/chroma_db_artifacts"
ARTIFACT_NAME = "chroma_databases-v0"

def weatherapi_forecast_periods(date_string: str, destino: str) -> str:
    """
    Obtém a previsão do tempo para a cidade da cidade destino em uma data específica,
    separada em manhã, tarde e noite.

    Args:
        date_string (str): Uma string contendo a data no formato yyyy-mm-dd.

    Returns:
        str: Uma string contendo as previsões separadas por períodos para a data especificada.
    """
    try:
        params = {
            "key": WEATHER_API,
            "q": destino.capitalize(),
            "dt": date_string,
            "aqi": "no",
            "alerts": "no",
            "lang": "pt"
        }
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if data and data.get("forecast") and data["forecast"].get("forecastday"):
            forecast_data = data["forecast"]["forecastday"][0]
            hourly_data = forecast_data.get("hour", [])

            periods = {
                "Manhã": range(5, 12),
                "Tarde": range(13, 18),
                "Noite": range(18, 24)
            }

            result = f"Previsão para {date_string} em {destino.capitalize()}:\n"
            for period, hours in periods.items():
                result += f"\n{period}:\n"
                filtered_data = [
                    {
                        "hora": hour["time"].split(" ")[1],
                        "temperatura": hour["temp_c"],
                        "condicao": hour["condition"]["text"],
                        "chance_de_chuva": hour["chance_of_rain"],
                        "umidade": hour["humidity"]
                    }
                    for hour in hourly_data if int(hour["time"].split(" ")[1].split(":")[0]) in hours
                ]
                if filtered_data:
                    for hour in filtered_data:
                        result += (
                            f"Hora: {hour['hora']} - "
                            f"Temp: {hour['temperatura']}°C, "
                            f"Condição: {hour['condicao']}, "
                            f"Chance de chuva: {hour['chance_de_chuva']}%, "
                            f"Umidade: {hour['umidade']}%\n"
                        )
                else:
                    result += "  Nenhuma previsão encontrada para este período.\n"

            return result
        else:
            return "Previsão não encontrada para essa data."
    except requests.exceptions.RequestException as e:
        return f"Erro ao buscar informações meteorológicas: {str(e)}"
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

def download_artifact(artifact_name: str, project_path: str) -> str:
    """Downloads the specified artifact from WandB and returns the local directory."""
    api = wandb.Api()
    artifact = api.artifact(f"{project_path}/{artifact_name}:latest")
    download_path = artifact.download()
    return download_path

def query_rag(query_text: str, destino: str) -> str:
    embedding_function = get_embedding_function()

    artifact_dir = os.path.join("artifacts", ARTIFACT_NAME, "chroma")
    if not os.path.exists(artifact_dir):
        os.makedirs("artifacts", exist_ok=True)
        print(f"📥 Downloading WandB artifact '{ARTIFACT_NAME}'...")
        download_path = download_artifact(ARTIFACT_NAME, WANDB_PROJECT)
        print(f"✅ Artifact downloaded to: {download_path}")
        artifact_dir = os.path.join("artifacts", ARTIFACT_NAME, "chroma")
    else:
        print(f"📦 Using existing artifact at: {artifact_dir}")

    chroma_city_path = os.path.join(artifact_dir, destino)
    print(chroma_city_path)
    if not os.path.exists(chroma_city_path):
        return f"Banco de dados para '{destino}' não encontrado no artefato."

    db = Chroma(persist_directory=chroma_city_path, embedding_function=embedding_function)

    results = db.similarity_search_with_score(f"{destino}: {query_text}", k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text