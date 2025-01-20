from dotenv import load_dotenv
import os
import requests
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

load_dotenv()


CHROMA_PATH = "chroma"
WEATHER_API = os.getenv('WEATHER_API')
BASE_URL = "http://api.weatherapi.com/v1/forecast.json"

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

def query_rag(query_text: str, destino: str) -> str:
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=f"{CHROMA_PATH}/{destino}", embedding_function=embedding_function)

    
    results = db.similarity_search_with_score(f"{destino}: {query_text}", k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text