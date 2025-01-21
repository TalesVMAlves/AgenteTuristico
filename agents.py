import os
from datetime import datetime
from langchain import hub
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.tools.render import render_text_description
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import streamlit as st

from planing_tools import weatherapi_forecast_periods, query_rag
from calendar_tools import list_calendar_list, list_calendar_events, insert_calendar_event, create_calendar
from dotenv import load_dotenv

hoje = datetime.today()

data_formatada = hoje.strftime("%Y/%m/%d")
# Pega o dia da semana em inglês
dia_da_semana_ingles = hoje.strftime("%A")

# Dicionário para mapear os dias da semana de inglês para português
dias_da_semana_pt = {
    "Monday": "segunda-feira",
    "Tuesday": "terça-feira",
    "Wednesday": "quarta-feira",
    "Thursday": "quinta-feira",
    "Friday": "sexta-feira",
    "Saturday": "sábado",
    "Sunday": "domingo"
}

# Traduz o dia da semana para português
dia_da_semana_pt = dias_da_semana_pt.get(dia_da_semana_ingles, dia_da_semana_ingles)

# Formata a data para utilizar no prompt
data_atual = f"{dia_da_semana_pt}, {data_formatada}"

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.6,
    max_tokens= 1000,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)

def transfer_to_calendar_agent(input_str):
    return calendar_agent_executor.invoke({"input": input_str})
def transfer_to_travel_agent(input_str):
    return travel_agent_executor.invoke({"input": input_str})

ddg_search = DuckDuckGoSearchAPIWrapper()

travel_planing_tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="""Essa ferramenta DEVE ser utilizada para buscar eventos relevantes no período fornecido pelo usuário. 
        Ela é útil para obter informações sobre eventos ou atividades especiais que estão acontecendo na cidade de destino nas datas que o usuário informou. 
        O modelo deve usá-la para complementar as sugestões de atividades."""
    ),
    Tool(
        name="Weather Forecast",
        func=lambda date_string: weatherapi_forecast_periods(date_string, st.session_state.selected_destino),
        description="""Esta ferramenta DEVE ser usada obrigatoriamente *antes* de gerar o roteiro turístico, e somente após coletar todas as informações necessárias do usuário, incluindo o intervalo exato de datas. 
        A consulta do clima deve ser feita separadamente para cada dia do período informado, garantindo que as atividades planejadas no roteiro sejam compatíveis com as condições climáticas previstas. 

        **Quando usar:**
        - O clima é um fator relevante para a definição de atividades no roteiro.
        - O usuário especificou o intervalo de datas para o planejamento.

        **Instruções:**
        1. Certifique-se de coletar as datas exatas do período solicitado pelo usuário.
        2. Consulte a previsão do tempo para cada dia separadamente no intervalo de datas fornecido.
        3. Use o resultado da previsão para ajustar o planejamento das atividades de acordo com as condições climáticas.

        **Exemplo de uso:**
        - Entrada do usuário: "Planeje um roteiro entre os dias 1º de agosto e 4 de março."
        - Ações do modelo:
            Action input: 2025/08/01
            Action input: 2025/08/02
            Action input: 2025/08/03
            Action input: 2025/08/04
    """
    ),
    Tool(
        name="Query RAG",
        func=lambda query_text: query_rag(query_text, st.session_state.selected_destino, st.session_state.chroma_city_path),
        description="""Esta ferramenta deve ser usada quando o modelo souber a cidade de destino e os interesses do usuário, com o objetivo de fornecer informações sobre pontos turísticos e atrações que se alinham com esses interesses. 
        O modelo deve utilizar essa ferramenta para sugerir atividades e lugares específicos a visitar, baseados na cidade e nos interesses fornecidos."""
    ),
    Tool(
        name="Calendar Agent",
        func=transfer_to_calendar_agent,
        description="Esse agente lida com tudo relacionado ao calendário. As mensagens enviadas a ele devem estar em Português."
    ),
]
from calendar_tools import list_calendar_list, list_calendar_events, insert_calendar_event, create_calendar

google_calendar_tools = [
    Tool(
        name="Create Calendar List",
        func=create_calendar,
        description="""
        Use a função create_calendar_list para criar um novo calendário.
        Forneça o resumo (título) do calendário que você deseja criar.
        - Exemplo de uso: 'Meu Calendário de Viagens'
        - Essa função vai criar um novo calendário com o resumo fornecido.
        """
    ),
    Tool(
        name="List Calendar List",
        func=list_calendar_list,
        description="""
        Use a função list_calendar_list para obter a lista dos seus calendários disponíveis no Google Calendar.
        Se você quiser limitar o número de calendários retornados, forneça o número máximo desejado.
        - Exemplo de uso: Forneça o número máximo de calendários a serem listados, como 50. Se não quiser um limite, apenas diga para listar 200 o máximo por padrão.
        """
    ),
    Tool(
        name="List Calendar Events",
        func=list_calendar_events,
        description="""
        Use a função list_calendar_events para obter a lista de eventos de um calendário específico.
        Você precisa fornecer o ID do calendário e, opcionalmente, o número máximo de eventos a serem listados.
        - Exemplo de uso para o calendário principal: forneça 'primary' como o ID do calendário.
        - Exemplo de uso com limite: forneça o ID do calendário e o número máximo de eventos, como 'primary', 20.
        - Para listar eventos de um calendário específico, primeiro liste seus calendários para encontrar o ID correto.
        """
    ),
    Tool(
    name="Insert Calendar Event",
    func=insert_calendar_event,
    description="""
        Use a função Insert Calendar Event para adicionar **um único evento** a um calendário Google.
        Esta função deve ser chamada **uma vez para cada evento** que precisa ser agendado.

        A entrada para esta função deve ser uma descrição clara e completa de **um único evento**
        em linguagem natural, contendo todas as informações necessárias para agendá-lo individualmente.

        Se o usuário pediu para criar um novo calendário, você DEVE utilizar o calendar_id desse novo calendário para a inserção dos eventos. 
        
        **Formato de entrada esperado:**

        **Descrição em Linguagem Natural (obrigatório):**
        Forneça uma descrição textual detalhada para **um único evento**. Inclua o calendar_id (se fornecido pelo usuário), o que será feito,
        quando (data e hora específicas), onde (local exato) e quaisquer detalhes adicionais relevantes
        para esse evento em particular.

        **Exemplos de descrições de eventos individuais:**
        - "Agendar almoço com a Maria na sexta-feira ao meio-dia no restaurante X."
        - "Criar um compromisso para pagar as contas no dia 10 de janeiro às 9h da manhã."
        - "Bloquear minha agenda para foco no projeto Y na próxima segunda-feira, das 14h às 17h, na minha mesa."

        **Como usar esta função corretamente:**

        1. **Identifique cada evento individualmente:** Se a intenção é agendar múltiplos eventos,
           trate cada um como uma chamada separada para esta função.
        2. **Obtenha o ID do calendário (se necessário):** Se o usuário mencionar um nome de
           calendário específico, use a função 'list_calendar_list' **primeiro** para obter
           o 'calendar_id' correspondente.
        3. **Chame esta função com a descrição de um único evento:** Forneça a descrição completa
           em linguagem natural para o evento específico que você deseja agendar.

        **Informações essenciais para cada descrição de evento:**
        - **O que:** Uma descrição concisa do evento (ex: "Reunião de equipe", "Consulta médica").
        - **Quando:** A data e hora de início e fim **precisas** do evento.
        - **Onde:** O local **específico** do evento.
        - **Fuso Horário (opcional):** Se aplicável, mencione o fuso horário se for diferente
          de "America/Fortaleza".

        **Exemplo de uso**

        *Usuário: Agendar visita ao Aquário Natal sábado das 9h às 12h.*
        *Agente:* (Chama 'Insert Calendar Event' com a Action Input: "Agendar visita ao Aquário Natal sábado das 9h às 12h.")

        **Observação IMPORTANTE:** Esta função **NÃO** deve ser usada para criar múltiplos eventos
        com uma única chamada. Cada chamada deve corresponder a **um único evento**.

        Ao final da execução desta função, você DEVE fornecer ao usuário a confirmação do agendamento
        desse evento específico, incluindo o título, data, hora.

        No final forneça o link para o calendário google: [https://www.google.com/calendar]
    """
    )
]

travel_planing_prompt = hub.pull("tales/agente_turismo")
google_calendar_prompt = hub.pull("tales/agente_calendario")

llm_with_stop = llm.bind(stop=["\nObservation"])

planing_prompt = travel_planing_prompt.partial(
    tools=render_text_description(travel_planing_tools),
    tool_names=", ".join([t.name for t in travel_planing_tools]),
)

calendar_prompt = google_calendar_prompt.partial(
    tools=render_text_description(google_calendar_tools),
    tool_names=", ".join([t.name for t in google_calendar_tools]),
)

history = ChatMessageHistory()
memory = ConversationBufferWindowMemory(
    k=20, 
    chat_memory=history, 
    memory_key="chat_history",
    input_key="input",
    other_memory_key=["destino"])

travel_planing_agent = (
    {
        "input": lambda x: x["input"],
        "destino": lambda x: x.get("destino"),
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
        "data_atual": lambda x: data_atual,
    }
    | planing_prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

travel_agent_executor = AgentExecutor(agent=travel_planing_agent, tools=travel_planing_tools, verbose=True, memory=memory, handle_parsing_errors=True)

google_calendar_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
        "data_atual": lambda x: data_atual,
    }
    | calendar_prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

calendar_agent_executor = AgentExecutor(agent=google_calendar_agent, tools=google_calendar_tools, verbose=True, memory=memory)