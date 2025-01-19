import streamlit as st
from agents import travel_agent_executor
from unidecode import unidecode

DESTINOS = {
    "Natal": "natal",
    "Caicó": "caico",
    "Pipa": "pipa"
}

if __name__ == '__main__':

    st.title('Agente Turístico')
    st.sidebar.title('Escolha um destino')
    
    destino_selecionado = st.sidebar.selectbox('Destino', list(DESTINOS.keys()))
    
    if destino_selecionado:
        st.write(f"Você selecionou {destino_selecionado}")
        st.session_state.selected_destino = DESTINOS[destino_selecionado]

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input('Digite a sua mensagem.'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('user', avatar='🧑‍💻'):
            st.markdown(prompt)

        with st.chat_message('ai', avatar='🤖'):
            # Passa o destino selecionado como contexto para o agente
            response = travel_agent_executor.invoke({
                "input": prompt,
                "destino": unidecode(DESTINOS[destino_selecionado].lower())
            })
            agent_response = response['output']
            st.markdown(agent_response)

        st.session_state.messages.append({'role': 'assistant', 'content': agent_response})