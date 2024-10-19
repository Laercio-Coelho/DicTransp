import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

from loaders import *

API_KEY_GROQ = ['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']

TIPOS_ARQUIVOS_VALIDOS = ['Pdf']

CONFIG_MODELOS = {'Groq': 
                        {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
                         'chat': ChatGroq}}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado "DicTransp".
    Você foi criado e disponibilizado pela empresa LONTANO TRANSPORTES.
    Você possui acesso às seguintes informações vindas 
    de um documento desta empresa: {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o aplicativo!'''.format(tipo_arquivo, documento)

    #print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖Bem-vindo ao ChatBot do "DicTransp" da LONTANO Transportes', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Inicialize o chatbot.')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Faça uma pergunta')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
            }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Dicionário do Transporte','Parâmetros'])
    with tabs[0]:
        sobre = st.header('Criado pela GS Tecnologia da Informação')
        arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['.pdf'])
    with tabs[1]:
        parametros = st.header('Parâmetros de Funcionamento')
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        provedor = st.selectbox('Selecione um provedor para o chat', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo de linguagem', CONFIG_MODELOS[provedor]['modelos'])
        #api_key = st.selectbox('Selecione a API KEY da Groq', API_KEY_GROQ)
        api_key = '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'
        os.environ[API_KEY_GROQ] = api_key
        st.session_state[f'api_key_{provedor}'] = api_key
    
    if st.button('Inicializar Chatbot', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
