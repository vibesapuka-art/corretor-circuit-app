import streamlit as st
import pandas as pd

st.set_page_config(page_title="TESTE DE CONEXÃO", layout="wide")

st.title("✅ APLICAÇÃO CONECTADA COM SUCESSO!")
st.success("Se você está vendo esta mensagem, o Streamlit está rodando.")

st.header("Upload de Teste")
st.file_uploader("Arquivo de Teste", type=['csv', 'xlsx'])
