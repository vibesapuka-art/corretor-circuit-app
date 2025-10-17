# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Circuit Flow Completo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS DE ÚLTIMO RECURSO (Simplificado e em BLOCO ÚNICO <style>) ---
# A injeção de CSS foi consolidada em um bloco st.write() único para contornar
# o TypeError que ocorre no Streamlit Cloud ao usar st.markdown() repetidamente.
st.write("""
<style>
/* Alinha o texto dentro do st.text_area para a esquerda */
textarea {
    text-align: left !important;
    font-family: monospace;
    white-space: pre-wrap;
}
/* Garante que títulos e outros elementos fiquem alinhados à esquerda */
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
# NOVAS COLUNAS
COLUNA_GAIOLA = 'Gaiola' 
COLUNA_ID_UNICO = 'ID_UNICO' # ID temporário: Gaiola-Sequence (Ex: A1-1, G3-1)

# ... (Restante do seu código)
# ... (Funções de processamento, lógica das abas, etc.)
