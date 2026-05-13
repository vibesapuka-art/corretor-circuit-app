# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
import math
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode
from fastkml import kml
import zipfile 

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Circuit Flow Completo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS para garantir alinhamento à esquerda ---
st.markdown("""
<style>
.stTextArea [data-baseweb="base-input"], 
.stTextInput [data-baseweb="base-input"] {
    text-align: left;
    font-family: monospace;
}
div.stTextArea > label,
div.stTextInput > label {
    text-align: left !important; 
}
div[data-testid="stTextarea"] textarea {
    text-align: left !important; 
    font-family: monospace;
    white-space: pre-wrap;
}
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}
.ag-header-cell-text {
    white-space: normal !important;
    line-height: 1.2 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Configurações Globais ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
COLUNA_BAIRRO = 'Bairro' 

COLUNA_ADDRESS_CIRCUIT = 'address' 
COLUNA_NOTES_CIRCUIT = 'notes'

DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Origem_Correcao']

GMAPS_COL_ADDRESS = 'Destination Address'
GMAPS_COL_BAIRRO = 'Bairro'
GMAPS_COL_CITY = 'City'
GMAPS_COL_ZIPCODE = 'Zipcode/Postal code'
GMAPS_COL_LAT = 'Latitude'
GMAPS_COL_LON = 'Longitude'

# ===============================================
# FUNÇÕES DE BANCO DE DADOS
# ===============================================

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=10)
    return conn

def create_table_if_not_exists(conn):
    query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        Endereco_Completo_Cache TEXT PRIMARY KEY,
        Latitude_Corrigida REAL,
        Longitude_Corrigida REAL,
        Origem_Correcao TEXT DEFAULT 'Manual'
    );
    """
    conn.execute(query)
    conn.commit()

# ===============================================
# FUNÇÕES DE PROCESSAMENTO (CORRIGIDAS)
# ===============================================

@st.cache_data
def convert_google_maps_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('latin-1')
        
    lines = content.strip().splitlines()
    if not lines: return pd.DataFrame()

    header = lines[0]
    reparsed_data = [header] 

    for line in lines[1:]:
        if not line.strip(): continue
        match = re.match(r'(".*?")(,(.*))', line)
        if match:
            wkt_col, _, rest_of_line = match.groups()
            parts = [p.strip() for p in rest_of_line.split(',')]
            if len(parts) > 10:
                prefix = parts[0:4]
                suffix = parts[-5:]
                middle_parts = parts[4:len(parts)-5]
                destination_address = '"' + ', '.join(middle_parts).replace('"', '') + '"'
                new_line = f"{wkt_col},{','.join(prefix)},{destination_address},{','.join(suffix)}"
                reparsed_data.append(new_line)
            else:
                reparsed_data.append(line)
        else:
            reparsed_data.append(line)

    df = pd.read_csv(io.StringIO('\n'.join(reparsed_data)))
    
    # Lógica de concatenação de endereço completa
    def build_address(row):
        addr = str(row[GMAPS_COL_ADDRESS]).strip()
        bairro = str(row.get(GMAPS_COL_BAIRRO, "")).strip()
        city = str(row.get(GMAPS_COL_CITY, "")).strip()
        full = addr
        if bairro: full += f", {bairro}"
        if city and city not in full: full += f", {city}"
        return full

    df['Endereco_Completo_Cache'] = df.apply(build_address, axis=1)
    df['Latitude_Corrigida'] = pd.to_numeric(df[GMAPS_COL_LAT], errors='coerce')
    df['Longitude_Corrigida'] = pd.to_numeric(df[GMAPS_COL_LON], errors='coerce')
    return df[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']]

@st.cache_data
def parse_kml_data(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    k_obj = kml.KML()
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as kmz:
                kml_name = [n for n in kmz.namelist() if n.endswith('.kml')][0]
                k_obj.from_string(kmz.read(kml_name).decode('utf-8'))
        else:
            k_obj.from_string(file_bytes.decode('utf-8'))
        
        data = []
        def extract_recursive(features):
            for f in features:
                if hasattr(f, 'features'): extract_recursive(f.features())
                if isinstance(f, kml.Placemark) and f.geometry:
                    coords = list(f.geometry.coords)[0]
                    data.append({'Endereco_Completo_Cache': f.name, 'Latitude_Corrigida': coords[1], 'Longitude_Corrigida': coords[0]})
        extract_recursive(k_obj.features())
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro KML: {e}")
        return pd.DataFrame()

# ===============================================
# INTERFACE PRINCIPAL (UI) - ESSENCIAL PARA APARECER AS OPÇÕES
# ===============================================

conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("🚀 Circuit Flow - Sistema de Geocodificação")

aba1, aba2 = st.tabs(["📤 Importar Dados", "💾 Gerenciar Banco"])

with aba1:
    st.subheader("Importação de Arquivos")
    tipo = st.selectbox("Escolha o formato:", ["Google Maps CSV", "KML / KMZ"])
    arquivo = st.file_uploader("Upload do arquivo", type=['csv', 'kml', 'kmz'])

    if arquivo:
        if tipo == "Google Maps CSV":
            df_processado = convert_google_maps_csv(arquivo)
        else:
            df_processado = parse_kml_data(arquivo)
        
        if not df_processado.empty:
            st.success(f"{len(df_processado)} endereços processados!")
            st.dataframe(df_processado, use_container_width=True)
            if st.button("Salvar no Cache"):
                df_processado['Origem_Correcao'] = 'Importação'
                df_processado.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
                st.balloons()
        else:
            st.warning("Nenhum dado válido encontrado.")

with aba2:
    st.subheader("Conteúdo do Cache")
    df_cache = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    if not df_cache.empty:
        AgGrid(df_cache, editable=True, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
    else:
        st.info("O banco de dados está vazio.")
