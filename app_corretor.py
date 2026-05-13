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

# --- CSS para alinhamento ---
st.markdown("""
<style>
.stTextArea [data-baseweb="base-input"], .stTextInput [data-baseweb="base-input"] { text-align: left; font-family: monospace; }
h1, h2, h3, h4, .stMarkdown { text-align: left !important; }
</style>
""", unsafe_allow_html=True)

# --- Configurações de Banco de Dados ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
GMAPS_COL_ADDRESS = 'Destination Address'
GMAPS_COL_BAIRRO = 'Bairro'
GMAPS_COL_CITY = 'City'
GMAPS_COL_LAT = 'Latitude'
GMAPS_COL_LON = 'Longitude'

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

# --- Funções de Processamento ---

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
        def extract_placemarks(features):
            for f in features:
                if hasattr(f, 'features'): extract_placemarks(f.features())
                if isinstance(f, kml.Placemark) and f.geometry:
                    coords = list(f.geometry.coords)[0]
                    data.append({'Endereco_Completo_Cache': f.name, 'Latitude_Corrigida': coords[1], 'Longitude_Corrigida': coords[0]})
        extract_placemarks(k_obj.features())
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro no KML: {e}")
        return pd.DataFrame()

def convert_google_maps_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        content = uploaded_file.read().decode('utf-8')
    except:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('latin-1')
        
    lines = content.strip().splitlines()
    if not lines: return pd.DataFrame()
    
    # Reparo de CSV com excesso de vírgulas
    reparsed = [lines[0]]
    for line in lines[1:]:
        match = re.match(r'(".*?")(,(.*))', line)
        if match:
            wkt, _, rest = match.groups()
            parts = [p.strip() for p in rest.split(',')]
            if len(parts) > 10:
                middle = '"' + ', '.join(parts[4:-5]).replace('"', '') + '"'
                new_line = f"{wkt},{','.join(parts[:4])},{middle},{','.join(parts[-5:])}"
                reparsed.append(new_line)
            else: reparsed.append(line)
    
    df = pd.read_csv(io.StringIO('\n'.join(reparsed)))
    df['Endereco_Completo_Cache'] = df[GMAPS_COL_ADDRESS].astype(str) + ", " + df[GMAPS_COL_BAIRRO].astype(str)
    df['Latitude_Corrigida'] = pd.to_numeric(df[GMAPS_COL_LAT], errors='coerce')
    df['Longitude_Corrigida'] = pd.to_numeric(df[GMAPS_COL_LON], errors='coerce')
    return df[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']].dropna()

# ===============================================
# INTERFACE DO USUÁRIO (O QUE ESTAVA FALTANDO)
# ===============================================

conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("📍 Circuit Flow - Corretor de Geocalização")

tab1, tab2 = st.tabs(["Importar Dados", "Gerenciar Cache"])

with tab1:
    st.subheader("Upload de Arquivos")
    file_type = st.radio("Selecione o tipo de arquivo:", ["Google Maps CSV", "KML / KMZ"])
    uploaded_file = st.file_uploader("Escolha o arquivo", type=['csv', 'kml', 'kmz'])

    if uploaded_file:
        if file_type == "Google Maps CSV":
            df_result = convert_google_maps_csv(uploaded_file)
        else:
            df_result = parse_kml_data(uploaded_file)
        
        st.write(f"Encontrados {len(df_result)} registros.")
        if st.button("Salvar no Banco de Dados"):
            df_result['Origem_Correcao'] = 'Importação'
            df_result.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
            st.success("Dados importados com sucesso!")

with tab2:
    st.subheader("Cache de Localizações")
    df_cache = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    if not df_cache.empty:
        AgGrid(df_cache, editable=True)
    else:
        st.info("O cache está vazio.")
