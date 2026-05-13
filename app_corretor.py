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

EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Origem_Correcao']
PRIMARY_KEYS = ['Endereco_Completo_Cache'] 

GMAPS_COL_ADDRESS = 'Destination Address'
GMAPS_COL_BAIRRO = 'Bairro'
GMAPS_COL_CITY = 'City'
GMAPS_COL_ZIPCODE = 'Zipcode/Postal code'
GMAPS_COL_LAT = 'Latitude'
GMAPS_COL_LON = 'Longitude'

# ===============================================
# FUNÇÕES HELPER
# ===============================================

def apply_google_coords():
    coord_str = st.session_state.get('form_colar_coord', "")
    if not coord_str:
        return

    cleaned_str = coord_str.strip().replace(';', ',').replace(' ', ',')
    cleaned_str = re.sub(r',+', ',', cleaned_str)
    parts = cleaned_str.split(',')
    
    numeric_parts = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                float(p)
                numeric_parts.append(p)
            except ValueError:
                continue

    if len(numeric_parts) >= 2:
        try:
            lat = float(numeric_parts[0])
            lon = float(numeric_parts[1])
            if abs(lat) > 90 and abs(lon) <= 90:
                 lat, lon = lon, lat

            if abs(lat) <= 90 and abs(lon) <= 180:
                st.session_state['form_new_lat_num'] = lat
                st.session_state['form_new_lon_num'] = lon
                st.session_state['form_colar_coord'] = ""
                st.success(f"Coordenadas aplicadas: Lat {lat:.8f}, Lon {lon:.8f}")
            else:
                 st.error("Coordenadas inválidas.")
        except ValueError:
            st.error("Erro na conversão numérica.")
    else:
        st.error("Coordenadas não encontradas.")

def clear_lat_lon_fields():
    for key in ['form_new_endereco', 'form_colar_coord', 'form_new_lat_num', 'form_new_lon_num']:
        if key in st.session_state:
            st.session_state[key] = "" if "num" not in key else 0.0
    st.success("Campos limpos.")

# ===============================================
# BANCO DE DADOS
# ===============================================

@st.cache_resource
def get_db_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False, timeout=10)

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

@st.cache_data(hash_funcs={sqlite3.Connection: lambda _: "constant_db_hash"})
def load_geoloc_cache(conn):
    try:
        df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except:
        return pd.DataFrame(columns=CACHE_COLUMNS)

def save_single_entry_to_db(conn, endereco, lat, lon, origem='Manual'):
    query = f"INSERT OR REPLACE INTO {TABLE_NAME} (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida, Origem_Correcao) VALUES (?, ?, ?, ?);"
    conn.execute(query, (endereco, lat, lon, origem))
    conn.commit()
    load_geoloc_cache.clear()
    st.rerun()

# ===============================================
# KML / KMZ
# ===============================================

@st.cache_data
def parse_kml_data(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    k = kml.KML()
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as kmz:
                kml_names = [n for n in kmz.namelist() if n.endswith('.kml')]
                k.from_string(kmz.read(kml_names[0]).decode('utf-8'))
        else:
            k.from_string(file_bytes.decode('utf-8'))
    except Exception as e:
        st.error(f"Erro KML: {e}")
        return pd.DataFrame()

    data = []
    def extract_placemarks(features):
        for f in features:
            if hasattr(f, 'features'):
                extract_placemarks(f.features())
            if isinstance(f, kml.Placemark) and f.geometry:
                coords = list(f.geometry.coords)[0]
                data.append({'Endereco_Completo_Cache': f.name, 'Longitude_Corrigida': coords[0], 'Latitude_Corrigida': coords[1]})

    extract_placemarks(k.features())
    return pd.DataFrame(data)

# ===============================================
# GOOGLE MAPS CSV REPAIR (FIXED)
# ===============================================

@st.cache_data
def convert_google_maps_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        content = uploaded_file.read().decode('utf-8')
    except:
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
            wkt, _, rest = match.groups()
            parts = [p.strip() for p in rest.split(',')]
            if len(parts) > 10: # Se houver excesso de vírgulas no endereço
                prefix = parts[:4]
                suffix = parts[-5:]
                middle = '"' + ', '.join(parts[4:-5]).replace('"', '') + '"'
                new_line = f"{wkt},{','.join(prefix)},{middle},{','.join(suffix)}"
                reparsed_data.append(new_line)
            else:
                reparsed_data.append(line)
        else:
            reparsed_data.append(line)

    df = pd.read_csv(io.StringIO('\n'.join(reparsed_data)))
    
    # Normalização dos nomes para o Cache
    df['Endereco_Completo_Cache'] = df[GMAPS_COL_ADDRESS].astype(str).str.strip()
    # Adiciona bairro e cidade se existirem para tornar a chave única
    if GMAPS_COL_BAIRRO in df.columns:
        df['Endereco_Completo_Cache'] += ", " + df[GMAPS_COL_BAIRRO].astype(str)
    
    df['Latitude_Corrigida'] = df[GMAPS_COL_LAT]
    df['Longitude_Corrigida'] = df[GMAPS_COL_LON]
    df['Origem_Correcao'] = 'GoogleMaps_Import'
    
    return df[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Origem_Correcao']]

# --- Exemplo de inicialização simples ---
conn = get_db_connection()
create_table_if_not_exists(conn)
st.title("Circuit Flow - Sistema de Geocodificação")
st.write("Banco de dados pronto para uso.")
