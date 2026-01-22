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

DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Origem_Correcao']

# ===============================================
# FUNÇÕES DE BANCO DE DADOS (SQLite)
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
    try:
        conn.execute(query)
        conn.commit()
    except Exception as e:
        st.error(f"Erro ao criar tabela: {e}")

@st.cache_data(hash_funcs={sqlite3.Connection: lambda _: "constant_db_hash"})
def load_geoloc_cache(conn):
    try:
        df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except:
        return pd.DataFrame(columns=CACHE_COLUMNS)

# ===============================================
# FUNÇÕES HELPER E FORMULÁRIO
# ===============================================

def apply_google_coords():
    coord_str = st.session_state.get('form_colar_coord', "")
    if not coord_str: return
    cleaned_str = coord_str.strip().replace(';', ',').replace(' ', ',')
    parts = [p.strip() for p in cleaned_str.split(',') if p.strip()]
    if len(parts) >= 2:
        try:
            st.session_state['form_new_lat_num'] = float(parts[0])
            st.session_state['form_new_lon_num'] = float(parts[1])
            st.session_state['form_colar_coord'] = ""
        except: st.error("Erro no formato das coordenadas.")

# ===============================================
# REPARO DE CSV E CONVERSÃO KML
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
        parts = line.split(',')
        if len(parts) > 11:
            prefix = parts[:4] # AT ID, Sequence, Stop, SPX TN
            suffix = parts[-5:] # Bairro, City, Zip, Lat, Lon
            middle = [", ".join(parts[4:-5]).replace('"', '')] # Address reparado
            new_line = ",".join(prefix + [f'"{middle[0]}"'] + suffix)
            reparsed_data.append(new_line)
        else:
            reparsed_data.append(line)

    return pd.read_csv(io.StringIO('\n'.join(reparsed_data)))

@st.cache_data
def parse_kml_data(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    k = kml.KML()
    try:
        if uploaded_file.name.lower().endswith('.kmz'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as kmz:
                kml_name = [n for n in kmz.namelist() if n.endswith('.kml')][0]
                k.from_string(kmz.read(kml_name).decode('utf-8'))
        else:
            k.from_string(file_bytes.decode('utf-8'))
        
        data = []
        def process_features(features):
            for f in features:
                if isinstance(f, kml.Placemark) and f.geometry:
                    data.append({
                        'Endereco_Completo_Cache': f.name,
                        'Latitude_Corrigida': f.geometry.y,
                        'Longitude_Corrigida': f.geometry.x
                    })
                if hasattr(f, 'features'): process_features(f.features())
        
        process_features(k.features())
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro KML: {e}")
        return pd.DataFrame()

# ===============================================
# LÓGICA DE PROCESSAMENTO (O CORAÇÃO DO CÓDIGO)
# ===============================================

def trim_cidade_cep(endereco_completo):
    if pd.isna(endereco_completo): return None
    partes = str(endereco_completo).strip().upper().split(',')
    if len(partes) >= 3:
        return ','.join(partes[:-2]).strip().replace(', ', ',')
    return str(endereco_completo).upper().replace(', ', ',')

@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade, df_cache_geoloc):
    # MAPEAMENTO FLEXÍVEL (Aceita as colunas novas ou antigas)
    map_cols = {
        'Destination Address': 'Destination Address',
        'Address': 'Destination Address',
        'Sequence': 'Sequence',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'Bairro': 'Bairro',
        'City': 'City',
        'Zipcode/Postal code': 'Zipcode/Postal code'
    }
    df = df_entrada.rename(columns=map_cols).copy()

    # TRATAMENTO DE SEQUÊNCIA (Corrige o erro dos traços '-')
    df['Sequence'] = df['Sequence'].astype(str).replace('-', '0').replace('nan', '0').str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence'], errors='coerce').fillna(0).astype(int)

    # Chave para consulta no Cache
    df['Chave_Busca'] = (df['Destination Address'].astype(str).strip() + ', ' + 
                         df.get('Bairro', "").astype(str).strip()).str.upper()

    # Aplica correções do banco de dados (SQLite)
    if not df_cache_geoloc.empty:
        df_cache_geoloc['Chave_DB'] = df_cache_geoloc['Endereco_Completo_Cache'].apply(trim_cidade_cep)
        df_lookup = df_cache_geoloc.rename(columns={'Latitude_Corrigida': 'C_Lat', 'Longitude_Corrigida': 'C_Lon'})
        df = pd.merge(df, df_lookup[['Chave_DB', 'C_Lat', 'C_Lon']].drop_duplicates('Chave_DB'),
                      left_on='Chave_Busca', right_on='Chave_DB', how='left')
        
        mask = df['C_Lat'].notna()
        df.loc[mask, 'Latitude'] = df.loc[mask, 'C_Lat']
        df.loc[mask, 'Longitude'] = df.loc[mask, 'C_Lon']

    # AGRUPAMENTO PARA O CIRCUIT (Evita paradas duplicadas no mesmo endereço)
    df_agrupado = df.groupby(['Destination Address', 'Latitude', 'Longitude']).agg({
        'Sequence_Num': lambda x: ", ".join([str(i) for i in sorted(list(set(x))) if i > 0]),
        'Bairro': 'first',
        'City': 'first',
        'Zipcode/Postal code': 'first'
    }).reset_index()

    # Contagem de pacotes e notas finais
    counts = df.groupby(['Destination Address', 'Latitude', 'Longitude']).size().reset_index(name='Qtd')
    df_agrupado = df_agrupado.merge(counts, on=['Destination Address', 'Latitude', 'Longitude'])

    df_final = pd.DataFrame()
    df_final['Order ID'] = df_agrupado['Sequence_Num'].replace('', 'S/N')
    df_final['Address'] = df_agrupado['Destination Address'] + ", " + df_agrupado['Bairro'].fillna('')
    df_final['Latitude'] = df_agrupado['Latitude']
    df_final['Longitude'] = df_agrupado['Longitude']
    df_final['Notes'] = df_agrupado.apply(lambda x: f"Pacotes: {x['Qtd']} | {x['City']} | CEP: {x['Zipcode/Postal code']}", axis=1)

    return df_final, df

# ===============================================
# INTERFACE STREAMLIT (UI COMPLETA)
# ===============================================

def main():
    conn = get_db_connection()
    create_table_if_not_exists(conn)
    
    if 'df_kml_extraido' not in st.session_state: st.session_state['df_kml_extraido'] = pd.DataFrame()

    st.title("🚀 Circuit Flow Completo v3.0")
    
    tab1, tab2, tab3 = st.tabs(["📊 Processar Planilha", "🗺️ Cache & KML", "🔧 Configurações"])

    with tab1:
        uploaded_file = st.file_uploader("Suba a planilha (XLSX ou CSV)", type=['xlsx', 'csv'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df_raw = convert_google_maps_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            st.subheader("Prévia dos Dados Originais")
            st.dataframe(df_raw.head(5), use_container_width=True)

            if st.button("🪄 Gerar Planilha para o Circuit"):
                df_cache = load_geoloc_cache(conn)
                df_circuit, _ = processar_e_corrigir_dados(df_raw, 90, df_cache)
                
                st.success(f"Sucesso! {len(df_circuit)} endereços únicos processados.")
                
                # Exibição com AgGrid
                gb = GridOptionsBuilder.from_dataframe(df_circuit)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(editable=True)
                grid_opt = gb.build()
                AgGrid(df_circuit, gridOptions=grid_opt, height=450, theme='streamlit')
                
                # Botão de Download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_circuit.to_excel(writer, index=False)
                st.download_button("📥 Baixar Planilha Pronta", output.getvalue(), 
                                   file_name="Import_Circuit_Pronto.xlsx", 
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab2:
        st.subheader("Importar Pontos via Google Earth (KML/KMZ)")
        kml_file = st.file_uploader("Arraste o arquivo KML/KMZ aqui", type=['kml', 'kmz'])
        if kml_file:
            df_kml = parse_kml_data(kml_file)
            st.dataframe(df_kml)
            if st.button("💾 Salvar pontos no Banco de Dados"):
                for _, row in df_kml.iterrows():
                    conn.execute(f"INSERT OR REPLACE INTO {TABLE_NAME} (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida, Origem_Correcao) VALUES (?, ?, ?, ?)",
                                 (row['Endereco_Completo_Cache'], row['Latitude_Corrigida'], row['Longitude_Corrigida'], 'KML_Import'))
                conn.commit()
                st.success("Cache Geográfico atualizado!")

    with tab3:
        st.subheader("Gerenciamento do Sistema")
        if st.button("🔴 Apagar Todo o Banco de Dados (CUIDADO)"):
            conn.execute(f"DELETE FROM {TABLE_NAME}")
            conn.commit()
            st.warning("Banco de dados limpo.")

if __name__ == "__main__":
    main()
