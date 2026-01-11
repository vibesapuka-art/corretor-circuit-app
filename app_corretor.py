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

# --- CSS para garantir alinhamento à esquerda em TEXT AREAS e Checkboxes ---
st.markdown("""
<style>
/* Estilo para garantir alinhamento à esquerda em textareas e inputs */
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
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
COLUNA_BAIRRO = 'Bairro' 

# Colunas esperadas no arquivo de Pós-Roteirização (Saída do Circuit)
COLUNA_ADDRESS_CIRCUIT = 'address' 
COLUNA_NOTES_CIRCUIT = 'notes'


# --- Configurações de MIME Type ---
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# --- Configurações de Banco de Dados ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Origem_Correcao']
PRIMARY_KEYS = ['Endereco_Completo_Cache'] 

# Colunas esperadas no CSV de exportação do Google Maps
GMAPS_COL_ADDRESS = 'Destination Address'
GMAPS_COL_BAIRRO = 'Bairro'
GMAPS_COL_CITY = 'City'
GMAPS_COL_ZIPCODE = 'Zipcode/Postal code'
GMAPS_COL_LAT = 'Latitude'
GMAPS_COL_LON = 'Longitude'


# ===============================================
# FUNÇÕES HELPER (CALLBACKS DE FORMULÁRIO)
# ===============================================

def apply_google_coords():
    """Converte string de coordenadas (Lat, Lon) para os campos numéricos do formulário."""
    coord_str = st.session_state.get('form_colar_coord', "")
    if not coord_str:
        return

    # Limpeza e tentativa de extração de dois números
    # Permite vírgula ou espaço ou ponto e vírgula como separador, e ponto como decimal
    cleaned_str = coord_str.strip().replace(';', ',').replace(' ', ',')
    cleaned_str = re.sub(r',+', ',', cleaned_str)
    
    parts = cleaned_str.split(',')
    
    # Filtra por partes que se parecem com floats (pode ter um sinal de menos)
    numeric_parts = []
    for p in parts:
        p = p.strip()
        if p:
            # Tenta converter para float para garantir que é um número válido
            try:
                float(p)
                numeric_parts.append(p)
            except ValueError:
                continue

    if len(numeric_parts) >= 2:
        try:
            # Assumimos o padrão Lat, Lon (mais comum no Google Maps)
            lat = float(numeric_parts[0])
            lon = float(numeric_parts[1])
            
            # Validação simples: Lat entre -90/90, Lon entre -180/180.
            # Se Lat > 90, assume que o usuário inverteu e tenta a correção.
            if abs(lat) > 90 and abs(lon) <= 90:
                 lat_temp = lat
                 lat = lon
                 lon = lat_temp

            if abs(lat) <= 90 and abs(lon) <= 180:
                st.session_state['form_new_lat_num'] = lat
                st.session_state['form_new_lon_num'] = lon
                st.session_state['form_colar_coord'] = "" # Limpa o campo de texto
                st.success(f"Coordenadas aplicadas: Lat {lat:.8f}, Lon {lon:.8f}")
            else:
                 st.error("Coordenadas inválidas detectadas. Verifique a ordem ou se os valores são válidos.")
                 
        except ValueError:
            st.error("Formato de coordenada inválido. Certifique-se de usar ponto para decimais e separador (vírgula ou espaço) entre Lat e Lon.")
    else:
        st.error("Não foi possível encontrar duas coordenadas válidas (Latitude e Longitude) na string colada.")

def clear_lat_lon_fields():
    """Limpa todos os campos do formulário de entrada manual de cache."""
    if 'form_new_endereco' in st.session_state:
        st.session_state['form_new_endereco'] = ""
    if 'form_colar_coord' in st.session_state:
        st.session_state['form_colar_coord'] = ""
    if 'form_new_lat_num' in st.session_state:
        st.session_state['form_new_lat_num'] = 0.0
    if 'form_new_lon_num' in st.session_state:
        st.session_state['form_new_lon_num'] = 0.0
    st.success("Formulário de correção limpo.")


# ===============================================
# FUNÇÕES DE BANCO DE Dados (SQLite)
# ===============================================

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=10)
    return conn

def create_table_if_not_exists(conn):
    pk_str = ', '.join(PRIMARY_KEYS)
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
        
        if 'Origem_Correcao' not in df_cache.columns:
            conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN Origem_Correcao TEXT DEFAULT 'Manual'")
            conn.commit()
            df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn) 
            
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except pd.io.sql.DatabaseError:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de geolocalização: {e}")
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_single_entry_to_db(conn, endereco, lat, lon, origem='Manual'):
    upsert_query = f"""
    INSERT OR REPLACE INTO {TABLE_NAME} 
    (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida, Origem_Correcao) 
    VALUES (?, ?, ?, ?);
    """
    try:
        conn.execute(upsert_query, (endereco, lat, lon, origem))
        conn.commit()
        st.success(f"Correção salva para: **{endereco}** (Origem: {origem}).")
        load_geoloc_cache.clear() 
        st.rerun() 
    except Exception as e:
        st.error(f"Erro ao salvar a correção no banco de dados: {e}")
        
def clear_geoloc_cache_db(conn):
    query = f"DELETE FROM {TABLE_NAME};"
    try:
        conn.execute(query)
        conn.commit()
        load_geoloc_cache.clear()
        st.success("✅ **Sucesso!** Todos os dados do cache de geolocalização foram excluídos permanentemente.")
        st.rerun() 
    except Exception as e:
        st.error(f"❌ Erro ao limpar o cache: {e}")

def export_cache(df_cache, file_format='xlsx'):
    """Exporta o DataFrame de cache em XLSX ou CSV, garantindo o separador correto."""
    
    df_export = df_cache[CACHE_COLUMNS].copy()
    
    # Garantir que Lat/Lon usem ponto para CSV e 8 casas decimais
    df_export['Latitude_Corrigida'] = pd.to_numeric(df_export['Latitude_Corrigida'], errors='coerce').round(8)
    df_export['Longitude_Corrigida'] = pd.to_numeric(df_export['Longitude_Corrigida'], errors='coerce').round(8)
    
    buffer = io.BytesIO()
    
    if file_format == 'xlsx':
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer: 
            df_export.to_excel(writer, index=False, sheet_name='Cache_Geolocalizacao')
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "cache_geolocalizacao_backup.xlsx"
    
    elif file_format == 'csv':
        # Cria o CSV no buffer com separador "," (vírgula)
        df_export.to_csv(buffer, index=False, sep=',', encoding='utf-8')
        mime = "text/csv"
        filename = "cache_geolocalizacao_backup.csv"
        
    else:
        raise ValueError("Formato de arquivo não suportado para exportação.")
        
    buffer.seek(0)
    return buffer, mime, filename


# ===============================================
# FUNÇÕES DE KML/KMZ/XML
# ===============================================

@st.cache_data
def parse_kml_data(uploaded_file):
    """Lê um arquivo KML, KMZ ou XML e extrai nome (Endereço), Lat e Lon dos PlaceMarks."""
    
    file_bytes = uploaded_file.getvalue()
    k = kml.KML()
    
    is_kmz = uploaded_file.name.lower().endswith('.kmz')
    
    try:
        if is_kmz:
            # Tenta o método moderno/comum (from_bytes), que deveria funcionar na maioria das versões novas
            try:
                k.from_bytes(file_bytes) 
            except AttributeError:
                # PLANO B: Descompactação Manual do KMZ
                st.info("Tentando descompactação manual do KMZ (Plano B) devido a erro de 'from_bytes'.")
                with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as kmz_file:
                    # O arquivo KML principal dentro do KMZ geralmente é doc.kml
                    kml_name_list = [name for name in kmz_file.namelist() if name.endswith('.kml')]
                    if not kml_name_list:
                         raise IndexError("Nenhum arquivo .kml encontrado dentro do KMZ.")
                    
                    kml_name = kml_name_list[0]
                    kml_content = kmz_file.read(kml_name)
                    # Usa k.from_string() que é universal
                    k.from_string(kml_content.decode('utf-8'))
        else:
            # Tenta o parsing de KML/XML como string UTF-8
            k.from_string(file_bytes.decode('utf-8')) 
            
    except IndexError as ie:
         st.error(f"Erro: O arquivo KMZ não contém um arquivo .kml principal. Detalhe: {ie}")
         return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro Crítico ao processar o arquivo. Verifique se ele é um KML/KMZ válido. Erro: {e}")
        return pd.DataFrame()
    
    data = []
    
    try:
        features_to_iterate = list(k.features())
    except Exception as e:
        st.error(f"Erro ao tentar acessar os elementos KML. O arquivo está corrompido ou o formato é inválido (Tipo de erro: {type(e).__name__}).")
        return pd.DataFrame()
    
    for feature in features_to_iterate:
        if isinstance(feature, kml.Document):
            for doc_feature in feature.features():
                if isinstance(doc_feature, kml.Folder):
                    for folder_feature in doc_feature.features():
                        if isinstance(folder_feature, kml.Placemark) and folder_feature.geometry:
                             coords = list(folder_feature.geometry.coords)[0]
                             data.append({
                                 'Endereco_KML': folder_feature.name,
                                 'Longitude_KML': coords[0],
                                 'Latitude_KML': coords[1]
                             })
                elif isinstance(doc_feature, kml.Placemark) and doc_feature.geometry:
                    coords = list(doc_feature.geometry.coords)[0]
                    data.append({
                        'Endereco_KML': doc_feature.name,
                        'Longitude_KML': coords[0],
                        'Latitude_KML': coords[1]
                    })
        elif isinstance(feature, kml.Placemark) and feature.geometry:
             coords = list(feature.geometry.coords)[0]
             data.append({
                 'Endereco_KML': feature.name,
                 'Longitude_KML': coords[0],
                 'Latitude_KML': coords[1]
             })

    if not data:
        st.warning("Nenhum 'Placemark' (parada) com coordenadas válidas foi encontrado no seu KML/KMZ/XML.")
        return pd.DataFrame()
        
    df_kml = pd.DataFrame(data)
    df_kml['Endereco_Completo_Cache'] = df_kml['Endereco_KML'].astype(str).str.strip().str.rstrip(';')
    df_kml['Latitude_Corrigida'] = pd.to_numeric(df_kml['Latitude_KML'], errors='coerce')
    df_kml['Longitude_Corrigida'] = pd.to_numeric(df_kml['Longitude_KML'], errors='coerce')

    return df_kml.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']]


def import_kml_to_db(conn, df_kml_import):
    """Insere os dados do KML/KMZ/XML no banco de dados de cache. (Simple Upsert - Sem conflito)"""
    
    if df_kml_import.empty:
        st.error("Nenhum dado válido para importar.")
        return 0
        
    insert_count = 0
    
    try:
        with st.spinner(f"Processando a importação de {len(df_kml_import)} paradas do KML/KMZ/XML..."):
            for index, row in df_kml_import.iterrows():
                endereco = row['Endereco_Completo_Cache']
                lat = row['Latitude_Corrigida']
                lon = row['Longitude_Corrigida']
                
                upsert_query = f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida, Origem_Correcao) 
                VALUES (?, ?, ?, ?);
                """
                conn.execute(upsert_query, (endereco, lat, lon, 'KML_Import'))
                insert_count += 1
            
            conn.commit()
            load_geoloc_cache.clear() 
            count_after = len(load_geoloc_cache(conn)) 
            st.success(f"✅ Importação de KML/KMZ/XML concluída! **{insert_count}** entradas processadas. O cache agora tem **{count_after}** entradas.")
            st.rerun() 
            return count_after
            
    except Exception as e:
        st.error(f"Erro crítico ao inserir dados do KML/KMZ/XML no cache. Erro: {e}")
        return 0

# ===============================================
# FUNÇÃO DE CONVERSÃO DE CSV GOOGLE MAPS (CORREÇÃO FORÇADA)
# ===============================================

@st.cache_data
def convert_google_maps_csv(uploaded_file):
    """
    Tenta ler o CSV original. Se falhar, aplica o reparo interno forçado
    para corrigir a quebra da coluna 'Destination Address' causada por vírgulas.
    """
    
    # [Mantém a lógica de reparo do CSV que funcionou para o usuário]
    # 1. Leitura do arquivo como texto para reparo
    uploaded_file.seek(0)
    try:
        content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('latin-1')
    except Exception as e:
        st.error(f"Erro Crítico de Leitura de Arquivo: {e}")
        return pd.DataFrame()
        
    lines = content.strip().splitlines()
    if not lines:
        st.error("Arquivo CSV vazio.")
        return pd.DataFrame()

    # 2. Identificação do cabeçalho e estrutura
    header = lines[0]
    data_lines = lines[1:]
    
    colunas_finais = [
        'WKT', 'AT ID', 'Sequence', 'Stop', 'SPX TN', 
        GMAPS_COL_ADDRESS, GMAPS_COL_BAIRRO, GMAPS_COL_CITY, GMAPS_COL_ZIPCODE, 
        GMAPS_COL_LAT, GMAPS_COL_LON
    ]

    reparsed_data = [header] 
    NUM_FIXED_PREFIX = 4 
    NUM_FIXED_SUFFIX = 5 

    for line in data_lines:
        if not line.strip(): continue

        match = re.match(r'(".*?")(,(.*))', line)
        if not match:
             reparsed_data.append(line) 
             continue
             
        wkt_col = match.group(1) 
        rest_of_line = match.group(3) 
        
        parts = [p.strip() for p in rest_of_line.split(',')]
        N_parts = len(parts)

        if N_parts == (len(colunas_finais) - 1):
             reparsed_data.append(line)
             continue
        
        # --- REPARO INTERNO FORÇADO ---
        try:
            suffix = parts[-NUM_FIXED_SUFFIX:] 
            prefix = parts[0:NUM_FIXED_PREFIX]
            middle_parts_raw = parts[NUM_FIXED_PREFIX:N_parts - NUM_FIXED_SUFFIX] 
            
            if len(prefix) == NUM_FIXED_PREFIX and len(suffix) == NUM_FIXED_SUFFIX and middle_parts_raw:
                
                destination_address_quoted = '"' + ', '.join(middle_parts_raw).strip() + '"'
                
                new_line = (
                    wkt_col + ',' + 
                    ','.join(prefix) + ',' + 
                    destination_address_quoted + ',' + 
                    ','.join(suffix)
                )
                reparsed_data.append(new_line)
            else:
                reparsed_data.append(line)

        except Exception as e:
            reparsed_data.append(line)

    # 5. Leitura da linha de dados reparada com Pandas
    try:
        df = pd.read_csv(io.StringIO('\n'.join(reparsed_data)), sep=',')
        
        if len(df.columns) != 11:
             st.error(f"O reparo resultou em um número incorreto de colunas: {len(df.columns)}. Colunas esperadas: 11.")
             return pd.DataFrame()
             
    except Exception as e:
        st.error(f"❌ Erro Crítico: Falha na leitura do CSV após o reparo interno. Erro: {e}")
        return pd.DataFrame()
    
    # ---------------------------------------------------------------------------------------------------------------------
    # CONCATENAÇÃO FINAL
    # ---------------------------------------------------------------------------------------------------------------------
    
    required_cols = [GMAPS_COL_ADDRESS, GMAPS_COL_BAIRRO, GMAPS_COL_CITY, GMAPS_COL_LAT, GMAPS_COL_LON]
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"O arquivo CSV do Google Maps está faltando colunas essenciais. Colunas faltando: {', '.join(missing)}")
        return pd.DataFrame()
    
    if GMAPS_COL_ZIPCODE not in df.columns:
         df[GMAPS_COL_ZIPCODE] = ""
         
    df = df.fillna('')
    
    endereco_principal = df[GMAPS_COL_ADDRESS].astype(str).str.strip().str.strip('"')
    
    df['Endereco_Completo_Cache'] = endereco_principal
    
    df['Endereco_Completo_Cache'] = df.apply(
        lambda row: f"{row['Endereco_Completo_Cache']}, {row[GMAPS_COL_BAIRRO].strip()}" if row[GMAPS_COL_BAIRRO].strip() else row['Endereco_Completo_Cache'],
        axis=1
    )
    
    df['Endereco_Completo_Cache'] = df.apply(
        lambda row: f"{row['Endereco_Completo_Cache']}, {row[GMAPS_COL_CITY].strip()}" if row[GMAPS_COL_CITY].strip() and row[GMAPS_COL_CITY].strip() not in row[GMAPS_COL_BAIRRO].strip() else row['Endereco_Completo_Cache'],
        axis=1
    )
    
    df['Endereco_Completo_Cache'] = df.apply(
        lambda row: f"{row['Endereco_Completo_Cache']}
