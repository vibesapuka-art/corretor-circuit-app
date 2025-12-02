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
        lambda row: f"{row['Endereco_Completo_Cache']}, {row[GMAPS_COL_ZIPCODE].strip()}" if row[GMAPS_COL_ZIPCODE].strip() else row['Endereco_Completo_Cache'],
        axis=1
    )
    
    df['Endereco_Completo_Cache'] = df['Endereco_Completo_Cache'].str.replace(r',\s*,', ',', regex=True)
    df['Endereco_Completo_Cache'] = df['Endereco_Completo_Cache'].str.replace(r'^\s*,', '', regex=True) 
    df['Endereco_Completo_Cache'] = df['Endereco_Completo_Cache'].str.replace(r',\s*$', '', regex=True) 
    df['Endereco_Completo_Cache'] = df['Endereco_Completo_Cache'].str.strip()

    # ---------------------------------------------------------------------------------------------------------------------

    df = df.rename(columns={
        GMAPS_COL_LAT: 'Latitude_Corrigida',
        GMAPS_COL_LON: 'Longitude_Corrigida'
    })
    
    df_final = df[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']].copy()
    
    # Garante que as coordenadas são numéricas para comparação
    df_final['Latitude_Corrigida'] = pd.to_numeric(df_final['Latitude_Corrigida'], errors='coerce')
    df_final['Longitude_Corrigida'] = pd.to_numeric(df_final['Longitude_Corrigida'], errors='coerce')
    
    df_final = df_final.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])

    if df_final.empty:
        st.error("Nenhuma linha com Lat/Lon válida foi encontrada após a conversão.")
        return pd.DataFrame()
        
    return df_final.drop_duplicates(subset=['Endereco_Completo_Cache'])

# ===============================================
# FUNÇÕES DE SINCRONIZAÇÃO DE CACHE (NOVO)
# ===============================================

def manage_cache_overwrite(df_new_corrections, cache_df):
    """
    Compara as novas correções do CSV com o cache existente para identificar
    novas entradas e conflitos de geolocalização.
    """
    
    # 1. Prepare new corrections
    df_new = df_new_corrections[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']].copy()
    df_new = df_new.rename(columns={'Latitude_Corrigida': 'New_Lat', 'Longitude_Corrigida': 'New_Lon'})
    df_new['Endereco_Completo_Cache'] = df_new['Endereco_Completo_Cache'].astype(str).str.strip().str.rstrip(';')
    
    # 2. Prepare existing cache for merge
    df_cache_for_merge = cache_df[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']].copy()
    df_cache_for_merge.columns = ['Endereco_Completo_Cache', 'Cache_Lat', 'Cache_Lon']
    
    # 3. Merge to find matches
    df_merged = pd.merge(
        df_new, 
        df_cache_for_merge, 
        on='Endereco_Completo_Cache', 
        how='left'
    )
    
    # 4. Identify NEW entries (no match in cache)
    df_new_entries = df_merged[df_merged['Cache_Lat'].isna()].copy()
    df_new_entries = df_new_entries.rename(columns={'New_Lat': 'Latitude_Corrigida', 'New_Lon': 'Longitude_Corrigida'})
    df_new_entries = df_new_entries[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']].copy()
    
    # 5. Identify OVERWRITES/CONFLICTS (match found)
    df_conflicts = df_merged[df_merged['Cache_Lat'].notna()].copy()
    
    # Calcula a diferença: 0.00001 grau é aproximadamente 1 metro de diferença, considerado um conflito.
    TOLERANCE = 0.00001
    
    df_conflicts['New_Lat'] = pd.to_numeric(df_conflicts['New_Lat'], errors='coerce')
    df_conflicts['New_Lon'] = pd.to_numeric(df_conflicts['New_Lon'], errors='coerce')
    df_conflicts['Cache_Lat'] = pd.to_numeric(df_conflicts['Cache_Lat'], errors='coerce')
    df_conflicts['Cache_Lon'] = pd.to_numeric(df_conflicts['Cache_Lon'], errors='coerce')
    df_conflicts.dropna(subset=['New_Lat', 'New_Lon', 'Cache_Lat', 'Cache_Lon'], inplace=True)

    lat_diff = abs(df_conflicts['New_Lat'] - df_conflicts['Cache_Lat'])
    lon_diff = abs(df_conflicts['New_Lon'] - df_conflicts['Cache_Lon'])
    
    # Mantém apenas as alterações significativas (conflitos)
    conflict_mask = (lat_diff > TOLERANCE) | (lon_diff > TOLERANCE)
    df_conflicts_to_overwrite = df_conflicts[conflict_mask].copy()
    
    # Adiciona a coluna com as coordenadas novas para visualização
    df_conflicts_to_overwrite.rename(columns={
        'New_Lat': 'Latitude_Nova', 
        'New_Lon': 'Longitude_Nova',
        'Cache_Lat': 'Latitude_Atual',
        'Cache_Lon': 'Longitude_Atual'
    }, inplace=True)
    
    # Seleciona colunas de interesse para o relatório de conflitos
    df_conflicts_report = df_conflicts_to_overwrite[['Endereco_Completo_Cache', 'Latitude_Atual', 'Longitude_Atual', 'Latitude_Nova', 'Longitude_Nova']]
    
    return df_new_entries, df_conflicts_report

def perform_upsert(conn, df_to_upsert, origem='Import_Sync'):
    """Executa o salvamento ou substituição (upsert) dos dados no banco de dados."""
    insert_count = 0
    df_to_upsert['Origem_Correcao'] = origem 
    df_to_upsert.rename(columns={'Latitude_Nova': 'Latitude_Corrigida', 'Longitude_Nova': 'Longitude_Corrigida'}, inplace=True)
    
    if 'Latitude_Corrigida' not in df_to_upsert.columns:
        # Se veio do df_new_entries, as colunas já estão como 'Latitude_Corrigida'
        pass

    try:
        with st.spinner(f"Processando a importação de {len(df_to_upsert)} linhas..."):
            for index, row in df_to_upsert.iterrows():
                endereco = row['Endereco_Completo_Cache']
                lat = row['Latitude_Corrigida']
                lon = row['Longitude_Corrigida']
                origem_correcao = row['Origem_Correcao']
                
                upsert_query = f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida, Origem_Correcao) 
                VALUES (?, ?, ?, ?);
                """
                conn.execute(upsert_query, (endereco, lat, lon, origem_correcao))
                insert_count += 1
            
            conn.commit()
            load_geoloc_cache.clear()
            st.success(f"Sincronização concluída! **{insert_count}** entradas atualizadas/inseridas.")
            return insert_count
    except Exception as e:
        st.error(f"Erro crítico ao sincronizar dados no cache. Erro: {e}")
        return 0


# ===============================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO (ATUALIZADA)
# ===============================================

def limpar_endereco(endereco):
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    endereco = re.sub(r'\s+', ' ', endereco)
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    return endereco

def get_most_common_or_empty(x):
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    return x_limpo.mode().iloc[0]

@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade, df_cache_geoloc):
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, COLUNA_BAIRRO, 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None, pd.DataFrame(), pd.DataFrame() # Retorna 3 valores
        
    df = df_entrada.copy()
    
    df[COLUNA_BAIRRO] = df[COLUNA_BAIRRO].astype(str).str.strip().replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    
    df['Chave_Busca_Cache'] = (
        df[COLUNA_ENDERECO].astype(str).str.strip() + 
        ', ' + 
        df[COLUNA_BAIRRO].astype(str).str.strip()
    )
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*$', '', regex=True)
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*,', ',', regex=True)

    
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)
    
    # Conversão das coordenadas originais para numérico para evitar erros na comparação/processamento
    df[COLUNA_LATITUDE] = pd.to_numeric(df[COLUNA_LATITUDE], errors='coerce')
    df[COLUNA_LONGITUDE] = pd.to_numeric(df[COLUNA_LONGITUDE], errors='coerce')
    
    # [NOVO]: Salva as coordenadas originais do CSV para o relatório de status
    df['Lat_Original_CSV'] = df[COLUNA_LATITUDE].copy()
    df['Lon_Original_CSV'] = df[COLUNA_LONGITUDE].copy()

    
    # PASSO 1: APLICAR LOOKUP NO CACHE DE GEOLOCALIZAÇÃO
    if not df_cache_geoloc.empty:
        df_cache_lookup = df_cache_geoloc.rename(columns={
            'Endereco_Completo_Cache': 'Chave_Cache_DB', 
            'Latitude_Corrigida': 'Cache_Lat',
            'Longitude_Corrigida': 'Cache_Lon'
        })
        
        df = pd.merge(
            df, 
            df_cache_lookup, 
            left_on='Chave_Busca_Cache', 
            right_on='Chave_Cache_DB',   
            how='left'
        )
        
        # Máscara de sucesso (match no cache)
        cache_mask = df['Cache_Lat'].notna()
        
        # Aplica a correção do cache (Overwrite)
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Cache_Lat']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Cache_Lon']
        
        # 1. IDENTIFY CORRECTED (Para o relatório)
        df_corrected = df[cache_mask].copy()
        df_corrected_unique = df_corrected.drop_duplicates(subset=['Chave_Cache_DB'])
        
        # Colunas para o relatório de corrigidos
        df_corrected_unique = df_corrected_unique[['Chave_Cache_DB', 'Cache_Lat', 'Cache_Lon', 'Lat_Original_CSV', 'Lon_Original_CSV']]
        df_corrected_unique.columns = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Latitude_Original', 'Longitude_Original']
        
        # 2. IDENTIFY UNCORRECTED (NEW) (Para o relatório)
        df_uncorrected = df[~cache_mask].copy()
        df_uncorrected_unique = df_uncorrected.drop_duplicates(subset=['Chave_Busca_Cache'])
        
        # Colunas para o relatório de não corrigidos (usa a coord. original do CSV)
        df_uncorrected_unique = df_uncorrected_unique[['Chave_Busca_Cache', 'Lat_Original_CSV', 'Lon_Original_CSV']]
        df_uncorrected_unique.columns = ['Endereco_Completo_Cache', 'Latitude_Original', 'Longitude_Original']
        
        # Limpa as colunas temporárias do DF principal
        df = df.drop(columns=['Chave_Busca_Cache', 'Chave_Cache_DB', 'Cache_Lat', 'Cache_Lon', 'Lat_Original_CSV', 'Lon_Original_CSV'], errors='ignore')
        
    else:
        # Se o cache está vazio, todos são não corrigidos
        df_corrected_unique = pd.DataFrame(columns=['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida', 'Latitude_Original', 'Longitude_Original'])
        df_uncorrected_unique = df.drop_duplicates(subset=['Chave_Busca_Cache'])[['Chave_Busca_Cache', COLUNA_LATITUDE, COLUNA_LONGITUDE]]
        df_uncorrected_unique.columns = ['Endereco_Completo_Cache', 'Latitude_Original', 'Longitude_Original']
        df = df.drop(columns=['Chave_Busca_Cache', 'Lat_Original_CSV', 'Lon_Original_CSV'], errors='ignore')


    # PASSO 2: FUZZY MATCHING (CORREÇÃO DE ENDEREÇO E AGRUPAMENTO)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching e Agrupamento...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos == 0:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")
        return None, pd.DataFrame(), pd.DataFrame()
    
    for i, end_principal in enumerate(enderecos_unicos):
        if end_principal not in mapa_correcao:
            matches = process.extract(
                end_principal, 
                enderecos_unicos, 
                scorer=fuzz.WRatio, 
                limit=None
            )
            grupo_matches = [
                match[0] for match in matches 
                if match[1] >= limite_similaridade
            ]
            
            df_grupo = df[df['Endereco_Limpo'].isin(grupo_matches)]
            endereco_oficial_original = get_most_common_or_empty(df_grupo[COLUNA_ENDERECO])
            if not endereco_oficial_original:
                 endereco_oficial_original = end_principal 
            
            for end_similar in grupo_matches:
                mapa_correcao[end_similar] = endereco_oficial_original
                
        progresso_bar.progress((i + 1) / total_unicos, text=f"Processando {i+1} de {total_unicos} endereços únicos...")
    
    progresso_bar.empty()
    st.success("Fuzzy Matching concluído!")

    # Aplicação do Endereço Corrigido (Chave de Agrupamento)
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # Agrupamento (Chave: Endereço Corrigido + Cidade + BAIRRO)
    colunas_agrupamento = ['Endereco_Corrigido', 'City', COLUNA_BAIRRO] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'\*', '', str(y))) if re.sub(r'\*', '', str(y)).isdigit() else float('inf'))))), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        Bairro_Agrupado=(COLUNA_BAIRRO, get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        Min_Sequence=('Sequence_Num', 'min') 
    ).reset_index()

    # Ordenação
    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    # Formatação do DF para o CIRCUIT 
    endereco_completo_circuit = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro_Agrupado'].str.strip() 
    )
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*,', ',', regex=True)
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*$', '', regex=True) 
    
    # A coluna de Notes deve conter o Order ID e outras infos
    notas_completas = (
        df_agrupado['Sequences_Agrupadas'] + '; ' +
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode_Agrupado']
    )
    
    # Colunas essenciais para importação (com coordenadas)
    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    }) 
    
    # Adicionando uma coluna 'Sequence_Base' para manter a ordem de importação, se for usado o split
    df_circuit.insert(0, 'Sequence_Base', range(1, len(df_circuit) + 1))
    
    # Retorna o DF principal e os DFs de status de correção
    return df_circuit, df_corrected_unique, df_uncorrected_unique 


def split_dataframe_for_drivers(df_circuit, num_motoristas):
    # [Mantido inalterado]
    if df_circuit is None or df_circuit.empty:
        return {}
    
    COLUNAS_EXPORT_SPLIT = ['Address', 'Latitude', 'Longitude', 'Notes']
    df_export = df_circuit[['Sequence_Base'] + COLUNAS_EXPORT_SPLIT].copy()
    
    df_export.rename(columns={'Notes': 'Notes', 'Address': 'Address'}, inplace=True)
    
    total_paradas = len(df_export)
    
    if num_motoristas <= 0:
        return {} 
    
    paradas_base = total_paradas // num_motoristas
    restante = total_paradas % num_motoristas
    rotas_divididas = {}
    start_index = 0
    
    for i in range(num_motoristas):
        paradas_motorista = paradas_base + (1 if i < restante else 0)
        
        end_index = start_index + paradas_motorista
        
        df_motorista = df_export.iloc[start_index:end_index].copy()
        
        df_motorista.insert(1, 'Order ID', df_motorista['Notes'].apply(lambda x: str(x).split(';')[0].strip()))
        
        df_motorista = df_motorista.drop(columns=['Sequence_Base'])
        
        df_motorista = df_motorista[['Order ID', 'Address', 'Latitude', 'Longitude', 'Notes']]
        
        rotas_divididas[f"Motorista {i+1} ({len(df_motorista)} Paradas)"] = df_motorista
        
        start_index = end_index
        
    return rotas_divididas


def is_not_purely_volumous(ids_string):
    # [Mantido inalterado]
    if pd.isna(ids_string) or not ids_string:
        return False
        
    ids = [
        i.strip() 
        for i in str(ids_string).replace(' ', '').split(',') 
        if i.strip()
    ]
    
    if not ids:
        return False 

    for id_pacote in ids:
        if not id_pacote.endswith('*'):
            return True 
    
    return False 


def processar_rota_para_impressao(df_input):
    # [Mantido inalterado]
    
    df_input.columns = df_input.columns.str.strip().str.lower()
    
    if COLUNA_NOTES_CIRCUIT not in df_input.columns and 'order id' not in df_input.columns:
        raise KeyError(f"As colunas de endereço ('{COLUNA_ADDRESS_CIRCUIT}') e notas/id ('{COLUNA_NOTES_CIRCUIT}' ou 'order id') não foram encontradas.") 
        
    df = df_input.copy()
    
    if COLUNA_NOTES_CIRCUIT not in df.columns and 'order id' in df.columns:
        df[COLUNA_NOTES_CIRCUIT] = df['order id'].astype(str)
        
    df[COLUNA_NOTES_CIRCUIT] = df[COLUNA_NOTES_CIRCUIT].astype(str).str.strip('"')
    df = df.dropna(subset=[COLUNA_NOTES_CIRCUIT]) 
    
    df_split = df[COLUNA_NOTES_CIRCUIT].str.split(';', n=1, expand=True)
    df['Ordem ID'] = df_split[0].str.strip() 
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    df['ID_Pacote_Limpo'] = df['Ordem ID'].str.strip() 
    
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    df['Address_Clean'] = df[COLUNA_ADDRESS_CIRCUIT].astype(str)
    
    coluna_filtro = 'ID_Pacote_Limpo' 
    
    df_final_geral = df[['Lista de Impressão', 'Address_Clean']].copy() 
    
    df_volumosos = df[df[coluna_filtro].str.contains(r'\*', regex=True, na=False)].copy()
    df_volumosos_impressao = df_volumosos[['Lista de Impressão', 'Address_Clean']].copy() 
    
    df_nao_volumosos = df[
        df[coluna_filtro].apply(is_not_purely_volumous)
    ].copy() 
    
    df_nao_volumosos_impressao = df_nao_volumosos[['Lista de Impressão', 'Address_Clean']].copy()
    
    df_limpo_para_split_pos = df[[COLUNA_ADDRESS_CIRCUIT, COLUNA_NOTES_CIRCUIT]].copy()
    df_limpo_para_split_pos.columns = ['Address', 'Notes'] 
    
    return df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao, df_limpo_para_split_pos


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

# 1. Conexão com o Banco de Dados (Executada uma vez)
conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("🗺️ Flow Completo Circuit (Pré, Pós e Cache)")

# CRIAÇÃO DAS ABAS
tab1, tab_split, tab2, tab3, tab_geodata_import = st.tabs([
    "🚀 Pré-Roteirização (Importação)", 
    "✂️ Split Route (Dividir)", 
    "📋 Pós-Roteirização (Impressão/Cópia)", 
    "💾 Gerenciar Cache de Geolocalização", 
    "🌎 Importar Pontos de Correção (GeoData)"
])


# ----------------------------------------------------------------------------------
# VARIÁVEIS DE ESTADO (SESSION STATE)
# ----------------------------------------------------------------------------------

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'volumoso_ids' not in st.session_state:
    st.session_state['volumoso_ids'] = set() 
if 'df_circuit_agrupado_pre' not in st.session_state: 
    st.session_state['df_circuit_agrupado_pre'] = None
if 'df_kml_extraido' not in st.session_state:
    st.session_state['df_kml_extraido'] = pd.DataFrame()
if 'df_csv_convertido' not in st.session_state: 
    st.session_state['df_csv_convertido'] = pd.DataFrame()
if 'df_corrected_unique_pre' not in st.session_state: # NOVO
    st.session_state['df_corrected_unique_pre'] = pd.DataFrame()
if 'df_uncorrected_unique_pre' not in st.session_state: # NOVO
    st.session_state['df_uncorrected_unique_pre'] = pd.DataFrame()
if 'df_new_entries' not in st.session_state: # NOVO (Tab 3)
    st.session_state['df_new_entries'] = pd.DataFrame()
if 'df_conflicts' not in st.session_state: # NOVO (Tab 3)
    st.session_state['df_conflicts'] = pd.DataFrame()
if 'overwrite_confirmed' not in st.session_state: # NOVO (Tab 3)
    st.session_state['overwrite_confirmed'] = False


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa aplica as correções de **Geolocalização do Cache (100% Match)** e agrupa os endereços.")

    st.markdown("---")
    st.subheader("1.1 Carregar Planilha Original")

    uploaded_file_pre = st.file_uploader(
        "Arraste e solte o arquivo original (CSV/Excel) aqui:", 
        type=['csv', 'xlsx'],
        key="file_pre"
    )

    if uploaded_file_pre is not None:
        try:
            if uploaded_file_pre.name.endswith('.csv'):
                df_input_pre = pd.read_csv(uploaded_file_pre)
            else:
                df_input_pre = pd.read_excel(uploaded_file_pre, sheet_name=0)
            
            colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, COLUNA_BAIRRO, 'City', 'Zipcode/Postal code']
            for col in colunas_essenciais:
                 if col not in df_input_pre.columns:
                     raise KeyError(f"A coluna '{col}' está faltando na sua planilha.")
            
            if st.session_state.get('last_uploaded_name') != uploaded_file_pre.name:
                 st.session_state['volumoso_ids'] = set()
                 st.session_state['last_uploaded_name'] = uploaded_file_pre.name
                 st.session_state['df_circuit_agrupado_pre'] = None
                 # Limpa os relatórios de status anteriores
                 st.session_state['df_corrected_unique_pre'] = pd.DataFrame()
                 st.session_state['df_uncorrected_unique_pre'] = pd.DataFrame()


            st.session_state['df_original'] = df_input_pre.copy()
            st.success(f"Arquivo '{uploaded_file_pre.name}' carregado! Total de **{len(df_input_pre)}** registros.")
            
        except KeyError as ke:
             st.error(f"Erro de Coluna: {ke}")
             st.session_state['df_original'] = None
             st.session_state['df_circuit_agrupado_pre'] = None
        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar o arquivo. Verifique o formato. Erro: {e}")

    
    st.markdown("---")
    st.subheader("1.2 Marcar Pacotes Volumosos (Volumosos = *)")
    
    if st.session_state['df_original'] is not None:
        
        df_temp = st.session_state['df_original'].copy()
        
        df_temp['Order_Num'] = df_temp[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
        df_temp['Order_Num'] = pd.to_numeric(df_temp['Order_Num'], errors='coerce')
        
        df_ordens_unicas = df_temp.drop_duplicates(subset=[COLUNA_SEQUENCE]).sort_values(by='Order_Num')
        ordens_originais_sorted = df_ordens_unicas[COLUNA_SEQUENCE].astype(str).tolist()
        
        def update_volumoso_ids(order_id, is_checked):
            if is_checked:
                st.session_state['volumoso_ids'].add(order_id)
            elif order_id in st.session_state['volumoso_ids']:
                st.session_state['volumoso_ids'].remove(order_id)

        st.caption("Marque os números das ordens de serviço que são volumosas (serão marcadas com *):")
        st.info("A lista abaixo está ordenada corretamente pela Sequence (1, 2, 3, ...)")

        NUM_COLS = 5
        total_items = len(ordens_originais_sorted)
        chunked_list = [
            ordens_originais_sorted[i:i + NUM_COLS] 
            for i in range(0, total_items, NUM_COLS)
        ]

        with st.container(height=300):
            for row_chunk in chunked_list:
                cols = st.columns(len(row_chunk)) 
                for col_index, order_id in enumerate(row_chunk):
                    with cols[col_index]: 
                        is_checked = order_id in st.session_state['volumoso_ids']
                        st.checkbox(
                            str(order_id), 
                            value=is_checked, 
                            key=f"vol_{order_id}",
                            on_change=update_volumoso_ids, 
                            args=(order_id, not is_checked) 
                        )

        st.info(f"**{len(st.session_state['volumoso_ids'])}** pacotes marcados como volumosos.")
        
        st.markdown("---")
        st.subheader("1.3 Configurar e Processar")
        
        limite_similaridade_ajustado = st.slider(
            'Ajuste a Precisão do Corretor (Fuzzy Matching):',
            min_value=80,
            max_value=100,
            value=100, 
            step=1,
            help="Use 100% para garantir que endereços na mesma rua com números diferentes não sejam agrupados (recomendado)."
        )
        st.info(f"O limite de similaridade está em **{limite_similaridade_ajustado}%**.")
        
        
        if st.button("🚀 Iniciar Corretor e Agrupamento", key="btn_pre_final"):
            
            df_para_processar = st.session_state['df_original'].copy()
            df_para_processar[COLUNA_SEQUENCE] = df_para_processar[COLUNA_SEQUENCE].astype(str)
            
            for id_volumoso in st.session_state['volumoso_ids']:
                str_id_volumoso = str(id_volumoso)
                df_para_processar.loc[
                    df_para_processar[COLUNA_SEQUENCE] == str_id_volumoso, 
                    COLUNA_SEQUENCE
                ] = str_id_volumoso + '*'

            df_cache = load_geoloc_cache(conn)

            result = None 
            df_corrected_unique = pd.DataFrame()
            df_uncorrected_unique = pd.DataFrame()
            
            with st.spinner('Aplicando cache 100% match e processando dados...'):
                 try:
                     # CHAMA A FUNÇÃO ATUALIZADA (retorna 3 DFs)
                     result = processar_e_corrigir_dados(df_para_processar, limite_similaridade_ajustado, df_cache)
                 except Exception as e:
                     st.error(f"Erro Crítico durante a correção e agrupamento: {e}")
                     result = None 
                 
                 if isinstance(result, (list, tuple)) and len(result) == 3:
                     df_circuit, df_corrected_unique, df_uncorrected_unique = result
                 else:
                     df_circuit = None
                     
            
            if df_circuit is not None:
                st.session_state['df_circuit_agrupado_pre'] = df_circuit
                st.session_state['df_corrected_unique_pre'] = df_corrected_unique
                st.session_state['df_uncorrected_unique_pre'] = df_uncorrected_unique
                
                st.markdown("---")
                st.header("✅ Resultado Concluído!")
                
                # NOVO: RELATÓRIO DE STATUS DE CORREÇÃO
                st.subheader("📊 Status da Correção de Geolocalização")
                
                # 1. Corrigidos pelo Cache
                if not df_corrected_unique.empty:
                    st.success(f"✅ **{len(df_corrected_unique)}** Endereços Corrigidos pelo Cache (100% Match).")
                    with st.expander("Clique para ver os endereços corrigidos e as coordenadas utilizadas"):
                        st.dataframe(df_corrected_unique, use_container_width=True)
                else:
                    st.info("Nenhuma correção de geolocalização foi aplicada pelo cache nesta planilha (100% Match).")
                
                # 2. Não Corrigidos (Novos/Problemas)
                if not df_uncorrected_unique.empty:
                    st.warning(f"⚠️ **{len(df_uncorrected_unique)}** Endereços NOVOS ou NÃO Corrigidos (Coordenadas Originais do CSV usadas).")
                    with st.expander("Clique para ver os endereços NOVOS que precisam de correção manual"):
                        st.dataframe(df_uncorrected_unique, use_container_width=True)
                        
                        # Área de texto para cópia rápida
                        addresses_to_copy = '\n'.join(df_uncorrected_unique['Endereco_Completo_Cache'].astype(str).tolist())
                        st.text_area(
                            "Copie os endereços para corrigir no cache (Aba 💾 Gerenciar Cache):",
                            addresses_to_copy,
                            height=150,
                            key="copy_uncorrected_list"
                        )
                
                st.markdown("---")
                # FIM DO NOVO RELATÓRIO

                total_entradas = len(st.session_state['df_original'])
                total_agrupados = len(df_circuit)
                
                st.metric(
                    label="Endereços Únicos Agrupados",
                    value=total_agrupados,
                    delta=f"-{total_entradas - total_agrupados} agrupados"
                )
                
                df_volumosos_separado = df_circuit[
                    df_circuit['Order ID'].astype(str).str.contains(r'\*', regex=True)
                ].copy()
                
                st.subheader("Arquivo para Roteirização (Circuit)")
                st.dataframe(df_circuit.drop(columns=['Sequence_Base']), use_container_width=True) 
                
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                    df_circuit.drop(columns=['Sequence_Base']).to_excel(writer, index=False, sheet_name='Circuit_Import_Geral')
                    if not df_volumosos_separado.empty:
                        df_volumosos_separado.drop(columns=['Sequence_Base']).to_excel(writer, index=False, sheet_name='APENAS_VOLUMOSOS')
                        st.info(f"O arquivo de download conterá uma aba extra com **{len(df_volumosos_separado)}** endereços que incluem pacotes volumosos.")
                    else:
                        st.info("Nenhum pacote volumoso marcado.")
                        
                buffer_circuit.seek(0)
                
                st.download_button(
                    label="📥 Baixar ARQUIVO GERAL PARA CIRCUIT",
                    data=buffer_circuit,
                    file_name="Circuit_Import_FINAL_GERAL.xlsx",
                    mime=EXCEL_MIME_TYPE, 
                    key="download_excel_circuit"
                )
                
                st.markdown("---")
                st.info("Agora, você pode usar o arquivo na aba **✂️ Split Route** ou este arquivo geral no Circuit.")


# ----------------------------------------------------------------------------------
# ABA 1.5: SPLIT ROUTE (DIVIDIR ROTAS)
# [Mantido inalterado]
# ----------------------------------------------------------------------------------
with tab_split:
    st.header("✂️ Dividir Rota PRÉ-Roteirização (Downloads Individuais)")
    st.caption("A divisão é feita no arquivo agrupado. Baixe um arquivo **individual** para cada motorista.")
    
    st.markdown("---")
    
    df_rota_para_split = st.session_state.get('df_circuit_agrupado_pre')
    
    if df_rota_para_split is not None and not df_rota_para_split.empty:
        
        st.info(f"Rota agrupada carregada da Pré-Roteirização: **{len(df_rota_para_split)} paradas** únicas.")
        
        st.subheader("1. Configurar Divisão")
        
        num_motoristas = st.slider(
            'Número de Motoristas para Divisão:',
            min_value=2,
            max_value=10, 
            value=2,
            step=1,
            key="num_motoristas_split_pre"
        )
        
        if st.button("➡️ Dividir e Gerar Botões de Download Individual", key="btn_split_route_pre"):
            
            rotas_divididas = split_dataframe_for_drivers(df_rota_para_split, num_motoristas)
            
            st.markdown("---")
            st.header("✅ Lista e Downloads Individuais")
            st.success("O arquivo agrupado foi dividido. Visualize a lista de paradas e baixe o arquivo exclusivo de cada motorista.")
            
            for i, (nome_rota, df_rota) in rotas_divididas.items():
                
                st.markdown("___")
                st.subheader(f"Lista para {nome_rota}")
                
                st.dataframe(df_rota, use_container_width=True)
                
                buffer_individual = io.BytesIO()
                with pd.ExcelWriter(buffer_individual, engine='openypxl') as writer:
                    df_rota.to_excel(writer, index=False, sheet_name='Rota_Motorista')
                    
                buffer_individual.seek(0)
                
                file_name = f"Circuit_Rota_{i+1}_{len(df_rota)}_Paradas.xlsx"
                
                st.download_button(
                    label=f"⬇️ Baixar Arquivo de Importação para {nome_rota}",
                    data=buffer_individual,
                    file_name=file_name,
                    mime=EXCEL_MIME_TYPE, 
                    key=f"download_split_{i+1}"
                )
            
            st.markdown("---")
            st.info("Cada arquivo baixado contém a lista de paradas na ordem sequencial, com coordenadas, para ser otimizada individualmente no Circuit/Spoke.")


# ----------------------------------------------------------------------------------
# ABA 2: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO E SEPARAÇÃO DE VOLUMOSOS)
# [Mantido inalterado]
# ----------------------------------------------------------------------------------

with tab2:
    st.header("3. Limpar Saída do Circuit para Impressão")
    st.warning("⚠️ Atenção: Use o arquivo CSV/Excel que foi gerado *após a conversão* do PDF da rota do Circuit.")

    st.markdown("---")
    st.subheader("3.1 Carregar Arquivo da Rota Otimizada")

    uploaded_file_pos = st.file_uploader(
        "Arraste e solte o arquivo da rota do Circuit aqui (CSV/Excel):", 
        type=['csv', 'xlsx'],
        key="file_pos"
    )

    sheet_name_default = "Table 3" 
    sheet_name = sheet_name_default
    
    df_final_geral = None 
    df_volumosos_impressao = None 
    df_nao_volumosos_impressao = None

    copia_data_geral = "Nenhum arquivo carregado ou nenhum dado válido encontrado após o processamento."
    copia_data_volumosos = "Nenhum pacote volumoso encontrado na rota."
    copia_data_nao_volumosos = "Nenhum pacote não-volumoso encontrado na rota."

    if uploaded_file_pos is not None and uploaded_file_pos.name.endswith('.xlsx'):
        sheet_name = st.text_input(
            "Seu arquivo é um Excel (.xlsx). Digite o nome da aba com os dados da rota (ex: Table 3):", 
            value=sheet_name_default,
            key="sheet_name_input_pos"
        )

    if uploaded_file_pos is not None:
        try:
            if uploaded_file_pos.name.endswith('.csv'):
                df_input_pos = pd.read_csv(uploaded_file_pos)
            else:
                df_input_pos = pd.read_excel(uploaded_file_pos, sheet_name=sheet_name)
            
            results = processar_rota_para_impressao(df_input_pos)
            
            df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao, _ = results
            
            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_input_pos)}** registros na sequência otimizada.")
            
            if df_final_geral is not None and not df_final_geral.empty:
                st.markdown("---")
                st.subheader("3.2 Resultado Final (Lista de Impressão GERAL)")
                
                df_visualizacao_geral = df_final_geral.copy()
                df_visualizacao_geral.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                st.dataframe(df_visualizacao_geral, use_container_width=True)

                copia_data_geral = '\n'.join(df_final_geral['Lista de Impressão'].astype(str).tolist())
                
                
                st.markdown("---")
                st.header("✅ Lista de Impressão APENAS NÃO-VOLUMOSOS")
                
                if not df_nao_volumosos_impressao.empty:
                    st.success(f"Foram encontrados **{len(df_nao_volumosos_impressao)}** endereços com pacotes NÃO-volumosos.")
                    df_visualizacao_nao_vol = df_nao_volumosos_impressao.copy()
                    df_visualizacao_nao_vol.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                    st.dataframe(df_visualizacao_nao_vol, use_container_width=True)
                    copia_data_nao_volumosos = '\n'.join(df_nao_volumosos_impressao['Lista de Impressão'].astype(str).tolist())
                else:
                    st.info("Todos os pedidos nesta rota estão marcados como volumosos (ou a lista está vazia).")
                    
                st.markdown("---")
                st.header("📦 Lista de Impressão APENAS VOLUMOSOS")
                
                if not df_volumosos_impressao.empty:
                    st.warning(f"Foram encontrados **{len(df_volumosos_impressao)}** endereços com pacotes volumosos.")
                    df_visualizacao_vol = df_volumosos_impressao.copy()
                    df_visualizacao_vol.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                    st.dataframe(df_visualizacao_vol, use_container_width=True)
                    copia_data_volumosos = '\n'.join(df_volumosos_impressao['Lista de Impressão'].astype(str).tolist())
                else:
                    st.info("Nenhum pedido volumoso detectado nesta rota.")


            else:
                 copia_data_geral = "O arquivo foi carregado, mas a coluna 'Notes' estava vazia ou o processamento não gerou resultados. Verifique o arquivo de rota do Circuit."


        except KeyError as ke:
            if "Table 3" in str(ke) or "Sheet" in str(ke):
                st.error(f"Erro de Aba: A aba **'{sheet_name}'** não foi encontrada no arquivo Excel.")
            elif 'address' in str(ke) or 'notes' in str(ke):
                 st.error(f"Erro de Coluna: O arquivo deve ter as colunas 'address' e 'notes' (ou 'order id'). Verifique o arquivo de rota.")
            else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato. Erro: {e}")
            
    
    # Renderização das áreas de cópia e download
    if uploaded_file_pos is not None:
        
        # --- ÁREA DE CÓPIA GERAL ---
        st.markdown("### 3.3 Copiar para a Área de Transferência (Lista GERAL)")
        st.info("Para copiar: **Selecione todo o texto** abaixo (Ctrl+A / Cmd+A) e pressione **Ctrl+C / Cmd+C**.")
        
        st.text_area(
            'Conteúdo da Lista de Impressão GERAL (Alinhado à Esquerda):', 
            copia_data_geral, 
            height=300,
            key="text_area_geral"
        )

        # --- ÁREA DE CÓPIA NÃO-VOLUMOSOS ---
        if not df_nao_volumosos_impressao.empty if df_nao_volumosos_impressao is not None else False:
            st.markdown("### 3.4 Copiar para a Área de Transferência (APENAS NÃO-Volumosos)")
            st.text_area(
                'Conteúdo da Lista de Impressão NÃO-VOLUMOSOS (Alinhado à Esquerda):', 
                copia_data_nao_volumosos, 
                height=150,
                key="text_area_nao_volumosos"
            )
        
        # --- ÁREA DE CÓPIA VOLUMOSOS ---
        if not df_volumosos_impressao.empty if df_volumosos_impressao is not None else False:
            st.markdown("### 3.5 Copiar para a Área de Transferência (APENAS Volumosos)")
            st.text_area(
                'Conteúdo da Lista de Impressão VOLUMOSOS (Alinhado à Esquerda):', 
                copia_data_volumosos, 
                height=150,
                key="text_area_volumosos"
            )
        
        
        # --- BOTÕES DE DOWNLOAD ---
        if df_final_geral is not None and not df_final_geral.empty:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer: 
                df_final_geral[['Lista de Impressão']].to_excel(writer, index=False, sheet_name='Lista Impressao Geral')
                
                if df_nao_volumosos_impressao is not None and not df_nao_volumosos_impressao.empty:
                    df_nao_volumosos_impressao[['Lista de Impressão']].to_excel(writer, index=False, sheet_name='Lista Nao Volumosos')
                    
                if df_volumosos_impressao is not None and not df_volumosos_impressao.empty:
                    df_volumosos_impressao[['Lista de Impressão']].to_excel(writer, index=False, sheet_name='Lista Volumosos')
                    
            buffer.seek(0)
            
            st.download_button(
                label="📥 Baixar Lista Limpa (Excel) - Geral + Separadas",
                data=buffer,
                file_name="Lista_Ordem_Impressao_FINAL.xlsx",
                mime=EXCEL_MIME_TYPE, 
                help="Baixe este arquivo. Ele contém três abas: a lista geral, a lista de não-volumosos e a lista de volumosos.",
                key="download_list"
            )


# ----------------------------------------------------------------------------------
# ABA 3: GERENCIAR CACHE DE GEOLOCALIZAÇÃO
# [Mantido inalterado - Apenas UI/Visualização]
# ----------------------------------------------------------------------------------
with tab3:
    st.header("💾 Gerenciamento Direto do Cache de Geolocalização")
    st.info("A chave de busca no pré-roteirização é a combinação exata de **Endereço + Bairro** da sua planilha original.")

    df_cache_original = load_geoloc_cache(conn).fillna("")
    
    
    # --- Formulário de Entrada Rápida ---
    st.subheader("4.1 Adicionar Nova Correção Rápida")
    
    with st.container():
        
        st.subheader("1. Preencher Endereço")
        if 'form_new_endereco' not in st.session_state:
            st.session_state['form_new_endereco'] = ""
            
        new_endereco = st.text_area(
            "1. Endereço COMPLETO no Cache (Copie e Cole do Circuit)", 
            key="form_new_endereco", 
            height=70,
            help="Cole o endereço exatamente como o Circuit o reconhece (incluindo o Bairro/Cidade). O sistema remove automaticamente o ' ; ' final, se houver."
        )
        
        st.markdown("---")
        st.subheader("2. Preencher Coordenadas (Use o método mais fácil)")
        
        col_input_coord, col_btn_coord = st.columns([3, 1])
        
        with col_input_coord:
            if 'form_colar_coord' not in st.session_state:
                st.session_state['form_colar_coord'] = ""
                
            st.text_input(
                "2. Colar Coordenadas Google (Ex: -23.5139753, -52.1131268)",
                key="form_colar_coord",
                help="Cole o texto de Lat e Lon copiados do Google Maps/Earth. O sistema tentará limpar a vírgula para decimal, mas ponto é preferencial."
            )
        with col_btn_coord:
            st.markdown("##") 
            st.button(
                "Aplicar Coordenadas", 
                on_click=apply_google_coords,
                key="btn_apply_coord",
            )
        
        st.caption("--- OU preencha ou ajuste manualmente (deve usar PONTO como separador decimal) ---")

        col_lat, col_lon = st.columns(2)
        
        if 'form_new_lat_num' not in st.session_state:
            st.session_state['form_new_lat_num'] = 0.0
        if 'form_new_lon_num' not in st.session_state:
            st.session_state['form_new_lon_num'] = 0.0
            
        with col_lat:
            new_latitude = st.number_input(
                "3. Latitude Corrigida", 
                value=st.session_state['form_new_lat_num'], 
                format="%.8f", 
                step=0.00000001,
                key="form_new_lat_num" 
            )
        with col_lon:
            new_longitude = st.number_input(
                "4. Longitude Corrigida", 
                value=st.session_state['form_new_lon_num'], 
                format="%.8f", 
                step=0.00000001,
                key="form_new_lon_num"
            )
            
        st.markdown("---")
        
        save_button_col, clear_button_col = st.columns(2)
        
        with save_button_col:
            if st.button("✅ Salvar Nova Correção no Cache", key="btn_save_quick"):
                
                lat_to_save = st.session_state.get('form_new_lat_num') 
                lon_to_save = st.session_state.get('form_new_lon_num')
                
                if not new_endereco or (abs(lat_to_save) == 0.0 and abs(lon_to_save) == 0.0 and st.session_state.get('form_colar_coord') == ""):
                    st.error("Preencha o endereço e as coordenadas (3 e 4) antes de salvar, ou use a ferramenta 'Aplicar Coordenadas'.")
                else:
                    try:
                        endereco_limpo = new_endereco.strip().rstrip(';')
                        save_single_entry_to_db(conn, endereco_limpo, lat_to_save, lon_to_save, origem='Manual')
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}. Verifique o formato do endereço.")
        
        with clear_button_col:
             st.button("❌ Limpar Formulário", on_click=clear_lat_lon_fields, key="btn_clear_form")


    
    st.markdown("---")
    
    st.subheader(f"4.2 Visualização do Cache Salvo (Total: {len(df_cache_original)})")
    st.caption("Esta tabela mostra os dados atualmente salvos. Use o formulário acima para adicionar ou substituir entradas.")
    
    st.dataframe(df_cache_original, use_container_width=True) 
    
    st.markdown("---")
    
    
    # --- BACKUP E RESTAURAÇÃO ---
    st.header("4.3 Backup e Restauração do Cache")
    st.caption("Gerencie o cache de geolocalização para migração ou segurança dos dados.")
    
    col_backup, col_restauracao = st.columns(2)
    
    with col_backup:
        st.markdown("#### 📥 Fazer Backup (Download)")
        st.info(f"Baixe o cache atual (**{len(df_cache_original)} entradas**).")
        
        if not df_cache_original.empty:
            
            # --- DOWNLOAD XLSX (PADRÃO) ---
            backup_xlsx, mime_xlsx, filename_xlsx = export_cache(df_cache_original, 'xlsx')
            st.download_button(
                label="⬇️ Baixar Backup do Cache (.xlsx)",
                data=backup_xlsx,
                file_name=filename_xlsx,
                mime=mime_xlsx, 
                key="download_backup_xlsx"
            )
            
            # --- DOWNLOAD CSV (NOVA OPÇÃO) ---
            backup_csv, mime_csv, filename_csv = export_cache(df_cache_original, 'csv')
            st.download_button(
                label="⬇️ Baixar Backup do Cache (.csv, Separador `,`)",
                data=backup_csv,
                file_name=filename_csv,
                mime=mime_csv, 
                key="download_backup_csv",
                help="Este arquivo CSV usa vírgula (,) como separador, garantindo que as colunas fiquem separadas corretamente para importação ou visualização em planilhas."
            )
            
        else:
            st.warning("O cache está vazio, não há dados para baixar.")


    with col_restauracao:
        st.markdown("#### 📤 Restaurar Cache (Upload)")
        st.warning("A restauração irá **substituir** entradas existentes (Endereço Completo) se a chave for igual.")
        
        uploaded_backup = st.file_uploader(
            "Arraste o arquivo de Backup (.xlsx ou .csv) aqui:", 
            type=['csv', 'xlsx'],
            key="upload_backup"
        )
        
        if uploaded_backup is not None:
            if st.button("⬆️ Iniciar Restauração de Backup", key="btn_restore_cache"):
                 # Simplificação da função import_cache_to_db para usar a lógica de upsert
                 # para manter o código limpo, vamos usar o novo flow de sincronização
                 # após a conversão, mas a função original de importação/substituição total
                 # ainda é mais rápida e estável para a restauração.
                 
                 # Reimplementando a lógica simplificada da função import_cache_to_db aqui,
                 # pois a remoção da função original causaria erros.

                try:
                    if uploaded_backup.name.endswith('.csv'):
                        df_import = pd.read_csv(uploaded_backup)
                    else: 
                        df_import = pd.read_excel(uploaded_backup, sheet_name=0)
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo: {e}")
                    df_import = pd.DataFrame()

                required_cols = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
                if not all(col in df_import.columns for col in required_cols):
                    st.error(f"Erro de Importação: O arquivo deve conter as colunas exatas: {', '.join(required_cols)}")
                    df_import = pd.DataFrame()
                
                if not df_import.empty:
                    df_import = df_import[required_cols].copy()
                    df_import['Origem_Correcao'] = 'Import_Backup'
                    df_import['Endereco_Completo_Cache'] = df_import['Endereco_Completo_Cache'].astype(str).str.strip().str.rstrip(';')
                    df_import['Latitude_Corrigida'] = pd.to_numeric(df_import['Latitude_Corrigida'], errors='coerce')
                    df_import['Longitude_Corrigida'] = pd.to_numeric(df_import['Longitude_Corrigida'], errors='coerce')
                    df_import = df_import.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])

                    if not df_import.empty:
                         perform_upsert(conn, df_import, origem='Import_Backup')
                    else:
                        st.warning("Nenhum dado válido de correção (Lat/Lon) foi encontrado no arquivo para importar.")
                    
                    
    # ----------------------------------------------------------------------------------
    # BLOCO DE LIMPAR TODO O CACHE (COM CONFIRMAÇÃO)
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.header("4.4 Limpar TODO o Cache de Geolocalização")
    st.error("⚠️ **ÁREA DE PERIGO!** Esta ação excluirá PERMANENTEMENTE todas as suas correções salvas.")
    
    if len(df_cache_original) > 0:
        confirm_clear = st.checkbox(
            f"Eu confirmo que desejo excluir permanentemente **{len(df_cache_original)}** entradas do cache.", 
            key="confirm_clear_cache"
        )
        
        if confirm_clear:
            if st.button("🔴 EXCLUIR TODOS OS DADOS DO CACHE AGORA", key="btn_final_clear_cache"):
                clear_geoloc_cache_db(conn)
    else:
        st.info("O cache já está vazio. Não há dados para excluir.")


# ----------------------------------------------------------------------------------
# ABA 5: IMPORTAR PONTOS DE CORREÇÃO (GEODATA)
# ----------------------------------------------------------------------------------

with tab_geodata_import:
    st.header("🌎 Importar Pontos de Correção para o Cache")
    st.info("Escolha abaixo o tipo de arquivo que você deseja usar para atualizar o cache de geolocalização.")
    
    tab_csv, tab_kml_xml = st.tabs([
        "📄 CSV do Google Maps (Conversão e Sincronização)",
        "🌏 KML/KMZ/XML (Google Maps/Earth)"
    ])
    
    # ======================================================
    # SUB-ABA CSV DO GOOGLE MAPS (SINCRONIZAÇÃO DE CONFLITOS)
    # ======================================================
    with tab_csv:
        st.subheader("1. Conversão e Sincronização de CSV do Google Maps")
        st.warning(f"⚠️ **Importante:** Esta função compara coordenadas. Uma diferença de Lat/Lon maior que **1 metro** será tratada como um **Conflito**.")
        st.caption("Se o endereço existir no cache, mas o novo Lat/Lon for diferente, será necessário autorizar a substituição.")

        uploaded_csv_gmaps = st.file_uploader(
            "Arraste e solte o arquivo CSV do Google Maps aqui:", 
            key="file_csv_gmaps"
        )
        
        if uploaded_csv_gmaps is not None:
            st.success(f"Arquivo '{uploaded_csv_gmaps.name}' carregado!")
            
            if st.button("➡️ Converter, Comparar e Exibir Conflitos", key="btn_convert_csv"):
                # Limpa o estado anterior
                st.session_state['df_new_entries'] = pd.DataFrame()
                st.session_state['df_conflicts'] = pd.DataFrame()
                st.session_state['overwrite_confirmed'] = False

                with st.spinner("Realizando conversão e detecção de conflitos..."):
                     df_convertido = convert_google_maps_csv(uploaded_csv_gmaps)
                     st.session_state['df_csv_convertido'] = df_convertido
                     
                     if not df_convertido.empty:
                          df_cache_atual = load_geoloc_cache(conn)
                          df_new_entries, df_conflicts = manage_cache_overwrite(df_convertido, df_cache_atual)
                          
                          st.session_state['df_new_entries'] = df_new_entries
                          st.session_state['df_conflicts'] = df_conflicts
                          
                     if df_convertido.empty:
                         st.error("Nenhum dado válido foi extraído após a conversão. Verifique as colunas do seu CSV.")
                         
            # Visualização dos dados convertidos
            if not st.session_state['df_csv_convertido'].empty:
                
                df_new_entries = st.session_state['df_new_entries']
                df_conflicts = st.session_state['df_conflicts']
                
                st.markdown("---")
                st.subheader(f"✅ Análise de Sincronização")
                
                # 1. Novas Entradas (sem conflito, prontas para inserção)
                if not df_new_entries.empty:
                    st.info(f"🆕 **{len(df_new_entries)}** Endereços NOVOS para Inserção no Cache.")
                    with st.expander("Clique para ver os novos endereços:"):
                         st.dataframe(df_new_entries, use_container_width=True)
                
                # 2. Conflitos de Substituição (requer autorização)
                if not df_conflicts.empty:
                    st.error(f"🔥 **{len(df_conflicts)}** CONFLITOS de Geolocalização Detectados.")
                    st.warning("As coordenadas no seu arquivo CSV **divergem** do que está salvo no Cache. Se você prosseguir, as coordenadas atuais serão **substituídas**.")
                    with st.expander("Clique para ver os endereços em conflito (Coordenadas Atuais vs. Novas):"):
                         st.dataframe(df_conflicts, use_container_width=True)
                    
                    # Checkbox de Autorização
                    if st.session_state['overwrite_confirmed'] == False:
                        st.session_state['overwrite_confirmed'] = st.checkbox(
                            f"Eu confirmo a substituição dos **{len(df_conflicts)}** endereços em conflito.", 
                            key="confirm_overwrite"
                        )
                
                # 3. Botão de Salvamento Final
                total_to_update = len(df_new_entries) + (len(df_conflicts) if st.session_state['overwrite_confirmed'] else 0)
                
                if total_to_update > 0:
                    if st.button(f"💾 Sincronizar e Salvar {total_to_update} Entradas no Cache", key="btn_sync_save_final"):
                        
                        df_to_upsert = df_new_entries.copy()
                        
                        if st.session_state['overwrite_confirmed']:
                             # Se autorizado, adiciona os conflitos para upsert
                             df_conflicts_to_save = df_conflicts.copy()
                             df_conflicts_to_save.rename(columns={'Latitude_Nova': 'Latitude_Corrigida', 'Longitude_Nova': 'Longitude_Corrigida'}, inplace=True)
                             
                             df_to_upsert = pd.concat([df_to_upsert, df_conflicts_to_save[['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']]])
                            
                        # Executa o upsert
                        perform_upsert(conn, df_to_upsert, origem='CSV_Sync')
                        st.session_state['df_csv_convertido'] = pd.DataFrame() # Limpa a visualização
                        st.session_state['overwrite_confirmed'] = False # Reseta a confirmação
                else:
                    st.info("Nenhum item novo ou conflito autorizado para salvar.")

    # ======================================================
    # SUB-ABA KML/KMZ/XML
    # ======================================================
    with tab_kml_xml:
        st.subheader("2. Importação de KML/KMZ/XML (Método Alternativo)")
        st.caption("Este método faz uma **substituição direta (Overwrite)** de qualquer entrada de cache existente com a mesma chave. Use com cautela.")

        uploaded_kml_kmz = st.file_uploader(
            "Arraste e solte o arquivo KML (.kml), KMZ (.kmz) ou XML (.xml) aqui:", 
            type=['kml', 'kmz', 'xml'], 
            key="file_kml_kmz"
        )
        
        if uploaded_kml_kmz is not None:
            st.success(f"Arquivo '{uploaded_kml_kmz.name}' carregado!")
            
            if st.button("➡️ Processar KML/KMZ/XML e Extrair Dados", key="btn_parse_kml_kmz"):
                with st.spinner("Processando o arquivo geoespacial..."):
                     df_kml = parse_kml_data(uploaded_kml_kmz)
                     st.session_state['df_kml_extraido'] = df_kml
                     
            if not st.session_state['df_kml_extraido'].empty:
                df_kml_visualizacao = st.session_state['df_kml_extraido']
                
                st.markdown("---")
                st.subheader(f"✅ {len(df_kml_visualizacao)} Pontos Extraídos (Prontos para Substituição)")
                st.dataframe(df_kml_visualizacao, use_container_width=True)
                
                st.markdown("---")
                
                if st.button(f"💾 Substituir e Salvar {len(df_kml_visualizacao)} Pontos no Cache", key="btn_save_kml_kmz_to_cache_final"):
                    import_kml_to_db(conn, df_kml_visualizacao)
                    
            elif uploaded_kml_kmz is not None and st.session_state['df_kml_extraido'].empty:
                 st.info("Carregue o arquivo e clique em 'Processar KML/KMZ/XML e Extrair Dados'.")
