# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
import math
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode


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

# --- Configurações de MIME Type (CORREÇÃO DE ERRO) ---
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# --- Configurações de Banco de Dados (GERAL) ---
DB_NAME = "geoloc_cache.sqlite"

# --- Configurações de Cache de Geolocalização (Lat/Lon Principal) ---
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
PRIMARY_KEYS = ['Endereco_Completo_Cache'] 

# --- V30: Configurações de Cache de Endereço Alternativo (Alt Address) ---
ALT_ADDRESS_TABLE_NAME = "alt_address_cache_v1"
# O Endereco_Principal_Cache é a chave de busca (Endereço + Bairro)
ALT_CACHE_COLUMNS = ['Endereco_Principal_Cache', 'Alt_Address', 'Alt_Latitude', 'Alt_Longitude']


# ===============================================
# FUNÇÕES DE BANCO DE Dados (SQLite)
# ===============================================

@st.cache_resource
def get_db_connection():
    """Cria e retorna a conexão com o banco de dados SQLite."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=10)
    return conn

def create_table_if_not_exists(conn):
    """Cria a tabela de cache de geolocalização principal se ela não existir."""
    pk_str = ', '.join(PRIMARY_KEYS)
    query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        Endereco_Completo_Cache TEXT PRIMARY KEY,
        Latitude_Corrigida REAL,
        Longitude_Corrigida REAL
    );
    """
    try:
        conn.execute(query)
    except Exception as e:
        st.error(f"Erro ao criar tabela de Geoloc: {e}")

# V30: Nova Função - Cria a tabela de Cache de Endereço Alternativo
def create_alt_address_table_if_not_exists(conn):
    """Cria a tabela de cache de endereço alternativo se ela não existir."""
    query = f"""
    CREATE TABLE IF NOT EXISTS {ALT_ADDRESS_TABLE_NAME} (
        Endereco_Principal_Cache TEXT PRIMARY KEY,
        Alt_Address TEXT,
        Alt_Latitude REAL,
        Alt_Longitude REAL
    );
    """
    try:
        conn.execute(query)
        conn.commit()
    except Exception as e:
        st.error(f"Erro ao criar tabela de Alt Address: {e}")

# CORREÇÃO CRÍTICA (UnhashableParamError)
@st.cache_data(hash_funcs={sqlite3.Connection: lambda _: "constant_db_hash"})
def load_geoloc_cache(conn):
    """Carrega todo o cache de geolocalização para um DataFrame."""
    try:
        df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except pd.io.sql.DatabaseError:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de geolocalização: {e}")
        return pd.DataFrame(columns=CACHE_COLUMNS)

# V30: Nova Função - Carrega o Cache de Endereço Alternativo
@st.cache_data(hash_funcs={sqlite3.Connection: lambda _: "constant_db_hash"})
def load_alt_address_cache(conn):
    """Carrega todo o cache de endereço alternativo para um DataFrame."""
    try:
        df_cache = pd.read_sql_query(f"SELECT * FROM {ALT_ADDRESS_TABLE_NAME}", conn)
        df_cache['Alt_Latitude'] = pd.to_numeric(df_cache['Alt_Latitude'], errors='coerce')
        df_cache['Alt_Longitude'] = pd.to_numeric(df_cache['Alt_Longitude'], errors='coerce')
        return df_cache
    except pd.io.sql.DatabaseError:
        return pd.DataFrame(columns=ALT_CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de endereço alternativo: {e}")
        return pd.DataFrame(columns=ALT_CACHE_COLUMNS)


def save_single_entry_to_db(conn, endereco, lat, lon):
    """Salva uma única entrada (Endereço Completo + Lat/Lon) no cache principal (UPSERT)."""
    
    upsert_query = f"""
    INSERT OR REPLACE INTO {TABLE_NAME} 
    (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida) 
    VALUES (?, ?, ?);
    """
    
    try:
        conn.execute(upsert_query, (endereco, lat, lon))
        conn.commit()
        st.success(f"Correção salva para: **{endereco}**.")
        
        load_geoloc_cache.clear() 
        st.rerun() 
        
    except Exception as e:
        st.error(f"Erro ao salvar a correção no banco de dados: {e}")

# V30: Nova Função - Salva uma única entrada no Cache de Endereço Alternativo
def save_single_alt_entry_to_db(conn, endereco_principal, alt_address, alt_lat, alt_lon):
    """Salva uma única entrada (Endereço Principal + Alt Address/Lat/Lon) no cache alternativo (UPSERT)."""
    
    upsert_query = f"""
    INSERT OR REPLACE INTO {ALT_ADDRESS_TABLE_NAME} 
    (Endereco_Principal_Cache, Alt_Address, Alt_Latitude, Alt_Longitude) 
    VALUES (?, ?, ?, ?);
    """
    
    try:
        conn.execute(upsert_query, (endereco_principal, alt_address, alt_lat, alt_lon))
        conn.commit()
        st.success(f"Endereço Alternativo salvo para a chave: **{endereco_principal}**.")
        
        load_alt_address_cache.clear() 
        st.rerun() 
        
    except Exception as e:
        st.error(f"Erro ao salvar o endereço alternativo no banco de dados: {e}")
        
        
def import_cache_to_db(conn, uploaded_file):
    """Importa o cache principal para o banco de dados (UPSERT)."""
    # (Lógica da V29 para importar cache principal permanece a mesma)
    
    # 1. Leitura do arquivo
    try:
        if uploaded_file.name.endswith('.csv'):
            df_import = pd.read_csv(uploaded_file)
        else: # Assumindo Excel (.xlsx)
            df_import = pd.read_excel(uploaded_file, sheet_name=0)
            
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return 0

    # 2. Validação e Preparação
    required_cols = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
    if not all(col in df_import.columns for col in required_cols):
        st.error(f"Erro de Importação: O arquivo deve conter as colunas exatas: {', '.join(required_cols)}")
        return 0

    # Conversão de tipos e limpeza
    df_import = df_import[required_cols].copy()
    df_import['Endereco_Completo_Cache'] = df_import['Endereco_Completo_Cache'].astype(str).str.strip().str.rstrip(';')
    
    # Padroniza coordenadas (troca vírgula por ponto para float)
    df_import['Latitude_Corrigida'] = df_import['Latitude_Corrigida'].astype(str).str.replace(',', '.', regex=False)
    df_import['Longitude_Corrigida'] = df_import['Longitude_Corrigida'].astype(str).str.replace(',', '.', regex=False)
    
    df_import['Latitude_Corrigida'] = pd.to_numeric(df_import['Latitude_Corrigida'], errors='coerce')
    df_import['Longitude_Corrigida'] = pd.to_numeric(df_import['Longitude_Corrigida'], errors='coerce')
    
    df_import = df_import.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])
    
    if df_import.empty:
        st.warning("Nenhum dado válido de correção (Lat/Lon) foi encontrado no arquivo para importar.")
        return 0
        
    # 3. Inserção no Banco (UPSERT)
    insert_count = 0
    try:
        with st.spinner(f"Processando a importação de {len(df_import)} linhas..."):
            for index, row in df_import.iterrows():
                endereco = row['Endereco_Completo_Cache']
                lat = row['Latitude_Corrigida']
                lon = row['Longitude_Corrigida']
                
                upsert_query = f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida) 
                VALUES (?, ?, ?);
                """
                conn.execute(upsert_query, (endereco, lat, lon))
                insert_count += 1
            
            conn.commit()
            
            # 4. Finalização
            load_geoloc_cache.clear()
            count_after = len(load_geoloc_cache(conn))
            
            st.success(f"Importação de backup concluída! **{insert_count}** entradas processadas (atualizadas ou adicionadas). O cache agora tem **{count_after}** entradas.")
            
            st.rerun() 
            
            return count_after

    except Exception as e:
        st.error(f"Erro crítico ao inserir dados no cache. Verifique se o arquivo está correto. Erro: {e}")
        return 0

# V30: Nova Função - Importa o Cache de Endereço Alternativo
def import_alt_cache_to_db(conn, uploaded_file):
    """Importa o cache alternativo para o banco de dados (UPSERT)."""
    
    # 1. Leitura do arquivo
    try:
        if uploaded_file.name.endswith('.csv'):
            df_import = pd.read_csv(uploaded_file)
        else: # Assumindo Excel (.xlsx)
            df_import = pd.read_excel(uploaded_file, sheet_name=0)
            
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return 0

    # 2. Validação e Preparação
    required_cols = ALT_CACHE_COLUMNS
    if not all(col in df_import.columns for col in required_cols):
        st.error(f"Erro de Importação: O arquivo deve conter as colunas exatas: {', '.join(required_cols)}")
        return 0

    # Conversão de tipos e limpeza
    df_import = df_import[required_cols].copy()
    df_import['Endereco_Principal_Cache'] = df_import['Endereco_Principal_Cache'].astype(str).str.strip().str.rstrip(';')
    df_import['Alt_Address'] = df_import['Alt_Address'].astype(str).str.strip().str.rstrip(';')
    
    # Padroniza coordenadas (troca vírgula por ponto para float)
    df_import['Alt_Latitude'] = df_import['Alt_Latitude'].astype(str).str.replace(',', '.', regex=False)
    df_import['Alt_Longitude'] = df_import['Alt_Longitude'].astype(str).str.replace(',', '.', regex=False)
    
    df_import['Alt_Latitude'] = pd.to_numeric(df_import['Alt_Latitude'], errors='coerce')
    df_import['Alt_Longitude'] = pd.to_numeric(df_import['Alt_Longitude'], errors='coerce')
    
    df_import = df_import.dropna(subset=['Alt_Latitude', 'Alt_Longitude'])
    
    if df_import.empty:
        st.warning("Nenhum dado válido de endereço alternativo (Lat/Lon) foi encontrado no arquivo para importar.")
        return 0
        
    # 3. Inserção no Banco (UPSERT)
    insert_count = 0
    try:
        with st.spinner(f"Processando a importação de {len(df_import)} linhas..."):
            for index, row in df_import.iterrows():
                endereco_principal = row['Endereco_Principal_Cache']
                alt_address = row['Alt_Address']
                alt_lat = row['Alt_latitude']
                alt_lon = row['Alt_Longitude']
                
                upsert_query = f"""
                INSERT OR REPLACE INTO {ALT_ADDRESS_TABLE_NAME} 
                (Endereco_Principal_Cache, Alt_Address, Alt_Latitude, Alt_Longitude) 
                VALUES (?, ?, ?, ?);
                """
                conn.execute(upsert_query, (endereco_principal, alt_address, alt_lat, alt_lon))
                insert_count += 1
            
            conn.commit()
            
            # 4. Finalização
            load_alt_address_cache.clear()
            count_after = len(load_alt_address_cache(conn))
            
            st.success(f"Importação de backup concluída! **{insert_count}** entradas processadas (atualizadas ou adicionadas). O cache de Alt Address agora tem **{count_after}** entradas.")
            
            st.rerun() 
            
            return count_after

    except Exception as e:
        st.error(f"Erro crítico ao inserir dados no cache. Verifique se o arquivo está correto. Erro: {e}")
        return 0


# ------------------------------------------------------------------
# FUNÇÃO PARA LIMPAR TODO O CACHE (EXCLUSÃO)
# ------------------------------------------------------------------
def clear_geoloc_cache_db(conn):
    """Exclui todos os dados da tabela de cache de geolocalização principal."""
    
    query = f"DELETE FROM {TABLE_NAME};"
    
    try:
        conn.execute(query)
        conn.commit()
        
        load_geoloc_cache.clear()
        st.success("✅ **Sucesso!** Todos os dados do cache de geolocalização (Principal) foram excluídos permanentemente.")
        st.rerun() 
        
    except Exception as e:
        st.error(f"❌ Erro ao limpar o cache principal: {e}")

# V30: Nova Função - Limpa o Cache de Endereço Alternativo
def clear_alt_address_cache_db(conn):
    """Exclui todos os dados da tabela de cache de endereço alternativo."""
    
    query = f"DELETE FROM {ALT_ADDRESS_TABLE_NAME};"
    
    try:
        conn.execute(query)
        conn.commit()
        
        load_alt_address_cache.clear()
        st.success("✅ **Sucesso!** Todos os dados do cache de Endereço Alternativo foram excluídos permanentemente.")
        st.rerun() 
        
    except Exception as e:
        st.error(f"❌ Erro ao limpar o cache de Alt Address: {e}")


# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# ===============================================

def limpar_endereco(endereco):
    """Normaliza o texto do endereço para melhor comparação, mantendo números e vírgulas."""
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    endereco = re.sub(r'\s+', ' ', endereco)
    
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return endereco

def get_most_common_or_empty(x):
    """Retorna o valor mais comum de uma Série Pandas."""
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    return x_limpo.mode().iloc[0]


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade, df_cache_geoloc, conn):
    """
    Função principal que aplica a correção (usando cache 100% match) e o agrupamento,
    incluindo o lookup de Endereço Alternativo.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, COLUNA_BAIRRO, 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None 

    df = df_entrada.copy()
    
    # Preparação
    df[COLUNA_BAIRRO] = df[COLUNA_BAIRRO].astype(str).str.strip().replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    
    
    # CHAVE DE BUSCA DE CACHE (Endereço Completo + Bairro)
    df['Chave_Busca_Cache'] = (
        df[COLUNA_ENDERECO].astype(str).str.strip() + 
        ', ' + 
        df[COLUNA_BAIRRO].astype(str).str.strip()
    )
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*$', '', regex=True)
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*,', ',', regex=True)

    
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)

    
    # =========================================================================
    # PASSO 1: APLICAR LOOKUP NO CACHE DE GEOLOCALIZAÇÃO (100% MATCH)
    # =========================================================================
    
    if not df_cache_geoloc.empty:
        df_cache_lookup = df_cache_geoloc.rename(columns={
            'Endereco_Completo_Cache': 'Chave_Cache_DB', 
            'Latitude_Corrigida': 'Cache_Lat',
            'Longitude_Corrigida': 'Cache_Lon'
        })
        
        df = pd.merge(
            df, 
            df_cache_lookup, 
            left_on='Chave_Busca_Cache', # Chave combinada da planilha
            right_on='Chave_Cache_DB',   # Endereço completo do cache
            how='left'
        )
        
        cache_mask = df['Cache_Lat'].notna()
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Cache_Lat']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Cache_Lon']
        
        df = df.drop(columns=['Chave_Cache_DB', 'Cache_Lat', 'Cache_Lon'], errors='ignore')
    
    # =========================================================================
    # V30: PASSO 2: MERGE COM CACHE DE ENDEREÇO ALTERNATIVO (PERSISTENTE)
    # =========================================================================
    df[COLUNA_SEQUENCE] = df[COLUNA_SEQUENCE].astype(str)
    
    # Carrega o cache persistente de Endereços Alternativos
    df_alt_address_cache = load_alt_address_cache(conn)
    
    if not df_alt_address_cache.empty:
        df_alt_address_cache = df_alt_address_cache.rename(columns={
            'Endereco_Principal_Cache': 'Chave_Alt_DB', 
        })
        
        # Merge baseado na mesma chave de busca (Endereço + Bairro)
        df = pd.merge(
            df,
            df_alt_address_cache[['Chave_Alt_DB', 'Alt_Address', 'Alt_Latitude', 'Alt_Longitude']],
            left_on='Chave_Busca_Cache',
            right_on='Chave_Alt_DB',
            how='left'
        )
        
        df = df.drop(columns=['Chave_Alt_DB'], errors='ignore')
    
    # Preenche NA para as novas colunas para permitir a agregação, caso não tenham vindo do merge
    if 'Alt_Address' not in df.columns:
         df['Alt_Address'] = None
         df['Alt_Latitude'] = None
         df['Alt_Longitude'] = None
         
    # Remove coluna auxiliar de busca (não é mais necessária após os merges)
    df = df.drop(columns=['Chave_Busca_Cache'], errors='ignore')
         
    # =========================================================================
    # PASSO 3: FUZZY MATCHING (CORREÇÃO DE ENDEREÇO E AGRUPAMENTO)
    # =========================================================================
    
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching e Agrupamento...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos > 0:
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

    # Aplicação do Endereço Corrigido (Chave de Agrupamento)
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # Agrupamento (Chave: Endereço Corrigido + Cidade + BAIRRO)
    colunas_agrupamento = ['Endereco_Corrigido', 'City', COLUNA_BAIRRO] 
    
    # =========================================================================
    # AGRUPAMENTO - ADICIONANDO ENDEREÇOS ALTERNATIVOS
    # =========================================================================
    # Os campos Alt_Address/Lat/Lon já estão no DF após o merge do Cache Alt Address
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'\*', '', str(y))) if re.sub(r'\*', '', str(y)).isdigit() else float('inf'))))), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        # Dados de Suporte
        Bairro_Agrupado=(COLUNA_BAIRRO, get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        
        Min_Sequence=('Sequence_Num', 'min'),
        
        # Campos Alt Address (Agrega os endereços alternativos únicos no grupo)
        Alt_Addresses_List=('Alt_Address', lambda x: '; '.join(x.dropna().astype(str).unique())),
        Alt_Latitudes_List=('Alt_Latitude', lambda x: '; '.join(map(str, x.dropna().astype(str).unique()))),
        Alt_Longitudes_List=('Alt_Longitude', lambda x: '; '.join(map(str, x.dropna().astype(str).unique()))),
        
    ).reset_index()

    # Ordenação
    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    # =========================================================================
    # FORMATAÇÃO DO DF PARA O CIRCUIT
    # =========================================================================
    endereco_completo_circuit = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro_Agrupado'].str.strip() 
    )
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*,', ',', regex=True)
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*$', '', regex=True) 
    
    # 1. Cria a coluna Alt_LatLon
    df_agrupado['Alt_LatLon'] = (
        df_agrupado['Alt_Latitudes_List'].astype(str) + ', ' + 
        df_agrupado['Alt_Longitudes_List'].astype(str)
    )
    df_agrupado['Alt_LatLon'] = df_agrupado['Alt_LatLon'].str.replace(r',\s*$', '', regex=True).str.strip() 
    df_agrupado['Alt_LatLon'] = df_agrupado['Alt_LatLon'].replace(' , ', '', regex=False).replace(r'^,$|^$', None, regex=True) 

    # 2. Adiciona as anotações alternativas se existirem
    alt_notes_mask = df_agrupado['Alt_Addresses_List'].notna() & (df_agrupado['Alt_Addresses_List'] != '')
    df_agrupado.loc[alt_notes_mask, 'Notes_Alternativas'] = (
        ' | ENDEREÇO ALTERNATIVO ENCONTRADO: ' + df_agrupado['Alt_Addresses_List'].astype(str) + 
        ' (COORDENADAS: ' + df_agrupado['Alt_LatLon'].astype(str) + ')'
    )
    df_agrupado['Notes_Alternativas'] = df_agrupado['Notes_Alternativas'].fillna('')


    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode_Agrupado'] +
        df_agrupado['Notes_Alternativas'] 
    )

    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas,
        'Alt_Address': df_agrupado['Alt_Addresses_List'].replace('', None), 
        'Alt_LatLon': df_agrupado['Alt_LatLon']
    }) 
    
    return df_circuit

# ... (Função processar_rota_para_impressao permanece a mesma) ...

def processar_rota_para_impressao(df_input):
    """
    Processa o DataFrame da rota, extrai 'Ordem ID' da coluna 'Notes' e prepara para cópia.
    """
    coluna_notes_lower = 'notes'
    
    if coluna_notes_lower not in df_input.columns:
        coluna_notes_lower = 'Notes' 
        if coluna_notes_lower not in df_input.columns:
             raise KeyError(f"A coluna 'Notes' ou 'notes' não foi encontrada.") 
    
    df = df_input.copy()
    df[coluna_notes_lower] = df[coluna_notes_lower].astype(str)
    df = df.dropna(subset=[coluna_notes_lower]) 
    
    # Coluna "Ordem ID" é o primeiro campo antes do ';'
    df_split = df[coluna_notes_lower].str.split(';', n=1, expand=True)
    
    df['Ordem ID'] = df_split[0].astype(str).str.strip().str.strip('"') 
    df['Anotações Completas'] = df_split[1].astype(str).str.strip().str.strip('"') if 1 in df_split.columns else ""
    
    # DataFrame GERAL (Para Cópia e Download)
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    # DataFrame FINAL GERAL
    df_final_geral = df[['Lista de Impressão', 'address']].copy() 
    
    # 1. FILTRAR VOLUMOSOS
    df_volumosos = df[df['Ordem ID'].str.contains(r'\*', regex=False, na=False)].copy()
    df_volumosos_impressao = df_volumosos[['Lista de Impressão', 'address']].copy() 
    
    # 2. FILTRAR NÃO-VOLUMOSOS
    df['Lista_IDs'] = df['Ordem ID'].str.split(' - ', n=1, expand=True)[0]
    df_nao_volumosos = df[df['Lista_IDs'].str.contains(r'\d+(?![\*])', regex=True, na=False)].copy() 
    
    df_nao_volumosos_impressao = df_nao_volumosos[['Lista de Impressão', 'address']].copy()
    
    df = df.drop(columns=['Lista_IDs'], errors='ignore')

    return df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

# 1. Conexão com o Banco de Dados (Executada uma vez)
conn = get_db_connection()
create_table_if_not_exists(conn)
create_alt_address_table_if_not_exists(conn) # V30: Cria a tabela de Endereço Alternativo

st.title("🗺️ Flow Completo Circuit (Pré, Pós e Cache)")

# CRIAÇÃO DAS ABAS (Quatro Abas)
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Pré-Roteirização (Importação)", 
    "📋 Pós-Roteirização (Impressão/Cópia)", 
    "💾 Gerenciar Cache de Geoloc", # Novo título
    "📌 Gerenciar Matches Alternativos" # Nova Aba
])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa aplica as correções dos **Caches Persistentes** e agrupa os endereços.")

    # Inicializa o estado
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'volumoso_ids' not in st.session_state:
        st.session_state['volumoso_ids'] = set() 
    
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
            
            # --- VALIDAÇÃO DE COLUNAS ---
            colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, COLUNA_BAIRRO, 'City', 'Zipcode/Postal code']
            for col in colunas_essenciais:
                 if col not in df_input_pre.columns:
                     raise KeyError(f"A coluna '{col}' está faltando na sua planilha.")
            
            # Resetar as marcações se um novo arquivo for carregado
            if st.session_state.get('last_uploaded_name') != uploaded_file_pre.name:
                 st.session_state['volumoso_ids'] = set()
                 st.session_state['last_uploaded_name'] = uploaded_file_pre.name
                 
            st.session_state['df_original'] = df_input_pre.copy()
            st.success(f"Arquivo '{uploaded_file_pre.name}' carregado! Total de **{len(df_input_pre)}** registros.")
            
        except KeyError as ke:
             st.error(f"Erro de Coluna: {ke}")
             st.session_state['df_original'] = None
        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar o arquivo. Verifique o formato. Erro: {e}")

    
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("1.2 Marcar Pacotes Volumosos (Volumosos = *)")
    
    if st.session_state['df_original'] is not None:
        
        df_temp = st.session_state['df_original'].copy()
        
        # --- ORDENAÇÃO NUMÉRICA CORRETA ---
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
        
        st.info("⚠️ **Endereços Alternativos** são carregados automaticamente do **Cache na Aba 4**.")
        
        limite_similaridade_ajustado = st.slider(
            'Ajuste a Precisão do Corretor (Fuzzy Matching):',
            min_value=80,
            max_value=100,
            value=100, 
            step=1,
            help="Use 100% para garantir que endereços na mesma rua com números diferentes não sejam agrupados (recomendado)."
        )
        st.info(f"O limite de similaridade está em **{limite_similaridade_ajustado}%**. Isso afeta o agrupamento, mas a geolocalização exata virá do cache.")
        
        
        if st.button("🚀 Iniciar Corretor e Agrupamento", key="btn_pre_final"):
            
            # 1. Aplicar a marcação * no DF antes de processar
            df_para_processar = st.session_state['df_original'].copy()
            df_para_processar[COLUNA_SEQUENCE] = df_para_processar[COLUNA_SEQUENCE].astype(str)
            
            for id_volumoso in st.session_state['volumoso_ids']:
                str_id_volumoso = str(id_volumoso)
                df_para_processar.loc[
                    df_para_processar[COLUNA_SEQUENCE] == str_id_volumoso, 
                    COLUNA_SEQUENCE
                ] = str_id_volumoso + '*'

            # 2. Carregar Cache de Geolocalização Principal
            df_cache = load_geoloc_cache(conn)

            # 3. Iniciar o processamento e agrupamento
            with st.spinner('Aplicando caches persistentes (Geoloc e Alternativo), processando dados...'):
                 # V30: Passando a conexão para carregar o Cache Alt Address internamente
                 df_circuit = processar_e_corrigir_dados(
                     df_para_processar, 
                     limite_similaridade_ajustado, 
                     df_cache,
                     conn # Passa a conexão
                 )
            
            if df_circuit is not None:
                
                st.markdown("---")
                st.header("✅ Resultado Concluído!")
                
                total_entradas = len(st.session_state['df_original'])
                total_agrupados = len(df_circuit)
                
                st.metric(
                    label="Endereços Únicos Agrupados",
                    value=total_agrupados,
                    delta=f"-{total_entradas - total_agrupados} agrupados"
                )
                
                # 1. FILTRAR DADOS PARA A NOVA ABA "APENAS_VOLUMOSOS"
                df_volumosos_separado = df_circuit[
                    df_circuit['Order ID'].astype(str).str.contains(r'\*', regex=True)
                ].copy()
                
                # --- SAÍDA PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                st.subheader("Arquivo para Roteirização (Circuit)")
                st.info("Colunas `Alt_Address` e `Alt_LatLon` estão no Excel para substituição manual no Circuit, se necessário.")
                
                # Exibe apenas as colunas principais para visualização
                df_display = df_circuit[['Order ID', 'Address', 'Latitude', 'Longitude', 'Notes']].copy()
                st.dataframe(df_display, use_container_width=True)
                
                # Download Circuit 
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openypxl') as writer:
                    df_circuit.to_excel(writer, index=False, sheet_name='Circuit_Import_Geral') 
                    
                    if not df_volumosos_separado.empty:
                        df_volumosos_separado.to_excel(writer, index=False, sheet_name='APENAS_VOLUMOSOS')
                        st.info(f"O arquivo de download conterá uma aba extra com **{len(df_volumosos_separado)}** endereços que incluem pacotes volumosos.")
                    else:
                        st.info("Nenhum pacote volumoso marcado. O arquivo de download terá apenas a aba principal.")
                        
                buffer_circuit.seek(0)
                
                st.download_button(
                    label="📥 Baixar ARQUIVO PARA CIRCUIT",
                    data=buffer_circuit,
                    file_name="Circuit_Import_FINAL_MARCADO.xlsx",
                    mime=EXCEL_MIME_TYPE, 
                    key="download_excel_circuit"
                )


# ----------------------------------------------------------------------------------
# ABA 2: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO E SEPARAÇÃO DE VOLUMOSOS)
# ----------------------------------------------------------------------------------

with tab2:
    st.header("2. Limpar Saída do Circuit para Impressão")
    st.warning("⚠️ Atenção: Use o arquivo CSV/Excel que foi gerado *após a conversão* do PDF da rota do Circuit.")

    st.markdown("---")
    st.subheader("2.1 Carregar Arquivo da Rota")

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
            key="sheet_name_input"
        )

    if uploaded_file_pos is not None:
        try:
            if uploaded_file_pos.name.endswith('.csv'):
                df_input_pos = pd.read_csv(uploaded_file_pos)
            else:
                df_input_pos = pd.read_excel(uploaded_file_pos, sheet_name=sheet_name)
            
            df_input_pos.columns = df_input_pos.columns.str.strip() 
            df_input_pos.columns = df_input_pos.columns.str.lower()
            
            if 'notes' not in df_input_pos.columns:
                df_input_pos.columns = [
                    col.lower() if col.lower() == 'notes' or col.lower() == 'address' else col
                    for col in df_input_pos.columns
                ]
            
            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_input_pos)}** registros.")
            
            df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao = processar_rota_para_impressao(df_input_pos)
            
            if df_final_geral is not None and not df_final_geral.empty:
                st.markdown("---")
                st.subheader("2.2 Resultado Final (Lista de Impressão GERAL)")
                
                df_visualizacao_geral = df_final_geral.copy()
                df_visualizacao_geral.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                st.dataframe(df_visualizacao_geral, use_container_width=True)

                copia_data_geral = '\n'.join(df_final_geral['Lista de Impressão'].astype(str).tolist())
                
                
                # --- SEÇÃO DEDICADA AOS NÃO-VOLUMOSOS ---
                st.markdown("---")
                st.header("✅ Lista de Impressão APENAS NÃO-VOLUMOSOS")
                
                if not df_nao_volumosos_impressao.empty:
                    st.success(f"Foram encontrados **{len(df_nao_volumosos_impressao)}** endereços com pacotes NÃO-volumosos (puros ou mistos) nesta rota.")
                    
                    df_visualizacao_nao_vol = df_nao_volumosos_impressao.copy()
                    df_visualizacao_nao_vol.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                    st.dataframe(df_visualizacao_nao_vol, use_container_width=True)
                    
                    copia_data_nao_volumosos = '\n'.join(df_nao_volumosos_impressao['Lista de Impressão'].astype(str).tolist())
                    
                else:
                    st.info("Nenhum pacote não-volumoso encontrado nesta rota (todos os pedidos são volumosos ou a lista está vazia).")
                    
                # --- SEÇÃO DEDICADA AOS VOLUMOSOS ---
                st.markdown("---")
                st.header("📦 Lista de Impressão APENAS VOLUMOSOS")
                
                if not df_volumosos_impressao.empty:
                    st.warning(f"Foram encontrados **{len(df_volumosos_impressao)}** endereços com pacotes volumosos (puros ou mistos) nesta rota.")
                    
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
                st.error(f"Erro de Aba: A aba **'{sheet_name}'** não foi encontrada no arquivo Excel. Verifique o nome da aba.")
            elif 'notes' in str(ke):
                 st.error(f"Erro de Coluna: A coluna 'Notes' ou 'notes' não foi encontrada. Verifique se o arquivo da rota está correto.")
            elif 'address' in str(ke):
                 st.error(f"Erro de Coluna: A coluna 'Address' (ou 'address') não foi encontrada. Verifique o arquivo de rota.")
            else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o arquivo da rota (PDF convertido) está no formato CSV ou Excel. Erro: {e}")
            
    
    # Renderização das áreas de cópia e download
    if uploaded_file_pos is not None:
        
        # --- ÁREA DE CÓPIA GERAL ---
        st.markdown("### 2.3 Copiar para a Área de Transferência (Lista GERAL)")
        st.info("Para copiar: **Selecione todo o texto** abaixo (Ctrl+A / Cmd+A) e pressione **Ctrl+C / Cmd+C**.")
        
        st.text_area(
            'Conteúdo da Lista de Impressão GERAL (Alinhado à Esquerda):', 
            copia_data_geral, 
            height=300,
            key="text_area_geral"
        )

        # --- ÁREA DE CÓPIA NÃO-VOLUMOSOS ---
        if not df_nao_volumosos_impressao.empty if df_nao_volumosos_impressao is not None else False:
            st.markdown("### 2.4 Copiar para a Área de Transferência (APENAS NÃO-Volumosos)")
            st.success("Lista Filtrada: Contém **somente** os endereços com pacotes **NÃO-volumosos** (puros ou agrupamentos mistos).")
            
            st.text_area(
                'Conteúdo da Lista de Impressão NÃO-VOLUMOSOS (Alinhado à Esquerda):', 
                copia_data_nao_volumosos, 
                height=150,
                key="text_area_nao_volumosos"
            )
        
        # --- ÁREA DE CÓPIA VOLUMOSOS ---
        if not df_volumosos_impressao.empty if df_volumosos_impressao is not None else False:
            st.markdown("### 2.5 Copiar para a Área de Transferência (APENAS Volumosos)")
            st.warning("Lista Filtrada: Contém **somente** os endereços com pacotes volumosos (puros ou agrupamentos mistos).")
            
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
# ABA 3: GERENCIAR CACHE DE GEOLOCALIZAÇÃO (PRINCIPAL LAT/LON)
# ----------------------------------------------------------------------------------

def clear_lat_lon_fields():
    """Limpa os campos de Latitude/Longitude e o campo de colar coordenadas."""
    if 'form_new_lat' in st.session_state:
        st.session_state['form_new_lat'] = ""
    if 'form_new_lon' in st.session_state:
        st.session_state['form_new_lon'] = ""
    if 'form_colar_coord' in st.session_state:
        st.session_state['form_colar_coord'] = ""
    if 'form_new_endereco' in st.session_state:
        st.session_state['form_new_endereco'] = ""

def apply_google_coords(lat_key, lon_key, coord_input_key):
    """Processa a string colada do Google Maps e preenche Lat/Lon."""
    coord_string = st.session_state.get(coord_input_key, '')
    if not coord_string:
        st.error("Nenhuma coordenada foi colada. Cole o texto do Google Maps, ex: -23,5139753, -52,1131268")
        return

    coord_string_clean = coord_string.strip()
    
    try:
        matches = re.findall(r'(-?\d+[\.,]\d+)', coord_string_clean.replace(' ', ''))
        
        if len(matches) >= 2:
            lat = float(matches[0].replace(',', '.'))
            lon = float(matches[1].replace(',', '.'))
            
            st.session_state[lat_key] = str(lat)
            st.session_state[lon_key] = str(lon)
            st.success(f"Coordenadas aplicadas: Lat: **{lat}**, Lon: **{lon}**")
            return
            
    except ValueError:
        parts = coord_string_clean.split(',')
        if len(parts) >= 2:
             try:
                lat = float(parts[0].replace(',', '.').strip()) 
                lon = float(parts[1].replace(',', '.').strip())
                
                st.session_state[lat_key] = str(lat)
                st.session_state[lon_key] = str(lon)
                st.success(f"Coordenadas aplicadas: Lat: **{lat}**, Lon: **{lon}**")
                return
             except ValueError:
                pass 
                
    st.error(f"Não foi possível extrair duas coordenadas válidas da string: '{coord_string}'. Verifique o formato. Exemplo: -23.5139753, -52.1131268")

# Funções auxiliares para o Cache Alternativo
def clear_alt_address_fields():
    """Limpa os campos de Endereço Alternativo."""
    if 'form_alt_endereco_principal' in st.session_state:
        st.session_state['form_alt_endereco_principal'] = ""
    if 'form_alt_endereco' in st.session_state:
        st.session_state['form_alt_endereco'] = ""
    if 'form_alt_lat' in st.session_state:
        st.session_state['form_alt_lat'] = ""
    if 'form_alt_lon' in st.session_state:
        st.session_state['form_alt_lon'] = ""
    if 'form_alt_colar_coord' in st.session_state:
        st.session_state['form_alt_colar_coord'] = ""
        
def export_cache(df_cache, columns):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_cache[columns].to_excel(writer, index=False, sheet_name='Cache')
    buffer.seek(0)
    return buffer


with tab3:
    st.header("💾 Gerenciamento do Cache de Geolocalização Principal (Lat/Lon)")
    st.info("Aqui você registra as correções de Latitude e Longitude para endereços que o Circuit não consegue roteirizar.")

    # 1. Carrega o cache salvo
    df_cache_original = load_geoloc_cache(conn).fillna("")
    
    
    # =========================================================================
    # SEÇÃO 1: CACHE DE GEOLOCALIZAÇÃO (PRINCIPAL)
    # =========================================================================
    st.subheader("3.1 Registrar Nova Correção Principal (Lat/Lon)")
    st.caption("A chave de busca é a combinação exata de **Endereço + Bairro** da sua planilha original.")
    
    with st.expander("Adicionar Nova Correção Principal"):
        # Container para o formulário
        
        st.subheader("1. Preencher Endereço")
        if 'form_new_endereco' not in st.session_state:
            st.session_state['form_new_endereco'] = ""
            
        new_endereco = st.text_area(
            "1. Endereço COMPLETO no Cache (Chave de Busca)", 
            key="form_new_endereco", 
            height=70,
            help="Cole o endereço exatamente como o Circuit o reconhece (Endereço + Bairro). Ex: 'Rua Principal, 123, Bairro Exato'."
        )
        
        st.markdown("---")
        st.subheader("2. Preencher Coordenadas")
        
        col_input_coord, col_btn_coord = st.columns([3, 1])
        
        with col_input_coord:
            if 'form_colar_coord' not in st.session_state:
                st.session_state['form_colar_coord'] = ""
                
            st.text_input(
                "2. Colar Coordenadas Google (Ex: -23,5139753, -52,1131268)",
                key="form_colar_coord",
                help="Cole o texto de Lat e Lon copiados do Google Maps/Earth."
            )
        with col_btn_coord:
            st.markdown("##") 
            st.button(
                "Aplicar Coordenadas", 
                on_click=apply_google_coords,
                args=('form_new_lat', 'form_new_lon', 'form_colar_coord'),
                key="btn_apply_coord_geo",
            )
        
        st.caption("--- OU preencha ou ajuste manualmente (use PONTO como separador decimal) ---")

        col_lat, col_lon = st.columns(2)
        
        if 'form_new_lat' not in st.session_state:
            st.session_state['form_new_lat'] = ""
        if 'form_new_lon' not in st.session_state:
            st.session_state['form_new_lon'] = ""
            
        with col_lat:
            new_latitude = st.text_input("3. Latitude Corrigida", key="form_new_lat")
        with col_lon:
            new_longitude = st.text_input("4. Longitude Corrigida", key="form_new_lon")
            
        st.markdown("---")
        
        save_button_col, clear_button_col = st.columns(2)
        
        with save_button_col:
            if st.button("✅ Salvar Nova Correção Principal", key="btn_save_quick_geo"):
                
                lat_to_save = st.session_state.get('form_new_lat', '')
                lon_to_save = st.session_state.get('form_new_lon', '')
                
                if not new_endereco or not lat_to_save or not lon_to_save:
                    st.error("Preencha o endereço e as coordenadas (3 e 4) antes de salvar.")
                else:
                    try:
                        endereco_limpo = new_endereco.strip().rstrip(';')
                        lat = float(str(lat_to_save).strip().replace(',', '.'))
                        lon = float(str(lon_to_save).strip().replace(',', '.'))
                        
                        save_single_entry_to_db(conn, endereco_limpo, lat, lon)
                        
                    except ValueError:
                        st.error("Latitude e Longitude devem ser números válidos.")
        
        with clear_button_col:
             st.button("❌ Limpar Formulário", on_click=clear_lat_lon_fields, key="btn_clear_form_geo")

    
    st.markdown(f"#### Visualização do Cache Principal (Total: {len(df_cache_original)})")
    st.dataframe(df_cache_original, use_container_width=True) 
    
    # ----------------------------------------------------------------------------------
    # BACKUP E RESTAURAÇÃO DO CACHE PRINCIPAL
    # ----------------------------------------------------------------------------------
    
    st.markdown("##### Backup/Restauração e Limpeza do Cache Principal")
    col_backup_geo, col_restauracao_geo, col_limpeza_geo = st.columns(3)
    
    with col_backup_geo:
        if not df_cache_original.empty:
            backup_file = export_cache(df_cache_original, CACHE_COLUMNS)
            st.download_button(
                label="⬇️ Baixar Backup Principal",
                data=backup_file,
                file_name="cache_geolocalizacao_principal.xlsx",
                mime=EXCEL_MIME_TYPE, 
                key="download_backup_geo"
            )
        else:
            st.caption("Cache principal vazio.")

    with col_restauracao_geo:
        uploaded_backup_geo = st.file_uploader(
            "Restaurar Principal:", 
            type=['csv', 'xlsx'],
            key="upload_backup_geo",
            label_visibility='collapsed'
        )
        if uploaded_backup_geo is not None:
            if st.button("⬆️ Iniciar Restauração Principal", key="btn_restore_cache_geo"):
                with st.spinner('Restaurando dados do arquivo...'):
                    import_cache_to_db(conn, uploaded_backup_geo)
                    
    with col_limpeza_geo:
        confirm_clear_geo = st.checkbox(
            "Confirmar exclusão (Principal)", 
            key="confirm_clear_cache_geo"
        )
        if confirm_clear_geo:
            if st.button("🔴 Excluir Cache Principal", key="btn_final_clear_cache_geo"):
                clear_geoloc_cache_db(conn)


# ----------------------------------------------------------------------------------
# ABA 4: GERENCIAR CACHE DE ENDEREÇO ALTERNATIVO (NOVA ABA)
# ----------------------------------------------------------------------------------

with tab4:
    st.header("📌 Gerenciamento de Matches de Endereço Alternativo")
    st.info("Utilize esta seção para registrar manualmente um endereço (chave de busca) e o endereço alternativo/ponto de referência, incluindo a geolocalização do ponto de referência.")
    
    # 1. Carrega o cache alternativo salvo
    df_alt_address_original = load_alt_address_cache(conn).fillna("")

    # =========================================================================
    # SEÇÃO 4.1: REGISTRO DE ENDEREÇO ALTERNATIVO
    # =========================================================================
    st.subheader("4.1 Registrar Novo Match Manual")
    st.caption("A chave de busca (Endereço Digitado) será usada na Aba 1 para aplicar o Match/Sugestão abaixo.")
    
    with st.expander("Adicionar Novo Match/Sugestão de Endereço Alternativo"): 
        
        st.subheader("1. Endereço Digitado (Chave de Busca para o Match)")
        if 'form_alt_endereco_principal' not in st.session_state:
            st.session_state['form_alt_endereco_principal'] = ""
        
        alt_endereco_principal = st.text_area(
            "1. Endereço Principal (Chave)",
            key="form_alt_endereco_principal", 
            height=70,
            help="Cole o endereço **exatamente** como o cliente digitou (Endereço + Bairro da sua planilha) para que o sistema encontre o match."
        )

        st.markdown("---")
        st.subheader("2. Sugestão de Endereço Alternativo (o Match) e Coordenadas") 
        
        if 'form_alt_endereco' not in st.session_state:
            st.session_state['form_alt_endereco'] = ""
        
        new_alt_endereco = st.text_area(
            "2. Endereço Alternativo COMPLETO (Sugestão do Match)",
            key="form_alt_endereco", 
            height=70,
            help="O novo endereço (corrigido/alternativo/referência) que será sugerido (Alt_Address)."
        )

        col_alt_input_coord, col_alt_btn_coord = st.columns([3, 1])
        
        with col_alt_input_coord:
            if 'form_alt_colar_coord' not in st.session_state:
                st.session_state['form_alt_colar_coord'] = ""
                
            st.text_input(
                "3. Colar Coordenadas Alternativas (Google Format)",
                key="form_alt_colar_coord",
                help="Coordenadas (Lat/Lon) do endereço alternativo."
            )
        with col_alt_btn_coord:
            st.markdown("##") 
            st.button(
                "Aplicar Coordenadas", 
                on_click=apply_google_coords,
                args=('form_alt_lat', 'form_alt_lon', 'form_alt_colar_coord'),
                key="btn_apply_coord_alt",
            )
        
        st.caption("--- OU preencha ou ajuste manualmente (use PONTO como separador decimal) ---")

        col_alt_lat, col_alt_lon = st.columns(2)
        
        if 'form_alt_lat' not in st.session_state:
            st.session_state['form_alt_lat'] = ""
        if 'form_alt_lon' not in st.session_state:
            st.session_state['form_alt_lon'] = ""
            
        with col_alt_lat:
            alt_latitude = st.text_input("4. Latitude Alternativa", key="form_alt_lat")
        with col_alt_lon:
            alt_longitude = st.text_input("5. Longitude Alternativa", key="form_alt_lon")
            
        st.markdown("---")
        
        save_button_col_alt, clear_button_col_alt = st.columns(2)
        
        with save_button_col_alt:
            if st.button("✅ Salvar Novo Endereço Alternativo", key="btn_save_quick_alt"):
                
                alt_lat_to_save = st.session_state.get('form_alt_lat', '')
                alt_lon_to_save = st.session_state.get('form_alt_lon', '')
                
                if not alt_endereco_principal or not new_alt_endereco or not alt_lat_to_save or not alt_lon_to_save:
                    st.error("Preencha o Endereço Principal (Chave), o Endereço Alternativo e as Coordenadas antes de salvar.")
                else:
                    try:
                        endereco_principal_limpo = alt_endereco_principal.strip().rstrip(';')
                        alt_address_limpo = new_alt_endereco.strip().rstrip(';')
                        alt_lat = float(str(alt_lat_to_save).strip().replace(',', '.'))
                        alt_lon = float(str(alt_lon_to_save).strip().replace(',', '.'))
                        
                        save_single_alt_entry_to_db(conn, endereco_principal_limpo, alt_address_limpo, alt_lat, alt_lon)
                        
                    except ValueError:
                        st.error("Latitude e Longitude devem ser números válidos.")
        
        with clear_button_col_alt:
             st.button("❌ Limpar Formulário Alt", on_click=clear_alt_address_fields, key="btn_clear_form_alt")

    
    st.markdown(f"#### Visualização do Cache Alternativo (Total: {len(df_alt_address_original)})")
    st.dataframe(df_alt_address_original, use_container_width=True) 
    
    # ----------------------------------------------------------------------------------
    # BACKUP E RESTAURAÇÃO DO CACHE ALTERNATIVO
    # ----------------------------------------------------------------------------------
    
    st.markdown("##### Backup/Restauração e Limpeza do Cache Alternativo")
    col_backup_alt, col_restauracao_alt, col_limpeza_alt = st.columns(3)

    with col_backup_alt:
        if not df_alt_address_original.empty:
            backup_file_alt = export_cache(df_alt_address_original, ALT_CACHE_COLUMNS)
            st.download_button(
                label="⬇️ Baixar Backup Alternativo",
                data=backup_file_alt,
                file_name="cache_alt_address_backup.xlsx",
                mime=EXCEL_MIME_TYPE, 
                key="download_backup_alt"
            )
        else:
            st.caption("Cache alternativo vazio.")
            
    with col_restauracao_alt:
        uploaded_backup_alt = st.file_uploader(
            "Restaurar Alternativo:", 
            type=['csv', 'xlsx'],
            key="upload_backup_alt",
            label_visibility='collapsed'
        )
        if uploaded_backup_alt is not None:
            if st.button("⬆️ Iniciar Restauração Alternativa", key="btn_restore_cache_alt"):
                with st.spinner('Restaurando dados do arquivo...'):
                    import_alt_cache_to_db(conn, uploaded_backup_alt)
                    
    with col_limpeza_alt:
        confirm_clear_alt = st.checkbox(
            "Confirmar exclusão (Alternativo)", 
            key="confirm_clear_cache_alt"
        )
        if confirm_clear_alt:
            if st.button("🔴 Excluir Cache Alternativo", key="btn_final_clear_cache_alt"):
                clear_alt_address_cache_db(conn)

