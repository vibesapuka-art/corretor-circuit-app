# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
import math
# Importação de st_aggrid (mantido para compatibilidade, mas sem uso prático nas abas)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode


# --- Configurações Iniciais da Página ---
st.set_page_page_config(
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

# --- Configurações de Banco de Dados ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
# Estrutura do Cache (Endereço Completo + Lat/Lon)
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
PRIMARY_KEYS = ['Endereco_Completo_Cache'] 


# ===============================================
# FUNÇÕES DE BANCO DE Dados (SQLite)
# ===============================================

@st.cache_resource
def get_db_connection():
    """
    Cria e retorna a conexão com o banco de dados SQLite.
    """
    conn = sqlite3.connect(DB_NAME, check_same_thread=False, timeout=10)
    return conn

def create_table_if_not_exists(conn):
    """Cria a tabela de cache de geolocalização se ela não existir."""
    # PRIMARY KEY composta pelo Endereço Completo
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
        conn.commit()
    except Exception as e:
        st.error(f"Erro ao criar tabela: {e}")


# CORREÇÃO CRÍTICA (UnhashableParamError)
@st.cache_data(hash_funcs={sqlite3.Connection: lambda _: "constant_db_hash"})
def load_geoloc_cache(conn):
    """Carrega todo o cache de geolocalização para um DataFrame."""
    try:
        # Tenta carregar a nova tabela
        df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except pd.io.sql.DatabaseError:
        # Retorna DataFrame vazio com as colunas corretas
        return pd.DataFrame(columns=CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de geolocalização: {e}")
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_single_entry_to_db(conn, endereco, lat, lon):
    """Salva uma única entrada (Endereço Completo + Lat/Lon) no cache (UPSERT)."""
    
    # Query de UPSERT com a nova estrutura
    upsert_query = f"""
    INSERT OR REPLACE INTO {TABLE_NAME} 
    (Endereco_Completo_Cache, Latitude_Corrigida, Longitude_Corrigida) 
    VALUES (?, ?, ?);
    """
    
    try:
        conn.execute(upsert_query, (endereco, lat, lon))
        conn.commit()
        st.success(f"Correção salva para: **{endereco}**.")
        
        # Limpa o cache do Streamlit para forçar o recarregamento na próxima vez
        load_geoloc_cache.clear() 
        # Rerun para atualizar a tabela na tela imediatamente
        st.rerun() 
        
    except Exception as e:
        st.error(f"Erro ao salvar a correção no banco de dados: {e}")
        
        
def import_cache_to_db(conn, uploaded_file):
    """Importa o conteúdo de um arquivo (Excel/CSV) para o cache do banco de dados (UPSERT)."""
    
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
                
                # Usa a lógica de UPSERT (INSERT OR REPLACE)
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
            
            # Força o recarregamento da tabela na tela
            st.rerun() 
            
            return count_after

    except Exception as e:
        st.error(f"Erro crítico ao inserir dados no cache. Verifique se o arquivo está correto. Erro: {e}")
        return 0
        
# ------------------------------------------------------------------
# FUNÇÃO PARA LIMPAR TODO O CACHE (EXCLUSÃO)
# ------------------------------------------------------------------
def clear_geoloc_cache_db(conn):
    """Exclui todos os dados da tabela de cache de geolocalização."""
    
    # Exclui todos os registros da tabela
    query = f"DELETE FROM {TABLE_NAME};"
    
    try:
        conn.execute(query)
        conn.commit()
        
        # Limpa o cache de dados do Streamlit e força recarregamento
        load_geoloc_cache.clear()
        st.success("✅ **Sucesso!** Todos os dados do cache de geolocalização foram excluídos permanentemente.")
        st.rerun() 
        
    except Exception as e:
        st.error(f"❌ Erro ao limpar o cache: {e}")


# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# ===============================================

def limpar_endereco(endereco):
    """
    Normaliza o texto do endereço para melhor comparação, mantendo números e vírgulas.
    """
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    endereco = re.sub(r'\s+', ' ', endereco)
    
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return endereco

def get_most_common_or_empty(x):
    """
    Retorna o valor mais comum de uma Série Pandas.
    """
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    return x_limpo.mode().iloc[0]


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade, df_cache_geoloc):
    """
    Função principal que aplica a correção (usando cache 100% match) e o agrupamento.
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
    
    
    # CHAVE DE BUSCA DE CACHE (Lógica Solicitada)
    # Combina o Endereço (da planilha) + Bairro (da planilha) para criar a chave de busca
    df['Chave_Busca_Cache'] = (
        df[COLUNA_ENDERECO].astype(str).str.strip() + 
        ', ' + 
        df[COLUNA_BAIRRO].astype(str).str.strip()
    )
    # Limpeza final da chave de busca (remove vírgulas extras se o bairro for vazio)
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*$', '', regex=True)
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*,', ',', regex=True)

    
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)

    
    # =========================================================================
    # PASSO 1: APLICAR LOOKUP NO CACHE DE GEOLOCALIZAÇÃO (100% MATCH)
    # =========================================================================
    
    if not df_cache_geoloc.empty:
        # Renomeia colunas do cache para evitar conflitos no merge
        df_cache_lookup = df_cache_geoloc.rename(columns={
            'Endereco_Completo_Cache': 'Chave_Cache_DB', 
            'Latitude_Corrigida': 'Cache_Lat',
            'Longitude_Corrigida': 'Cache_Lon'
        })
        
        # Merge do DataFrame principal com o cache usando a Chave de Busca Combinada
        df = pd.merge(
            df, 
            df_cache_lookup, 
            left_on='Chave_Busca_Cache', # Chave combinada da planilha
            right_on='Chave_Cache_DB',   # Endereço completo do cache
            how='left'
        )
        
        # Atualiza Latitude e Longitude SOMENTE se a correção existir no cache
        cache_mask = df['Cache_Lat'].notna()
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Cache_Lat']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Cache_Lon']
        
        st.info(f"Cache aplicado com 100% de match (Endereço + Bairro)! **{cache_mask.sum()}** registros de geolocalização foram corrigidos automaticamente pelas suas edições no cache.")

        # Remove colunas auxiliares
        df = df.drop(columns=['Chave_Busca_Cache', 'Chave_Cache_DB', 'Cache_Lat', 'Cache_Lon'], errors='ignore')
    
    # =========================================================================
    # PASSO 2: FUZZY MATCHING (CORREÇÃO DE ENDEREÇO E AGRUPAMENTO)
    # =========================================================================
    
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching e Agrupamento...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos == 0:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")
        return None
    
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
        
        # Dados de Suporte
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
    # Remove vírgulas duplicadas ou vírgulas seguidas de espaço (se o bairro for vazio)
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*,', ',', regex=True)
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*$', '', regex=True) 
    
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode_Agrupado']
    )

    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df
