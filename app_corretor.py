# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 # Adicionado para conexão com banco de dados nativo
# IMPORT AGGRID para permitir a edição da geolocalização
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configurações Iniciais da Página ---
st.set_page_page_config(
    page_title="Circuit Flow Completo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS para garantir alinhamento à esquerda em TEXT AREAS e Checkboxes ---
st.markdown("""
<style>
/* Alinha o texto de entrada na caixa de texto (útil para formulários) */
.stTextArea [data-baseweb="base-input"] {
    text-align: left;
    font-family: monospace;
}
/* *** CSS FORTE: Garante que o conteúdo e o título do st.text_area sejam alinhados à esquerda *** */
div.stTextArea > label {
    text-align: left !important; /* Título do text area */
}
/* Força o alinhamento à esquerda no campo de texto principal */
div[data-testid="stTextarea"] textarea {
    text-align: left !important; /* Conteúdo do text area */
    font-family: monospace;
    white-space: pre-wrap; /* Garante quebras de linha corretas */
}
/* Alinha os títulos e outros elementos em geral */
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}
/* Força a quebra de linha em cabeçalhos de tabela do AgGrid */
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

# --- Configurações de Banco de Dados ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc"
CACHE_COLUMNS = ['Endereco_Original_Cliente', 'Latitude_Corrigida', 'Longitude_Corrigida']


# ===============================================
# FUNÇÕES DE BANCO DE DADOS (SQLite)
# ===============================================

@st.cache_resource
def get_db_connection():
    """
    Cria e retorna a conexão com o banco de dados SQLite.
    CORRIGIDO: Usa st.experimental_connection com o driver sqlite3.
    """
    conn = st.experimental_connection(
        DB_NAME, 
        type="sql", 
        driver="sqlite3" 
    )
    return conn

def create_table_if_not_exists(conn):
    """Cria a tabela de cache de geolocalização se ela não existir."""
    query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        Endereco_Original_Cliente TEXT PRIMARY KEY,
        Latitude_Corrigida REAL,
        Longitude_Corrigida REAL
    );
    """
    try:
        conn.execute(query)
    except Exception as e:
         # Captura e exibe qualquer erro de execução do SQL
        st.error(f"Erro ao criar tabela: {e}")


def load_geoloc_cache(conn):
    """Carrega todo o cache de geolocalização para um DataFrame."""
    try:
        df_cache = conn.query(f"SELECT * FROM {TABLE_NAME}")
        # Garante que as colunas de geoloc sejam numéricas
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except Exception as e:
        # Se a tabela não existir ainda ou der erro, retorna um DataFrame vazio
        st.info("Cache de geolocalização não encontrado ou vazio. Será criado após a primeira correção.")
        return pd.DataFrame(columns=CACHE_COLUMNS)

def save_corrections_to_db(conn, df_correcoes):
    """Salva um DataFrame de correções no banco de dados, usando UPSERT."""
    # A coluna de endereço deve ser a chave primária
    df_correcoes = df_correcoes[CACHE_COLUMNS].copy()
    
    # Remover NaNs da Latitude/Longitude para evitar erros de tipo no DB
    df_correcoes = df_correcoes.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])
    
    data_tuples = [tuple(row) for row in df_correcoes.values]
    
    # SQLite UPSERT (ON CONFLICT REPLACE)
    upsert_query = f"""
    INSERT OR REPLACE INTO {TABLE_NAME} 
    (Endereco_Original_Cliente, Latitude_Corrigida, Longitude_Corrigida) 
    VALUES (?, ?, ?);
    """
    
    try:
        with conn.session as session:
            for row_data in data_tuples:
                session.execute(upsert_query, row_data)
            session.commit()
        
        st.success(f"Cache de geolocalização atualizado! Foram salvos **{len(data_tuples)}** registros únicos.")
        
        # Invalida o cache do Streamlit para o novo load
        load_geoloc_cache.clear() 
    except Exception as e:
        st.error(f"Erro ao salvar as correções no banco de dados: {e}")

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
    Função principal que aplica a correção, o agrupamento e o lookup no cache.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None, None 

    df = df_entrada.copy()
    
    # Preparação
    df['Bairro'] = df['Bairro'].astype(str).replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    df['Original_Address_For_Cache'] = df[COLUNA_ENDERECO].astype(str) # Mantém a string original

    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)

    
    # =========================================================================
    # PASSO 1: APLICAR LOOKUP NO CACHE DE GEOLOCALIZAÇÃO
    # =========================================================================
    
    if not df_cache_geoloc.empty:
        # Merge do DataFrame principal com o cache
        df = pd.merge(
            df, 
            df_cache_geoloc, 
            left_on='Original_Address_For_Cache', 
            right_on='Endereco_Original_Cliente', 
            how='left'
        )
        
        # Atualiza Latitude e Longitude se a correção existir no cache
        cache_mask = df['Latitude_Corrigida'].notna()
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Latitude_Corrigida']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Longitude_Corrigida']
        
        st.info(f"Cache aplicado! {cache_mask.sum()} registros de geolocalização foram corrigidos automaticamente.")

        # Remove colunas auxiliares
        df = df.drop(columns=['Endereco_Original_Cliente', 'Latitude_Corrigida', 'Longitude_Corrigida'], errors='ignore')
    
    # =========================================================================
    # PASSO 2: FUZZY MATCHING (CORREÇÃO DE ENDEREÇO E AGRUPAMENTO)
    # =========================================================================
    
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # Fuzzy Matching
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos == 0:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")
        return None, None
    
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

    # Agrupamento (Chave: Endereço Corrigido + Cidade)
    colunas_agrupamento = ['Endereco_Corrigido', 'City'] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        # Lista de todos os Endereços Originais do Cliente que foram agrupados
        Enderecos_Originais=(COLUNA_ENDERECO, lambda x: ', '.join(x.astype(str).unique())),
        
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'\*', '', str(y))) if re.sub(r'\*', '', str(y)).isdigit() else float('inf'))))), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        # Mantém a geoloc (que pode ter sido corrigida pelo cache)
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        # Dados de Suporte
        Bairro_Agrupado=('Bairro', get_most_common_or_empty),
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
    
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode_Agrupado']
    )

    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'],
