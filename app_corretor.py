# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
# IMPORT AGGRID para permitir a edição da geolocalização
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configurações Iniciais da Página ---
# Usando st.set_page_config
st.set_page_config(
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
            
            # CORRIGIDO: Certificando-se de que a linha não contém erros de colchetes abertos.
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

    # CORREÇÃO: Chave '}' faltante e colchetes verificados.
    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    }) 
    
    # Retorna o DF do Circuit E o DF Agrupado para revisão de Geoloc
    return df_circuit, df_agrupado[['Enderecos_Originais', 'Latitude', 'Longitude']].copy()


# ===============================================
# FUNÇÕES DE PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
# ===============================================

def processar_rota_para_impressao(df_input):
    """
    Processa o DataFrame da rota, extrai 'Ordem ID' da coluna 'Notes' e prepara para cópia.
    """
    coluna_notes_lower = 'notes'
    
    if coluna_notes_lower not in df_input.columns:
        raise KeyError(f"A coluna '{coluna_notes_lower}' não foi encontrada.")
    
    df = df_input.copy()
    df[coluna_notes_lower] = df[coluna_notes_lower].astype(str)
    df = df.dropna(subset=[coluna_notes_lower]) 
    
    df[coluna_notes_lower] = df[coluna_notes_lower].str.strip('"')
    
    df_split = df[coluna_notes_lower].str.split(';', n=1, expand=True)
    df['Ordem ID'] = df_split[0].str.strip()
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    # DataFrame FINAL GERAL
    df_final_geral = df[['Lista de Impressão', 'address']].copy() 
    
    # FILTRAR VOLUMOSOS
    df_volumosos = df[df['Ordem ID'].str.contains(r'\*', regex=True)].copy()
    df_volumosos_impressao = df_volumosos[['Lista de Impressão', 'address']].copy() 
    
    # FILTRAR NÃO-VOLUMOSOS
    df_nao_volumosos = df[~df['Ordem ID'].str.contains(r'\*', regex=True)].copy() 
    df_nao_volumosos_impressao = df_nao_volumosos[['Lista de Impressão', 'address']].copy()
    
    return df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

# 1. Conexão com o Banco de Dados (Executada uma vez)
conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("🗺️ Flow Completo Circuit (Pré e Pós-Roteirização)")

# CRIAÇÃO DAS ABAS 
tab1, tab2 = st.tabs(["🚀 Pré-Roteirização (Importação)", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa aplica o **Cache de Geolocalização** e agrupa os endereços.")

    # Inicializa o estado
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'volumoso_ids' not in st.session_state:
        st.session_state['volumoso_ids'] = set() 
    if 'df_agrupado_para_edicao' not in st.session_state:
        st.session_state['df_agrupado_para_edicao'] = None

    
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
            colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
            for col in colunas_essenciais:
                 if col not in df_input_pre.columns:
                     raise KeyError(f"A coluna '{col}' está faltando na sua planilha.")
            
            # Resetar as marcações se um novo arquivo for carregado
            if st.session_state.get('last_uploaded_name') != uploaded_file_pre.name:
                 st.session_state['volumoso_ids'] = set()
                 st.session_state['last_uploaded_name'] = uploaded_file_pre.name
                 st.session_state['df_agrupado_para_edicao'] = None


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
        df_temp['Order_Num'] = pd.to_numeric(df_temp[COLUNA_SEQUENCE], errors='coerce').fillna(float('inf'))
        ordens_originais_sorted = df_temp.sort_values('Order_Num')[COLUNA_SEQUENCE].astype(str).unique()
        
        def update_volumoso_ids(order_id, is_checked):
            if is_checked:
                st.session_state['volumoso_ids'].add(order_id)
            elif order_id in st.session_state['volumoso_ids']:
                st.session_state['volumoso_ids'].remove(order_id)

        st.caption("Marque os números das ordens de serviço que são volumosas (serão marcadas com *):")

        # Container para os checkboxes
        with st.container(height=300):
            cols = st.columns(5)
            col_index = 0
            for order_id in ordens_originais_sorted:
                with cols[col_index % 5]:
                    is_checked = order_id in st.session_state['volumoso_ids']
                    st.checkbox(
                        str(order_id), 
                        value=is_checked, 
                        key=f"vol_{order_id}",
                        on_change=update_volumoso_ids, 
                        args=(order_id, not is_checked) 
                    )
                col_index += 1


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
            
            # 1. Aplicar a marcação * no DF antes de processar
            df_para_processar = st.session_state['df_original'].copy()
            df_para_processar[COLUNA_SEQUENCE] = df_para_processar[COLUNA_SEQUENCE].astype(str)
            
            for id_volumoso in st.session_state['volumoso_ids']:
                str_id_volumoso = str(id_volumoso)
                df_para_processar.loc[
                    df_para_processar[COLUNA_SEQUENCE] == str_id_volumoso, 
                    COLUNA_SEQUENCE
                ] = str_id_volumoso + '*'

            # 2. Carregar Cache de Geolocalização
            df_cache = load_geoloc_cache(conn)

            # 3. Iniciar o processamento e agrupamento
            with st.spinner('Processando dados, aplicando cache e agrupando...'):
                 df_
