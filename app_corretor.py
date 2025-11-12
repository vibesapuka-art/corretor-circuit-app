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

# Colunas esperadas no arquivo de Pós-Roteirização (Saída do Circuit)
COLUNA_ADDRESS_CIRCUIT = 'address' 
COLUNA_NOTES_CIRCUIT = 'notes'


# --- Configurações de MIME Type ---
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# --- Configurações de Banco de Dados ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
PRIMARY_KEYS = ['Endereco_Completo_Cache'] 


# ===============================================
# FUNÇÕES DE BANCO DE Dados (SQLite)
# (Mantidas do Código Anterior, Omitidas para Brevidade)
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
        Longitude_Corrigida REAL
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
    except pd.io.sql.DatabaseError:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de geolocalização: {e}")
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_single_entry_to_db(conn, endereco, lat, lon):
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
        
def import_cache_to_db(conn, uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df_import = pd.read_csv(uploaded_file)
        else: 
            df_import = pd.read_excel(uploaded_file, sheet_name=0)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return 0

    required_cols = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']
    if not all(col in df_import.columns for col in required_cols):
        st.error(f"Erro de Importação: O arquivo deve conter as colunas exatas: {', '.join(required_cols)}")
        return 0

    df_import = df_import[required_cols].copy()
    df_import['Endereco_Completo_Cache'] = df_import['Endereco_Completo_Cache'].astype(str).str.strip().str.rstrip(';')
    df_import['Latitude_Corrigida'] = df_import['Latitude_Corrigida'].astype(str).str.replace(',', '.', regex=False)
    df_import['Longitude_Corrigida'] = df_import['Longitude_Corrigida'].astype(str).str.replace(',', '.', regex=False)
    df_import['Latitude_Corrigida'] = pd.to_numeric(df_import['Latitude_Corrigida'], errors='coerce')
    df_import['Longitude_Corrigida'] = pd.to_numeric(df_import['Longitude_Corrigida'], errors='coerce')
    df_import = df_import.dropna(subset=['Latitude_Corrigida', 'Longitude_Corrigida'])
    
    if df_import.empty:
        st.warning("Nenhum dado válido de correção (Lat/Lon) foi encontrado no arquivo para importar.")
        return 0
        
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
            load_geoloc_cache.clear()
            count_after = len(load_geoloc_cache(conn))
            st.success(f"Importação de backup concluída! **{insert_count}** entradas processadas. O cache agora tem **{count_after}** entradas.")
            st.rerun() 
            return count_after
    except Exception as e:
        st.error(f"Erro crítico ao inserir dados no cache. Erro: {e}")
        return 0
        
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


# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# (Mantidas do Código Anterior, Omitidas para Brevidade)
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
            return None, [] 

    df = df_entrada.copy()
    corrected_addresses = [] 
    
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
        
        cache_mask = df['Cache_Lat'].notna()
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Cache_Lat']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Cache_Lon']
        corrected_addresses = df.loc[cache_mask, 'Chave_Cache_DB'].unique().tolist()
        
        df = df.drop(columns=['Chave_Busca_Cache', 'Chave_Cache_DB', 'Cache_Lat', 'Cache_Lon'], errors='ignore')
    
    # PASSO 2: FUZZY MATCHING (CORREÇÃO DE ENDEREÇO E AGRUPAMENTO)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching e Agrupamento...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos == 0:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")
        return None, []
    
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
    
    return df_circuit, corrected_addresses 


# ===============================================
# FUNÇÃO DE SPLIT DE ROTAS
# ===============================================

def split_dataframe_for_drivers(df_circuit, num_motoristas):
    """
    Divide o DataFrame (o agrupado da Pré-Roteirização) em N DataFrames,
    distribuindo as paradas de forma mais equitativa possível, MANTENDO A ORDEM.
    """
    if df_circuit is None or df_circuit.empty:
        return {}
    
    # Garante que as colunas essenciais para importação do Circuit estejam presentes
    COLUNAS_EXPORT_SPLIT = ['Address', 'Latitude', 'Longitude', 'Notes']
    df_export = df_circuit[['Sequence_Base'] + COLUNAS_EXPORT_SPLIT].copy()
    
    # O Circuit lê 'Order ID' e 'Notes', mas como estamos na pré, 
    # as colunas Latitude e Longitude são cruciais.
    # Vamos usar 'Notes' como o Order ID/Notes para simplificar.
    df_export.rename(columns={'Notes': 'Notes', 'Address': 'Address'}, inplace=True)
    
    total_paradas = len(df_export)
    
    if num_motoristas <= 0:
        return {} 
    
    paradas_base = total_paradas // num_motoristas
    restante = total_paradas % num_motoristas
    rotas_divididas = {}
    start_index = 0
    
    for i in range(num_motoristas):
        # O primeiro 'restante' de motoristas recebe uma parada a mais
        paradas_motorista = paradas_base + (1 if i < restante else 0)
        
        end_index = start_index + paradas_motorista
        
        df_motorista = df_export.iloc[start_index:end_index].copy()
        
        # O nome da coluna 'Order ID' é opcional, mas vamos mantê-lo para o Circuit
        # Usaremos 'Notes' para preencher Order ID, já que ele contém os IDs agrupados
        df_motorista.insert(1, 'Order ID', df_motorista['Notes'].apply(lambda x: str(x).split(';')[0].strip()))
        
        # Remove a coluna 'Sequence_Base' antes de exportar
        df_motorista = df_motorista.drop(columns=['Sequence_Base'])
        
        # Colunas finais para exportação
        df_motorista = df_motorista[['Order ID', 'Address', 'Latitude', 'Longitude', 'Notes']]
        
        # Nome da Rota (com contagem de paradas)
        rotas_divididas[f"Motorista {i+1} ({len(df_motorista)} Paradas)"] = df_motorista
        
        start_index = end_index
        
    return rotas_divididas


# ===============================================
# FUNÇÕES DE PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
# (Mantidas do Código Anterior, Omitidas para Brevidade)
# ===============================================

def is_not_purely_volumous(ids_string):
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
    
    # Tenta normalizar as colunas (se estiverem em maiúsculas/minúsculas diferentes)
    df_input.columns = df_input.columns.str.strip().str.lower()
    
    if COLUNA_NOTES_CIRCUIT not in df_input.columns or COLUNA_ADDRESS_CIRCUIT not in df_input.columns:
        # A coluna 'order id' também pode ser usada se 'notes' não estiver presente
        if 'order id' not in df_input.columns:
            raise KeyError(f"As colunas de endereço ('{COLUNA_ADDRESS_CIRCUIT}') e notas/id ('{COLUNA_NOTES_CIRCUIT}' ou 'order id') não foram encontradas.") 
        
    df = df_input.copy()
    
    # Se 'notes' estiver faltando, mas 'order id' existir (caso de importação/exportação padrão)
    if COLUNA_NOTES_CIRCUIT not in df.columns and 'order id' in df.columns:
        df[COLUNA_NOTES_CIRCUIT] = df['order id'].astype(str)
        
    df[COLUNA_NOTES_CIRCUIT] = df[COLUNA_NOTES_CIRCUIT].astype(str)
    df = df.dropna(subset=[COLUNA_NOTES_CIRCUIT]) 
    
    df[COLUNA_NOTES_CIRCUIT] = df[COLUNA_NOTES_CIRCUIT].str.strip('"')
    
    # 1. Separa o campo 'Notes' pelo PONTO E VÍRGULA
    # Ex: '1,2,3*; Pacotes: 3 | Cidade: Curitiba | CEP: 80000000'
    df_split = df[COLUNA_NOTES_CIRCUIT].str.split(';', n=1, expand=True)
    df['Ordem ID'] = df_split[0].str.strip() 
    
    # O segundo item é o restante da anotação
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    # 2. TRATAMENTO CRÍTICO (ISOLAMENTO DO ID PELO HÍFEN) - NÃO É MAIS NECESSÁRIO AQUI, 
    # POIS O ID JÁ ESTÁ LIMPO NO CAMPO 'Ordem ID'
    df['ID_Pacote_Limpo'] = df['Ordem ID'].str.strip() 
    
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    # Adiciona a coluna de endereço para visualização na lista de impressão
    df['Address_Clean'] = df[COLUNA_ADDRESS_CIRCUIT].astype(str)
    
    coluna_filtro = 'ID_Pacote_Limpo' 
    
    # DataFrame FINAL GERAL
    df_final_geral = df[['Lista de Impressão', 'Address_Clean']].copy() 
    
    # FILTRAR VOLUMOSOS 
    df_volumosos = df[df[coluna_filtro].str.contains(r'\*', regex=True, na=False)].copy()
    df_volumosos_impressao = df_volumosos[['Lista de Impressão', 'Address_Clean']].copy() 
    
    # FILTRAR NÃO-VOLUMOSOS
    df_nao_volumosos = df[
        df[coluna_filtro].apply(is_not_purely_volumous)
    ].copy() 
    
    df_nao_volumosos_impressao = df_nao_volumosos[['Lista de Impressão', 'Address_Clean']].copy()
    
    # Retorna o DF original *limpo* (com colunas normalizadas) para o Split, se necessário
    # NOTA: df_limpo_para_split_pos não é mais usado no split, mas mantido por segurança.
    df_limpo_para_split_pos = df[[COLUNA_ADDRESS_CIRCUIT, COLUNA_NOTES_CIRCUIT]].copy()
    df_limpo_para_split_pos.columns = ['Address', 'Notes'] # Normaliza os nomes para uso no Split
    
    return df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao, df_limpo_para_split_pos


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

# 1. Conexão com o Banco de Dados (Executada uma vez)
conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("🗺️ Flow Completo Circuit (Pré, Pós e Cache)")

# CRIAÇÃO DAS ABAS 
tab1, tab_split, tab2, tab3 = st.tabs(["🚀 Pré-Roteirização (Importação)", "✂️ Split Route (Dividir)", "📋 Pós-Roteirização (Impressão/Cópia)", "💾 Gerenciar Cache de Geolocalização"])


# ----------------------------------------------------------------------------------
# VARIÁVEIS DE ESTADO (SESSION STATE)
# ----------------------------------------------------------------------------------

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'volumoso_ids' not in st.session_state:
    st.session_state['volumoso_ids'] = set() 
# Este é o DF agrupado e com coordenadas, pronto para o Circuit.
if 'df_circuit_agrupado_pre' not in st.session_state: 
    st.session_state['df_circuit_agrupado_pre'] = None


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
            
            # Limpa o estado se for um novo arquivo
            if st.session_state.get('last_uploaded_name') != uploaded_file_pre.name:
                 st.session_state['volumoso_ids'] = set()
                 st.session_state['last_uploaded_name'] = uploaded_file_pre.name
                 st.session_state['df_circuit_agrupado_pre'] = None


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
            with st.spinner('Aplicando cache 100% match e processando dados...'):
                 try:
                     result = processar_e_corrigir_dados(df_para_processar, limite_similaridade_ajustado, df_cache)
                 except Exception as e:
                     st.error(f"Erro Crítico durante a correção e agrupamento: {e}")
                     result = None 
                 
                 if isinstance(result, (list, tuple)) and len(result) == 2:
                     df_circuit, corrected_addresses = result
                 else:
                     df_circuit = None
                     corrected_addresses = []
            
            if df_circuit is not None:
                st.session_state['df_circuit_agrupado_pre'] = df_circuit
                
                st.markdown("---")
                st.header("✅ Resultado Concluído!")
                
                if corrected_addresses:
                    st.success(f"Cache de Geolocalização Aplicado! **{len(corrected_addresses)}** endereços únicos foram corrigidos (100% Match):")
                    corrected_text = '\n'.join([f"- {addr}" for addr in corrected_addresses])
                    with st.expander("Clique para ver a lista completa de endereços corrigidos pelo cache"):
                         st.markdown(corrected_text)
                else:
                    st.info("Nenhuma correção de geolocalização foi aplicada pelo cache nesta planilha (100% Match).")
                
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
                st.dataframe(df_circuit.drop(columns=['Sequence_Base']), use_container_width=True) # Remove Sequence_Base para visualização
                
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                    # Remove Sequence_Base para a importação final, pois o Circuit não precisa dela
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
# ABA 1.5: SPLIT ROUTE (DIVIDIR ROTAS) - AGORA PRÉ-ROTEIRIZAÇÃO
# ----------------------------------------------------------------------------------

with tab_split:
    st.header("✂️ Dividir Rota PRÉ-Roteirização (Com Coordenadas)")
    st.caption("A divisão será feita no arquivo agrupado da Pré-Roteirização. Cada motorista receberá sua parte com Lat/Lon para otimizar *individualmente* no Circuit.")
    
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
        
        if st.button(f"➡️ Dividir em {num_motoristas} Rotas Sequenciais para Motoristas", key="btn_split_route_pre"):
            
            rotas_divididas = split_dataframe_for_drivers(df_rota_para_split, num_motoristas)
            
            st.markdown("---")
            st.header("✅ Resultado da Divisão")
            st.success("O arquivo agrupado foi dividido equitativamente. Cada Motorista deve importar **sua aba** no Circuit para otimizar a rota.")
            
            # Prepara o arquivo Excel com todas as abas
            buffer_split = io.BytesIO()
            with pd.ExcelWriter(buffer_split, engine='openpyxl') as writer:
                
                for nome_rota, df_rota in rotas_divididas.items():
                    # Garante um nome de aba válido
                    sheet_name = nome_rota.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")[:31]
                    
                    # O df_rota já contém 'Order ID', 'Address', 'Latitude', 'Longitude', 'Notes'
                    df_rota.to_excel(writer, index=False, sheet_name=sheet_name)
                    
                    st.subheader(f"Rota para {nome_rota}")
                    # Mostra as colunas principais para visualização
                    st.dataframe(df_rota, use_container_width=True)
                    
            buffer_split.seek(0)

            # Botão de Download
            st.download_button(
                label=f"📥 Baixar Arquivo de Rotas Divididas ({num_motoristas} Abas)",
                data=buffer_split,
                file_name=f"Circuit_Rotas_Split_PRE_OTIMIZACAO_{num_motoristas}_DRIVERS.xlsx",
                mime=EXCEL_MIME_TYPE, 
                key="download_split_routes_pre"
            )
            
            st.info("Cada aba (Motorista 1, Motorista 2, etc.) deve ser importada individualmente no Circuit para otimização.")

    else:
        st.warning("⚠️ **Etapa Pendente:** Por favor, vá para a aba **🚀 Pré-Roteirização** e clique em '🚀 Iniciar Corretor e Agrupamento' primeiro. O arquivo agrupado será carregado aqui automaticamente.")


# ----------------------------------------------------------------------------------
# ABA 2: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO E SEPARAÇÃO DE VOLUMOSOS)
# (Mantido o fluxo de carregamento de arquivo, pois o input é a SAÍDA DO CIRCUIT)
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
            
            # CHAMA A FUNÇÃO DE PROCESSAMENTO (AGORA RETORNA 4 OBJETOS)
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
# (Mantido como estava)
# ----------------------------------------------------------------------------------

def clear_lat_lon_fields():
    if 'form_new_lat_num' in st.session_state:
        st.session_state['form_new_lat_num'] = 0.0 
    if 'form_new_lon_num' in st.session_state:
        st.session_state['form_new_lon_num'] = 0.0 
    if 'form_colar_coord' in st.session_state:
        st.session_state['form_colar_coord'] = ""
    if 'form_new_endereco' in st.session_state:
        st.session_state['form_new_endereco'] = ""


def apply_google_coords():
    coord_string = st.session_state.get('form_colar_coord', '')
    if not coord_string:
        st.error("Nenhuma coordenada foi colada. Cole o texto do Google Maps, ex: -23,5139753, -52,1131268")
        return

    coord_string_clean = coord_string.strip()
    
    try:
        matches = re.findall(r'(-?\d+[\.,]\d+)', coord_string_clean.replace(' ', ''))
        
        if len(matches) >= 2:
            lat = float(matches[0].replace(',', '.'))
            lon = float(matches[1].replace(',', '.'))
            
            st.session_state['form_new_lat_num'] = lat
            st.session_state['form_new_lon_num'] = lon
            st.success(f"Coordenadas aplicadas: Lat: **{lat}**, Lon: **{lon}**")
            return
            
    except ValueError:
        parts = coord_string_clean.split(',')
        if len(parts) >= 2:
             try:
                lat = float(parts[0].replace(',', '.').strip()) 
                lon = float(parts[1].replace(',', '.').strip())
                
                st.session_state['form_new_lat_num'] = lat
                st.session_state['form_new_lon_num'] = lon
                st.success(f"Coordenadas aplicadas: Lat: **{lat}**, Lon: **{lon}**")
                return
             except ValueError:
                pass 
                
    st.error(f"Não foi possível extrair duas coordenadas válidas da string: '{coord_string}'. Verifique o formato. Exemplo: -23.5139753, -52.1131268")


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
                "2. Colar Coordenadas Google (Ex: -23,5139753, -52,1131268)",
                key="form_colar_coord",
                help="Cole o texto de Lat e Lon copiados do Google Maps/Earth."
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
                
                if not new_endereco or (lat_to_save == 0.0 and lon_to_save == 0.0 and st.session_state.get('form_colar_coord') == ""):
                    st.error("Preencha o endereço e as coordenadas (3 e 4) antes de salvar, ou use a ferramenta 'Aplicar Coordenadas'.")
                else:
                    try:
                        endereco_limpo = new_endereco.strip().rstrip(';')
                        save_single_entry_to_db(conn, endereco_limpo, lat_to_save, lon_to_save)
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
        
        def export_cache(df_cache):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer: 
                df_cache[CACHE_COLUMNS].to_excel(writer, index=False, sheet_name='Cache_Geolocalizacao')
            buffer.seek(0)
            return buffer
            
        if not df_cache_original.empty:
            backup_file = export_cache(df_cache_original)
            st.download_button(
                label="⬇️ Baixar Backup do Cache (.xlsx)",
                data=backup_file,
                file_name="cache_geolocalizacao_backup.xlsx",
                mime=EXCEL_MIME_TYPE, 
                key="download_backup"
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
                with st.spinner('Restaurando dados do arquivo...'):
                    import_cache_to_db(conn, uploaded_backup)
                    
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
