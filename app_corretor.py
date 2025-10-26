# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 # Importação padrão do SQLite
# IMPORT AGGRID para permitir a edição da geolocalização
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# --- Configurações Iniciais da Página ---
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
    Cria e retorna a conexão com o banco de dados SQLite usando sqlite3.connect.
    CORREÇÃO: Usando sqlite3 nativo para evitar o erro de st.experimental_connection.
    """
    # check_same_thread=False é necessário para rodar no Streamlit
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
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
        # Usa o método execute() da conexão sqlite3
        conn.execute(query)
        conn.commit()
    except Exception as e:
         # Captura e exibe qualquer erro de execução do SQL
        st.error(f"Erro ao criar tabela: {e}")


def load_geoloc_cache(conn):
    """Carrega todo o cache de geolocalização para um DataFrame."""
    try:
        # Usa pd.read_sql_query com a conexão sqlite3
        df_cache = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        # Garante que as colunas de geoloc sejam numéricas
        df_cache['Latitude_Corrigida'] = pd.to_numeric(df_cache['Latitude_Corrigida'], errors='coerce')
        df_cache['Longitude_Corrigida'] = pd.to_numeric(df_cache['Longitude_Corrigida'], errors='coerce')
        return df_cache
    except pd.io.sql.DatabaseError:
        # Se a tabela não existir ainda ou der erro de DB, retorna um DataFrame vazio
        st.info("Cache de geolocalização não encontrado ou vazio. Será criado após a primeira correção.")
        return pd.DataFrame(columns=CACHE_COLUMNS)
    except Exception as e:
        st.error(f"Erro ao carregar cache de geolocalização: {e}")
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
        # Usa executemany e commit padrão do sqlite3
        conn.executemany(upsert_query, data_tuples)
        conn.commit()
        
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
    # CHAVE DE CACHE: Mantém a string ORIGINAL para o lookup
    df['Original_Address_For_Cache'] = df[COLUNA_ENDERECO].astype(str) 

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
        # ESSENCIAL: Lista de todos os Endereços Originais do Cliente que foram agrupados
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
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    }) 
    
    # Retorna o DF do Circuit E o DF Agrupado para revisão de Geoloc
    # df_agrupado_para_edicao usará Enderecos_Originais como chave para o cache
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
            # Detecta o tipo de arquivo e carrega
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
                 df_circuit, df_agrupado_edicao = processar_e_corrigir_dados(df_para_processar, limite_similaridade_ajustado, df_cache)
            
            if df_circuit is not None:
                st.session_state['df_agrupado_para_edicao'] = df_agrupado_edicao # Salva para edição posterior
                
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
                st.dataframe(df_circuit, use_container_width=True)
                
                # Download Circuit 
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_circuit"
                )
                
    # =========================================================================
    # 1.4 EDIÇÃO MANUAL E SALVAMENTO DE GEOLOCALIZAÇÃO
    # =========================================================================
    st.markdown("---")
    st.header("1.4 Edição Manual da Geolocalização (Salvar no Cache)")

    if st.session_state['df_agrupado_para_edicao'] is not None and not st.session_state['df_agrupado_para_edicao'].empty:
        
        df_edicao = st.session_state['df_agrupado_para_edicao'].copy()
        # Colunas claras para o usuário
        df_edicao.columns = ['Endereços Originais do Cliente', 'Latitude', 'Longitude']

        st.caption("Edite as colunas **Latitude** e **Longitude** manualmente. A correção será aplicada a **todos os Endereços Originais** listados na primeira coluna e salva no cache para uso futuro.")
        st.warning("⚠️ **Atenção:** As coordenadas salvas substituirão as antigas para os endereços originais listados.")

        # --- Configuração AgGrid ---
        gb = GridOptionsBuilder.from_dataframe(df_edicao)
        # Primeira coluna não é editável, mas seu conteúdo é a chave para o cache
        gb.configure_column('Endereços Originais do Cliente', editable=False, wrapText=True, autoHeight=True) 
        gb.configure_columns(['Latitude', 'Longitude'], type=["numericColumn", "customNumericFormat"], precision=6, editable=True)
        gb.configure_grid_options(domLayout='normal', enableCellTextSelection=True, rowHeight=60)
        gridOptions = gb.build()
        
        # Exibir AgGrid
        grid_response = AgGrid(
            df_edicao,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=False,
            allow_unsafe_jscode=True, 
            enable_enterprise_modules=False,
            height=350,
            width='100%',
            reload_data=True,
            key='geoloc_editor'
        )
        
        df_edited = grid_response['data']

        if st.button("💾 Salvar Correções de Geolocalização no Cache", key="btn_save_cache"):
            
            correcoes_dict = {}
            for index, row in df_edited.iterrows():
                # Lista de endereços originais do cliente separados por ', '
                enderecos_originais = row['Endereços Originais do Cliente'].split(', ')
                
                # Garante que Latitude e Longitude são numéricas, se o usuário digitou texto
                lat = pd.to_numeric(row['Latitude'], errors='coerce')
                lon = pd.to_numeric(row['Longitude'], errors='coerce')
                
                if pd.isna(lat) or pd.isna(lon):
                    st.error(f"Erro: Latitude ou Longitude inválida para o grupo de endereços: {row['Endereços Originais do Cliente']}. Não será salva.")
                    continue
                
                # Aplica a mesma correção (lat/lon) a cada endereço original
                for original_addr in enderecos_originais:
                    correcoes_dict[original_addr.strip()] = (lat, lon)
            
            # Criar o DF para Inserção no Cache
            df_final_cache = pd.DataFrame([
                [addr, lat, lon] 
                for addr, (lat, lon) in correcoes_dict.items()
            ], columns=CACHE_COLUMNS)

            save_corrections_to_db(conn, df_final_cache)
            st.session_state['df_agrupado_para_edicao'] = None # Limpa a tela de edição

    elif st.session_state['df_agrupado_para_edicao'] is not None and st.session_state['df_agrupado_para_edicao'].empty:
         st.info("Nenhum dado para edição. Carregue e processe a planilha primeiro.")

    # Limpa a sessão se o arquivo for removido
    elif uploaded_file_pre is None and st.session_state.get('df_original') is not None:
        st.session_state['df_original'] = None
        st.session_state['volumoso_ids'] = set()
        st.session_state['last_uploaded_name'] = None
        st.session_state['df_agrupado_para_edicao'] = None
        st.rerun() 


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

    # Campo para o usuário especificar o nome da aba, útil para arquivos .xlsx
    if uploaded_file_pos is not None and uploaded_file_pos.name.endswith('.xlsx'):
        sheet_name = st.text_input(
            "Seu arquivo é um Excel (.xlsx). Digite o nome da aba com os dados da rota (ex: Table 3):", 
            value=sheet_name_default
        )

    if uploaded_file_pos is not None:
        try:
            if uploaded_file_pos.name.endswith('.csv'):
                df_input_pos = pd.read_csv(uploaded_file_pos)
            else:
                df_input_pos = pd.read_excel(uploaded_file_pos, sheet_name=sheet_name)
            
            df_input_pos.columns = df_input_pos.columns.str.strip() 
            df_input_pos.columns = df_input_pos.columns.str.lower()
            

            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_input_pos)}** registros.")
            
            df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao = processar_rota_para_impressao(df_input_pos)
            
            if df_final_geral is not None and not df_final_geral.empty:
                st.markdown("---")
                st.subheader("2.2 Resultado Final (Lista de Impressão GERAL)")
                st.caption("A tabela abaixo é apenas para visualização. Use a área de texto ou o download para cópia rápida.")
                
                df_visualizacao_geral = df_final_geral.copy()
                df_visualizacao_geral.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                st.dataframe(df_visualizacao_geral, use_container_width=True)

                copia_data_geral = '\n'.join(df_final_geral['Lista de Impressão'].astype(str).tolist())
                
                
                # --- SEÇÃO DEDICADA AOS NÃO-VOLUMOSOS ---
                st.markdown("---")
                st.header("✅ Lista de Impressão APENAS NÃO-VOLUMOSOS")
                
                if not df_nao_volumosos_impressao.empty:
                    st.success(f"Foram encontrados **{len(df_nao_volumosos_impressao)}** endereços com pacotes NÃO-volumosos nesta rota.")
                    
                    df_visualizacao_nao_vol = df_nao_volumosos_impressao.copy()
                    df_visualizacao_nao_vol.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                    st.dataframe(df_visualizacao_nao_vol, use_container_width=True)
                    
                    copia_data_nao_volumosos = '\n'.join(df_nao_volumosos_impressao['Lista de Impressão'].astype(str).tolist())
                    
                else:
                    st.info("Todos os pedidos nesta rota estão marcados como volumosos ou a lista está vazia.")
                    
                # --- SEÇÃO DEDICADA AOS VOLUMOSOS ---
                st.markdown("---")
                st.header("📦 Lista de Impressão APENAS VOLUMOSOS")
                
                if not df_volumosos_impressao.empty:
                    st.warning(f"Foram encontrados **{len(df_volumosos_impressao)}** endereços com pacotes volumosos nesta rota.")
                    
                    df_visualizacao_vol = df_volumosos_impressao.copy()
                    df_visualizacao_vol.columns = ['ID(s) Agrupado - Anotações', 'Endereço da Parada']
                    st.dataframe(df_visualizacao_vol, use_container_width=True)
                    
                    copia_data_volumosos = '\n'.join(df_volumosos_impressao['Lista de Impressão'].astype(str).tolist())
                    
                else:
                    st.info("Nenhum pedido volumoso detectado nesta rota (nenhum '*' encontrado no Order ID).")


            else:
                 copia_data_geral = "O arquivo foi carregado, mas a coluna 'Notes' estava vazia ou o processamento não gerou resultados. Verifique o arquivo de rota do Circuit."


        except KeyError as ke:
            if "Table 3" in str(ke) or "Sheet" in str(ke):
                st.error(f"Erro de Aba: A aba **'{sheet_name}'** não foi encontrada no arquivo Excel. Verifique o nome da aba.")
            elif 'notes' in str(ke):
                 st.error(f"Erro de Coluna: A coluna 'Notes' não foi encontrada. Verifique se o arquivo da rota está correto.")
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
            "Conteúdo da Lista de Impressão GERAL (Alinhado à Esquerda):", 
            copia_data_geral, 
            height=300,
            key="text_area_geral"
        )

        # --- ÁREA DE CÓPIA NÃO-VOLUMOSOS ---
        if not df_nao_volumosos_impressao.empty if df_nao_volumosos_impressao is not None else False:
            st.markdown("### 2.4 Copiar para a Área de Transferência (APENAS NÃO-Volumosos)")
            st.success("Lista Filtrada: Contém **somente** os endereços com pacotes **NÃO-volumosos** (sem o '*').")
            
            st.text_area(
                "Conteúdo da Lista de Impressão NÃO-VOLUMOSOS (Alinhado à Esquerda):", 
                copia_data_nao_volumosos, 
                height=150,
                key="text_area_nao_volumosos"
            )
        
        # --- ÁREA DE CÓPIA VOLUMOSOS ---
        if not df_volumosos_impressao.empty if df_volumosos_impressao is not None else False:
            st.markdown("### 2.5 Copiar para a Área de Transferência (APENAS Volumosos)")
            st.warning("Lista Filtrada: Contém **somente** os endereços com pacotes volumosos.")
            
            st.text_area(
                "Conteúdo da Lista de Impressão VOLUMOSOS (Alinhado à Esquerda):", 
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
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Baixe este arquivo. Ele contém três abas: a lista geral, a lista de não-volumosos e a lista de volumosos.",
                key="download_list"
            )
