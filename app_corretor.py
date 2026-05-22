# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
import math

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
</style>
""", unsafe_allow_html=True)

# --- Configurações Globais (Colunas Iniciais) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
COLUNA_BAIRRO = 'Bairro' 

# --- Configurações de MIME Type e Cache ---
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3" 
CACHE_COLUMNS = ['Endereco_Completo_Cache', 'Latitude_Corrigida', 'Longitude_Corrigida']

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
    except Exception:
        return pd.DataFrame(columns=CACHE_COLUMNS)

# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO
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
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada.")
            return None, [] 

    df = df_entrada.copy()
    corrected_addresses = [] 
    
    df[COLUNA_BAIRRO] = df[COLUNA_BAIRRO].astype(str).str.strip().replace('nan', '')
    df['City'] = df['City'].astype(str).replace('nan', '')
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '')
    
    df['Chave_Busca_Cache'] = df[COLUNA_ENDERECO].astype(str).str.strip() + ', ' + df[COLUNA_BAIRRO].astype(str).str.strip()
    df['Chave_Busca_Cache'] = df['Chave_Busca_Cache'].str.replace(r',\s*$', '', regex=True).str.replace(r',\s*,', ',', regex=True)

    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf'))

    if not df_cache_geoloc.empty:
        df_cache_lookup = df_cache_geoloc.rename(columns={
            'Endereco_Completo_Cache': 'Chave_Busca_Cache', 
            'Latitude_Corrigida': 'Cache_Lat',
            'Longitude_Corrigida': 'Cache_Lon'
        })
        df = pd.merge(df, df_cache_lookup, on='Chave_Busca_Cache', how='left')
        cache_mask = df['Cache_Lat'].notna()
        df.loc[cache_mask, COLUNA_LATITUDE] = df.loc[cache_mask, 'Cache_Lat']
        df.loc[cache_mask, COLUNA_LONGITUDE] = df.loc[cache_mask, 'Cache_Lon']
        corrected_addresses = df.loc[cache_mask, 'Chave_Busca_Cache'].unique().tolist()
        df = df.drop(columns=['Cache_Lat', 'Cache_Lon'], errors='ignore')
    
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    for end_principal in enderecos_unicos:
        if end_principal not in mapa_correcao:
            matches = process.extract(end_principal, enderecos_unicos, scorer=fuzz.WRatio, limit=None)
            grupo_matches = [match[0] for match in matches if match[1] >= limite_similaridade]
            endereco_oficial_original = get_most_common_or_empty(df[df['Endereco_Limpo'].isin(grupo_matches)][COLUNA_ENDERECO])
            if not endereco_oficial_original:
                 endereco_oficial_original = end_principal 
            for end_similar in grupo_matches:
                mapa_correcao[end_similar] = endereco_oficial_original
                
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    df_agrupado = df.groupby(['Endereco_Corrigido', 'City', COLUNA_BAIRRO]).agg(
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ', '.join(map(str, sorted(x, key=lambda y: int(re.sub(r'\*', '', str(y))) if re.sub(r'\*', '', str(y)).isdigit() else float('inf'))))), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        Bairro_Agrupado=(COLUNA_BAIRRO, get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        Min_Sequence=('Sequence_Num', 'min') 
    ).reset_index()

    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    endereco_completo_circuit = df_agrupado['Endereco_Corrigido'] + ', ' + df_agrupado['Bairro_Agrupado'].str.strip()
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*,', ',', regex=True).str.replace(r',\s*$', '', regex=True) 
    
    notas_completas = df_agrupado['Sequences_Agrupadas'] + '; Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + ' | Cidade: ' + df_agrupado['City'] + ' | CEP: ' + df_agrupado['Zipcode_Agrupado']
    
    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    }) 
    
    df_circuit.insert(0, 'Sequence_Base', range(1, len(df_circuit) + 1))
    return df_circuit, corrected_addresses 

# ===============================================
# FUNÇÃO DE SPLIT DE ROTAS
# ===============================================
def split_dataframe_for_drivers(df_circuit, num_motoristas):
    if df_circuit is None or df_circuit.empty:
        return {}
    
    df_export = df_circuit.copy()
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
        
        if 'Sequence_Base' in df_motorista.columns:
            df_motorista = df_motorista.drop(columns=['Sequence_Base'])
            
        df_motorista = df_motorista[['Order ID', 'Address', 'Latitude', 'Longitude', 'Notes']]
        rotas_divididas[f"Motorista {i+1} ({len(df_motorista)} Paradas)"] = df_motorista
        start_index = end_index
        
    return rotas_divididas

# ===============================================
# INTERFACE PRINCIPAL
# ===============================================
conn = get_db_connection()
create_table_if_not_exists(conn)

st.title("🗺️ Flow Completo Circuit")

tab1, tab_split, tab2, tab3 = st.tabs(["🚀 1. Pré-Roteirização", "✂️ 2. Split Route", "📋 3. Pós-Roteirização (Impressão)", "💾 4. Cache"])

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'volumoso_ids' not in st.session_state:
    st.session_state['volumoso_ids'] = set() 
if 'df_circuit_agrupado_pre' not in st.session_state: 
    st.session_state['df_circuit_agrupado_pre'] = None

# --- ABA 1: PRÉ-ROTEIRIZAÇÃO ---
with tab1:
    st.header("Passo 1: Enviar Planilha e Marcar Volumosos")
    uploaded_file_pre = st.file_uploader("Arraste o arquivo original (CSV/Excel):", type=['csv', 'xlsx'], key="file_pre")

    if uploaded_file_pre is not None:
        try:
            if uploaded_file_pre.name.endswith('.csv'):
                df_input_pre = pd.read_csv(uploaded_file_pre)
            else:
                df_input_pre = pd.read_excel(uploaded_file_pre, sheet_name=0)
            st.session_state['df_original'] = df_input_pre.copy()
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")

    if st.session_state['df_original'] is not None:
        st.subheader("Selecione os IDs Volumosos para marcar com (*)")
        
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

        NUM_COLS = 5
        chunked_list = [ordens_originais_sorted[i:i + NUM_COLS] for i in range(0, len(ordens_originais_sorted), NUM_COLS)]

        with st.container(height=200):
            for row_chunk in chunked_list:
                cols = st.columns(len(row_chunk)) 
                for col_index, order_id in enumerate(row_chunk):
                    with cols[col_index]: 
                        is_checked = order_id in st.session_state['volumoso_ids']
                        st.checkbox(str(order_id), value=is_checked, key=f"vol_{order_id}", on_change=update_volumoso_ids, args=(order_id, not is_checked))

        if st.button("🚀 Gerar Arquivo Único Para o Circuit", key="btn_pre_final"):
            df_para_processar = st.session_state['df_original'].copy()
            df_para_processar[COLUNA_SEQUENCE] = df_para_processar[COLUNA_SEQUENCE].astype(str)
            
            for id_volumoso in st.session_state['volumoso_ids']:
                str_id_volumoso = str(id_volumoso)
                df_para_processar.loc[df_para_processar[COLUNA_SEQUENCE] == str_id_volumoso, COLUNA_SEQUENCE] = str_id_volumoso + '*'

            df_cache = load_geoloc_cache(conn)
            df_circuit, _ = processar_e_corrigir_dados(df_para_processar, 100, df_cache)
            
            if df_circuit is not None:
                st.session_state['df_circuit_agrupado_pre'] = df_circuit
                df_visualizacao = df_circuit.drop(columns=['Sequence_Base'], errors='ignore')
                st.dataframe(df_visualizacao, use_container_width=True)
                
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                    df_visualizacao.to_excel(writer, index=False, sheet_name='Importar_No_Circuit')
                buffer_circuit.seek(0)
                st.download_button(
                    label="📥 Baixar PLANILHA COMPLETA (Jogar no Circuit)",
                    data=buffer_circuit,
                    file_name="Planilha_Para_O_Circuit.xlsx",
                    mime=EXCEL_MIME_TYPE
                )

# --- ABA 2: SPLIT ROUTE ---
with tab_split:
    st.header("✂️ Dividir Rota Antes de Jogar no Circuit")
    df_rota_para_split = st.session_state.get('df_circuit_agrupado_pre')
    if df_rota_para_split is not None:
        num_motoristas = st.slider('Número de Motoristas:', 2, 10, 2)
        if st.button("➡️ Dividir"):
            rotas_divididas = split_dataframe_for_drivers(df_rota_para_split, num_motoristas)
            for i, (nome_rota, df_rota) in enumerate(rotas_divididas.items()):
                st.subheader(nome_rota)
                st.dataframe(df_rota, use_container_width=True)
                buffer_individual = io.BytesIO()
                with pd.ExcelWriter(buffer_individual, engine='openpyxl') as writer:
                    df_rota.to_excel(writer, index=False, sheet_name='Rota')
                buffer_individual.seek(0)
                st.download_button(label=f"⬇️ Baixar {nome_rota}", data=buffer_individual, file_name=f"Rota_{i+1}.xlsx", mime=EXCEL_MIME_TYPE, key=f"dl_m_{i}")

# --- ABA 3: PÓS-ROTEIRIZAÇÃO (IMPRESSÃO INTELIGENTE) ---
with tab2:
    st.header("📋 Passo Final: Impressão e Triagem Física no Galpão")
    st.markdown("Insira aqui qualquer formato de planilha que saiu do Circuit.")
    
    uploaded_file_pos = st.file_uploader("Arraste a planilha que saiu do Circuit:", type=['csv', 'xlsx'], key="file_pos")
    
    if uploaded_file_pos is not None:
        try:
            if uploaded_file_pos.name.endswith('.csv'):
                df_pos = pd.read_csv(uploaded_file_pos)
            else:
                df_pos = pd.read_excel(uploaded_file_pos, sheet_name=0)
            
            # --- LOCALIZADOR DE COLUNAS AUTOMÁTICO (Fuzzy Mapeamento) ---
            # Encontra a melhor coluna mapeando por palavras-chave independentemente da posição
            col_id = None
            col_addr = None
            col_note = None
            
            for col in df_pos.columns:
                col_lower = str(col).lower().strip()
                if col_lower in ['order id', 'order_id', 'id', 'order id', '#']:
                    col_id = col
                elif col_lower in ['address', 'destination address', 'endereço', 'endereco', 'destinatário']:
                    col_addr = col
                elif col_lower in ['notes', 'notes_1', 'nota', 'notas', 'observações']:
                    col_note = col

            # Fallbacks caso venha com nomes completamente fora do padrão (Mapeia pela posição aproximada)
            if not col_id and len(df_pos.columns) > 0: col_id = df_pos.columns[0]
            if not col_addr and len(df_pos.columns) > 1: col_addr = df_pos.columns[1]
            if not col_note and len(df_pos.columns) > 2: col_note = df_pos.columns[2]

            if col_id and col_addr:
                # Garante que Notas exista como string limpa para não quebrar a conferência
                if col_note:
                    df_pos[col_note] = df_pos[col_note].fillna("").astype(str)
                else:
                    df_pos['Notes'] = ""
                    col_note = 'Notes'
                
                df_pos[col_id] = df_pos[col_id].fillna("").astype(str)
                df_pos[col_addr] = df_pos[col_addr].fillna("").astype(str)

                # Regra Inteligente: Caça o asterisco (*) em qualquer parte dos IDs ou notas
                df_pos['Possui_Volumoso'] = df_pos[col_id].str.contains(r'\*', regex=True) | df_pos[col_note].str.contains(r'\*', regex=True)
                
                # Monta a estrutura final padronizada para impressão
                colunas_saida = [col_id, col_addr, col_note]
                df_geral_entrega = df_pos[colunas_saida].copy()
                df_volumosos_print = df_pos[df_pos['Possui_Volumoso'] == True][colunas_saida].copy()
                df_comuns_print = df_pos[df_pos['Possui_Volumoso'] == False][colunas_saida].copy()
                
                st.success(f"Mapeamento Automático Concluído! Colunas Identificadas -> ID: [{col_id}] | Endereço: [{col_addr}] | Notas: [{col_note}]")
                
                # Exibição dos Relatórios e downloads
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("#### 📑 1. Tudo Junto")
                    st.caption("Ordem exata do Circuit para conferência geral.")
                    buf_g = io.BytesIO()
                    df_geral_entrega.to_excel(buf_g, index=False)
                    st.download_button("🖨️ Imprimir Geral", buf_g.getvalue(), "Conferenca_GERAL_Rota.xlsx", EXCEL_MIME_TYPE)
                    st.dataframe(df_geral_entrega, use_container_width=True)
                    
                with c2:
                    st.markdown("#### ⚠️ 2. Apenas Volumosos")
                    st.caption("Lista para o pátio separar cargas com (*).")
                    buf_v = io.BytesIO()
                    df_volumosos_print.to_excel(buf_v, index=False)
                    st.download_button("🖨️ Imprimir Volumosos", buf_v.getvalue(), "Triagem_VOLUMOSOS.xlsx", EXCEL_MIME_TYPE)
                    st.dataframe(df_volumosos_print, use_container_width=True)
                    
                with c3:
                    st.markdown("#### 📦 3. Pacotes Comuns")
                    st.caption("Lista limpa para a esteira padrão.")
                    buf_c = io.BytesIO()
                    df_comuns_print.to_excel(buf_c, index=False)
                    st.download_button("🖨️ Imprimir Comuns", buf_c.getvalue(), "Triagem_COMUNS.xlsx", EXCEL_MIME_TYPE)
                    st.dataframe(df_comuns_print, use_container_width=True)
            else:
                st.error("Não foi possível mapear as colunas básicas de ID e Endereço desse arquivo.")
                    
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
