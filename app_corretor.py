# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os
import json # Novo import para manipulação de JSON

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Circuit Flow Completo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURAÇÃO DO DICIONÁRIO FIXO ---
DICTIONARY_FILE = 'address_dictionary.json' # Nome do arquivo que será salvo/carregado

def load_dictionary(filepath):
    """Carrega o dicionário de correções do arquivo JSON."""
    if not os.path.exists(filepath):
        st.warning(f"Aviso: Arquivo de dicionário '{filepath}' não encontrado. Criando um novo.")
        return {}
    try:
        # Tenta carregar o dicionário
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Erro: O arquivo '{filepath}' está corrompido ou vazio. Retornando dicionário vazio.")
        return {}
    except Exception as e:
        st.error(f"Erro ao carregar o dicionário: {e}")
        return {}

def save_dictionary(filepath, data):
    """Salva o dicionário de correções no arquivo JSON."""
    try:
        # Salva o dicionário atualizado
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        st.success(f"Dicionário de correções salvo com sucesso em: {filepath}")
    except Exception as e:
        st.error(f"Erro ao salvar dicionário: {e}")

def normalize_address(address_string):
    """
    Normaliza a string de endereço para uso como chave de busca no dicionário
    e no fuzzy matching.
    """
    if pd.isna(address_string):
        return ""
    address_string = str(address_string).lower().strip()
    
    # Remove caracteres que NÃO são alfanuméricos (\w), espaço (\s) OU VÍRGULA (,)
    address_string = re.sub(r'[^\w\s,]', '', address_string) 
    
    # Substitui múltiplos espaços por um único
    address_string = re.sub(r'\s+', ' ', address_string)
    
    # Substitui abreviações comuns para padronização
    address_string = address_string.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return address_string

# --- Carregamento inicial do Dicionário (uma vez por sessão) ---
if 'fixed_dict' not in st.session_state:
    st.session_state['fixed_dict'] = load_dictionary(DICTIONARY_FILE)

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
    white-space: pre-wrap; /* Garante que quebras de linha corretas */
}
/* Alinha os títulos e outros elementos em geral */
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'

# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# ===============================================

def limpar_endereco(endereco):
    """
    Normaliza o texto do endereço para melhor comparação.
    Reusa a função normalize_address para consistência.
    """
    return normalize_address(endereco)


# Função auxiliar para lidar com valores vazios no mode()
def get_most_common_or_empty(x):
    """
    Retorna o valor mais comum de uma Série Pandas ou uma string vazia se todos forem NaN.
    """
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    return x_limpo.mode().iloc[0]


@st.cache_data(show_spinner=False)
def processar_e_corrigir_dados(df_entrada, limite_similaridade, fixed_dict):
    """
    Função principal que aplica a correção FIXA, depois o fuzzy matching e o agrupamento.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None, None 

    df = df_entrada.copy()
    
    # Preenchimento inicial e conversão de tipo
    df['Bairro'] = df['Bairro'].astype(str).replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    df[COLUNA_LATITUDE] = pd.to_numeric(df[COLUNA_LATITUDE], errors='coerce')
    df[COLUNA_LONGITUDE] = pd.to_numeric(df[COLUNA_LONGITUDE], errors='coerce')


    # 1. Limpeza e Normalização (Cria a chave de busca)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    
    # -----------------------------------------------------------------
    # 2. NOVO PASSO CRÍTICO: Aplica Correção do Dicionário Fixo
    # -----------------------------------------------------------------
    
    correcoes_aplicadas = 0
    
    def apply_fixed_correction(row):
        """Sobrescreve Lat/Lng se a chave normalizada existir no dicionário."""
        normalized_address = row['Endereco_Limpo']
        if normalized_address in fixed_dict:
            # Sobrescreve as coordenadas originais/geocodificadas com a correção manual
            row[COLUNA_LATITUDE] = fixed_dict[normalized_address]['lat']
            row[COLUNA_LONGITUDE] = fixed_dict[normalized_address]['lng']
            row['Source_Lat_Lng'] = 'FIXED_DICT'
            nonlocal correcoes_aplicadas # Permite modificar a variável externa
            correcoes_aplicadas += 1
        else:
            row['Source_Lat_Lng'] = 'ORIGINAL'
        return row

    # Aplica a função linha por linha para sobrescrever as colunas de Lat/Lng
    df = df.apply(apply_fixed_correction, axis=1)
    
    st.info(f"**{correcoes_aplicadas}** coordenadas foram sobrescritas pelo Dicionário Fixo de Correções.")
    # -----------------------------------------------------------------


    # Prepara coluna numérica de ordenação 
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)


    # 3. Fuzzy Matching para Agrupamento 
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching para Agrupamento...")
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

    # 4. Aplicação do Endereço Corrigido
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # 5. Agrupamento (POR ENDEREÇO CORRIGIDO E CIDADE)
    colunas_agrupamento = ['Endereco_Corrigido', 'City'] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        # Agrupa as sequências (que já contêm o *)
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'\*', '', str(y))) if re.sub(r'\*', '', str(y)).isdigit() else float('inf'))))), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        # AQUI usamos o valor que PODE TER SIDO CORRIGIDO PELO DICIONÁRIO FIXO
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        Bairro_Agrupado=('Bairro', get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        
        Min_Sequence=('Sequence_Num', 'min') 
        
    ).reset_index()

    # 6. ORDENAÇÃO
    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    # 7. Formatação do DF para o CIRCUIT 
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
    
    return df_circuit, df


# ===============================================
# FUNÇÕES DE PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
# ===============================================

def processar_rota_para_impressao(df_input):
    """
    Processa o DataFrame da rota, extrai 'Ordem ID' da coluna 'Notes' e prepara para cópia.
    
    RETORNA: 
    - df_final_geral: Lista de impressão de todos os pedidos.
    - df_volumosos: Lista de impressão APENAS dos pedidos que contêm '*' (volumosos).
    - df_nao_volumosos: Lista de impressão APENAS dos pedidos que NÃO contêm '*' (não-volumosos).
    """
    coluna_notes_lower = 'notes'
    
    if coluna_notes_lower not in df_input.columns:
        raise KeyError(f"A coluna '{coluna_notes_lower}' não foi encontrada.")
    
    df = df_input.copy()
    df[coluna_notes_lower] = df[coluna_notes_lower].astype(str)
    df = df.dropna(subset=[coluna_notes_lower]) 
    
    # 2. Separar a coluna Notes: Parte antes do ';' é o Order ID (que contém o *)
    df[coluna_notes_lower] = df[coluna_notes_lower].str.strip('"')
    
    # Divide a coluna na primeira ocorrência de ';'
    df_split = df[coluna_notes_lower].str.split(';', n=1, expand=True)
    df['Ordem ID'] = df_split[0].str.strip()
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    
    # 3. Formatação Final da Tabela (APENAS ID e ANOTAÇÕES)
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    # DataFrame FINAL GERAL
    df_final_geral = df[['Lista de Impressão', 'address']].copy() 
    
    # 4. FILTRAR VOLUMOSOS: Cria um DF separado APENAS para volumosos
    df_volumosos = df[df['Ordem ID'].str.contains(r'\*', regex=True)].copy()
    df_volumosos_impressao = df_volumosos[['Lista de Impressão', 'address']].copy()
    
    # 5. FILTRAR NÃO-VOLUMOSOS: Cria um DF separado APENAS para não-volumosos
    df_nao_volumosos = df[~df['Ordem ID'].str.contains(r'\*', regex=True)].copy() 
    df_nao_volumosos_impressao = df_nao_volumosos[['Lista de Impressão', 'address']].copy()
    
    return df_final_geral, df_volumosos_impressao, df_nao_volumosos_impressao


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

st.title("🗺️ Flow Completo Circuit (Pré e Pós-Roteirização)")

# CRIAÇÃO DAS ABAS 
tab1, tab2, tab3 = st.tabs(["🚀 Pré-Roteirização (Importação)", "🗃️ Gerenciar Dicionário Fixo", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa aplica correções fixas, corrige erros de digitação e agrupa pedidos.")

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
            
            colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
            for col in colunas_essenciais:
                 if col not in df_input_pre.columns:
                     raise KeyError(f"A coluna '{col}' está faltando na sua planilha.")
            
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

        with st.container(height=300):
            for order_id in ordens_originais_sorted:
                
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

            # Chama o processamento com o dicionário fixo
            df_circuit, df_processado_completo = processar_e_corrigir_dados(
                df_para_processar, 
                limite_similaridade_ajustado, 
                st.session_state.fixed_dict
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
                
                df_volumosos_separado = df_circuit[
                    df_circuit['Order ID'].astype(str).str.contains(r'\*', regex=True)
                ].copy()
                
                st.subheader("Arquivo para Roteirização (Circuit)")
                st.dataframe(df_circuit, use_container_width=True)
                
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

    elif uploaded_file_pre is None and st.session_state.get('df_original') is not None:
        st.session_state['df_original'] = None
        st.session_state['volumoso_ids'] = set()
        st.session_state['last_uploaded_name'] = None
        st.rerun() 


# ----------------------------------------------------------------------------------
# ABA 2: GERENCIAMENTO DO DICIONÁRIO FIXO
# ----------------------------------------------------------------------------------

with tab2:
    st.header("🗃️ Gerenciar Dicionário de Correções Fixas")
    st.caption("Correções salvas aqui serão aplicadas automaticamente ANTES do Fuzzy Matching.")
    
    
    # --- 2.1 Formulário de Cadastro ---
    st.subheader("2.1 Adicionar Nova Correção")

    with st.form("form_add_correction", clear_on_submit=True):
        
        address_input = st.text_input(
            "Endereço Exato Digitado (Chave)",
            placeholder="Ex: Rua das Palmeiras, 63",
            help="Insira o texto EXATO que o cliente digitou. A busca será feita com a versão normalizada."
        )

        col_lat, col_lng = st.columns(2)
        lat_input = col_lat.text_input("Latitude Corrigida", placeholder="-23.55000")
        lng_input = col_lng.text_input("Longitude Corrigida", placeholder="-46.63300")
        
        submitted = st.form_submit_button("➕ Adicionar Correção e Salvar")

        if submitted:
            if not address_input or not lat_input or not lng_input:
                st.error("Preencha todos os campos: Endereço, Latitude e Longitude.")
            else:
                try:
                    lat_value = float(lat_input)
                    lng_value = float(lng_input)
                    
                    normalized_key = normalize_address(address_input)
                    
                    if normalized_key in st.session_state.fixed_dict:
                        st.warning(f"A chave '{normalized_key}' já existe e será atualizada!")
                    
                    st.session_state.fixed_dict[normalized_key] = {
                        "lat": lat_value,
                        "lng": lng_value,
                        "original_string": address_input.strip() 
                    }
                    
                    save_dictionary(DICTIONARY_FILE, st.session_state.fixed_dict)
                    st.rerun()
                    
                except ValueError:
                    st.error("Latitude e Longitude devem ser números válidos.")
    
    st.markdown("---")
    
    # --- 2.2 Lista de Correções Ativas ---
    st.subheader(f"2.2 Correções Ativas ({len(st.session_state.fixed_dict)} entradas)")
    
    if st.session_state.fixed_dict:
        
        data_for_display = []
        for key, data in st.session_state.fixed_dict.items():
            data_for_display.append({
                "Chave Normalizada": key,
                "Endereço Original": data.get('original_string', key),
                "Latitude": data['lat'],
                "Longitude": data['lng']
            })
        
        df_display = pd.DataFrame(data_for_display)
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        buffer_json = io.StringIO()
        json.dump(st.session_state.fixed_dict, buffer_json, indent=4, ensure_ascii=False)
        buffer_json.seek(0)
        
        st.download_button(
            label="⬇️ Baixar JSON do Dicionário",
            data=buffer_json.read(),
            file_name=DICTIONARY_FILE,
            mime="application/json",
            key="download_dict_json"
        )
        
        def clear_dictionary():
            st.session_state.fixed_dict = {}
            save_dictionary(DICTIONARY_FILE, st.session_state.fixed_dict)
        
        if st.button("🔴 Limpar Todo o Dicionário Fixo", type="secondary"):
            st.warning("Tem certeza? Isso apagará TODAS as correções salvas.")
            if st.button("SIM, APAGAR DEFINITIVAMENTE", type="primary"):
                clear_dictionary()
                st.rerun()
                
    else:
        st.info("Nenhuma correção fixa cadastrada. Use o formulário acima para adicionar a primeira.")


# ----------------------------------------------------------------------------------
# ABA 3: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO E SEPARAÇÃO DE VOLUMOSOS)
# ----------------------------------------------------------------------------------

with tab3:
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
                 st.error(f"Erro de Coluna: A coluna 'notes' não foi encontrada. Verifique se o arquivo da rota está correto.")
             elif 'address' in str(ke):
                 st.error(f"Erro de Coluna: A coluna 'address' não foi encontrada. Verifique o arquivo de rota.")
             else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {e}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o arquivo da rota (PDF convertido) está no formato CSV ou Excel. Erro: {e}")
            
    
    if uploaded_file_pos is not None:
        
        st.markdown("### 2.3 Copiar para a Área de Transferência (Lista GERAL)")
        st.info("Para copiar: **Selecione todo o texto** abaixo (Ctrl+A / Cmd+A) e pressione **Ctrl+C / Cmd+C**.")
        
        st.text_area(
            "Conteúdo da Lista de Impressão GERAL (Alinhado à Esquerda):", 
            copia_data_geral, 
            height=300,
            key="text_area_geral"
        )

        if not df_nao_volumosos_impressao.empty if df_nao_volumosos_impressao is not None else False:
            st.markdown("### 2.4 Copiar para a Área de Transferência (APENAS NÃO-Volumosos)")
            st.success("Lista Filtrada: Contém **somente** os endereços com pacotes **NÃO-volumosos** (sem o '*').")
            
            st.text_area(
                "Conteúdo da Lista de Impressão NÃO-VOLUMOSOS (Alinhado à Esquerda):", 
                copia_data_nao_volumosos, 
                height=150,
                key="text_area_nao_volumosos"
            )
        
        if not df_volumosos_impressao.empty if df_volumosos_impressao is not None else False:
            st.markdown("### 2.5 Copiar para a Área de Transferência (APENAS Volumosos)")
            st.warning("Lista Filtrada: Contém **somente** os endereços com pacotes volumosos.")
            
            st.text_area(
                "Conteúdo da Lista de Impressão VOLUMOSOS (Alinhado à Esquerda):", 
                copia_data_volumosos, 
                height=150,
                key="text_area_volumosos"
            )
        
        
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
