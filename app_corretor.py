import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os

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

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
# NOVAS COLUNAS
COLUNA_GAIOLA = 'Gaiola' 
COLUNA_ID_UNICO = 'ID_UNICO' # ID temporário: Gaiola-Sequence (Ex: A1-1, A1-2, G3-1)

# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# ===============================================

def limpar_endereco(endereco):
    """
    Normaliza o texto do endereço para melhor comparação.
    MANTÉM NÚMEROS e VÍRGULAS (,) para que endereços com números diferentes
    não sejam agrupados.
    """
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    
    # Remove caracteres que NÃO são alfanuméricos (\w), espaço (\s) OU VÍRGULA (,)
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    
    # Substitui múltiplos espaços por um único
    endereco = re.sub(r'\s+', ' ', endereco)
    
    # Substitui abreviações comuns para padronização
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return endereco


# Função auxiliar para lidar com valores vazios no mode()
def get_most_common_or_empty(x):
    """
    Retorna o valor mais comum de uma Série Pandas ou uma string vazia se todos forem NaN.
    """
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    # iloc[0] é mais robusto que [0] em alguns ambientes
    return x_limpo.mode().iloc[0]


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e o agrupamento.
    A coluna ID_UNICO já estará ajustada com '*' se necessário.
    Retorna o DF para o Circuit e o DF original com as marcações.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code', COLUNA_GAIOLA, COLUNA_ID_UNICO]
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada. Verifique se o DataFrame foi carregado corretamente.")
            return None, None # Retorna dois Nones

    df = df_entrada.copy()
    
    # Preenchimento e Garantia de Tipos (Essencial)
    df['Bairro'] = df['Bairro'].astype(str).replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    df[COLUNA_GAIOLA] = df[COLUNA_GAIOLA].astype(str).replace('nan', '', regex=False)
    
    # COLUNA_ID_UNICO é a base para agrupamento, pois é única por pacote.
    # Esta coluna já deve estar com o '*' se for um volumoso.

    # Cria uma coluna numérica temporária para a ordenação (ignorando o * e tratando texto)
    # Aqui, usamos a coluna SEQUENCE (que é o número do pacote) para ordenar, não o ID_UNICO.
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    # Tenta converter para numérico, se falhar, preenche com um valor muito alto para ir para o final
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)


    # 1. Limpeza e Normalização (Fuzzy Matching)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching para Agrupamento
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching...")
    total_unicos = len(enderecos_unicos)
    for i, end_principal in enumerate(enderecos_unicos):
        if end_principal not in mapa_correcao:
            matches = process.extract(
                end_principal, 
                enderecos_unicos, 
                scorer=fuzz.WRatio, 
                limit=None
            )
            
            grupo_matches = [match[0] for match in matches if match[1] >= limite_similaridade]
            
            df_grupo = df[df['Endereco_Limpo'].isin(grupo_matches)]
            endereco_oficial_original = get_most_common_or_empty(df_grupo[COLUNA_ENDERECO])
            if not endereco_oficial_original:
                 endereco_oficial_original = end_principal 
            
            for end_similar in grupo_matches:
                mapa_correcao[end_similar] = endereco_oficial_original
                
            progresso_bar.progress((i + 1) / total_unicos, text=f"Processando {i+1} de {total_unicos} endereços únicos...")
    
    progresso_bar.empty()
    st.success("Fuzzy Matching concluído!")

    # 3. Aplicação do Endereço Corrigido
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # 4. Agrupamento (POR ENDEREÇO CORRIGIDO E CIDADE)
    colunas_agrupamento = ['Endereco_Corrigido', 'City'] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        # Agrupa os IDs ÚNICOS (Gaiola-Sequence) que já contêm o '*'
        # Usamos uma ordenação customizada para garantir que o número original do pacote seja respeitado
        Sequences_Agrupadas=(COLUNA_ID_UNICO, 
                             lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'[^\d]', '', str(y).split('-')[-1])) if re.sub(r'[^\d]', '', str(y).split('-')[-1]).isdigit() else float('inf'))))
                            ), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        # Agrupa as informações comuns
        Bairro_Agrupado=('Bairro', get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        
        # Agrupa as gaiolas (mantém TODAS as gaiolas únicas daquele endereço)
        Gaiola_Agrupada=(COLUNA_GAIOLA, lambda x: ','.join(sorted(x.unique()))),
        
        # Captura o menor número de sequência original (sem *) para ordenação
        Min_Sequence=('Sequence_Num', 'min') 
        
    ).reset_index()

    # 5. ORDENAÇÃO: Ordena o DataFrame pelo menor número de sequência. (CRUCIAL!)
    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    # 6. Formatação do DF para o CIRCUIT 
    endereco_completo_circuit = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro_Agrupado'].str.strip() 
    )
    
    endereco_completo_circuit = endereco_completo_circuit.str.replace(r',\s*,', ',', regex=True)
    
    # Incluindo TODAS as Gaiolas nas Notas!
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(int).astype(str) + 
        ' | Gaiola(s): ' + df_agrupado['Gaiola_Agrupada'] +
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode_Agrupado']
    )

    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], # IDs ÚNICOS Agrupados
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    })
    
    return df_circuit, df 


# ===============================================
# FUNÇÕES DE PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
# ===============================================

def extract_circuit_info(df_input_raw):
    """
    Processa o DataFrame da rota do Circuit (raw) para extrair o Order ID e Anotações.
    """
    df = df_input_raw.copy()
    
    # 1. Padronização de Colunas
    df.columns = df.columns.str.strip().str.lower()
    
    # Garante que colunas essenciais existam
    if 'notes' not in df.columns or '#' not in df.columns:
        raise KeyError("O arquivo da rota deve conter as colunas '#' (Sequência de Parada) e 'Notes'.")

    # 2. Processa Notes para obter o ID agrupado (que pode ter o *)
    df['notes'] = df['notes'].astype(str).str.strip('"')
    df = df.dropna(subset=['notes']) 
    
    # Divide a coluna na primeira ocorrência de ';'
    df_split = df['notes'].str.split(';', n=1, expand=True)
    # O ID agrupado (ex: "A1-1,G3-2*") é a primeira parte.
    df['Ordem ID'] = df_split[0].str.strip().str.strip('"') 
    
    # Anotações completas (o resto da string)
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    # 3. Formata a Lista de Impressão
    df['Lista de Impressão'] = (
        df['Ordem ID'].astype(str) + 
        ' - ' + 
        df['Anotações Completas'].astype(str)
    )
    
    return df

def processar_rota_para_impressao(df_input_raw):
    """ Retorna apenas a coluna formatada para cópia (Lista de Impressão) """
    df_extracted = extract_circuit_info(df_input_raw)
    return df_extracted[['Lista de Impressão']]


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

st.title("🗺️ Flow Completo Circuit (Pré e Pós-Roteirização)")

# CRIAÇÃO DAS ABAS 
tab1, tab2 = st.tabs(["🚀 Pré-Roteirização (Importação)", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa unifica rotas de diferentes gaiolas, corrige erros de digitação, marca volumes e agrupa pedidos.")

    # Inicializa o estado para armazenar o DataFrame e as ordens marcadas
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'volumoso_ids' not in st.session_state:
        st.session_state['volumoso_ids'] = set() 
    if 'last_uploaded_name' not in st.session_state:
         st.session_state['last_uploaded_name'] = None
    
    st.markdown("---")
    st.subheader("1.1 Carregar Planilhas Originais e Definir Gaiolas")
    st.info("Carregue **todas** as planilhas (máximo 5) que você deseja unificar em uma rota única. **A numeração dos pacotes (Sequence) será combinada com a Gaiola para criar IDs únicos.**")

    uploaded_files_pre = st.file_uploader(
        "Arraste e solte os arquivos originais (CSV/Excel) aqui:", 
        type=['csv', 'xlsx'],
        accept_multiple_files=True, 
        key="file_pre"
    )

    df_list = [] # Lista para armazenar os DataFrames de todas as planilhas
    gaiolas_ok = True
    
    if uploaded_files_pre:
        st.markdown("#### Defina o Código da Gaiola para cada arquivo:")
        
        # Usa um form para submeter todas as entradas de gaiola de uma vez
        with st.form("gaiola_form"):
            for i, uploaded_file in enumerate(uploaded_files_pre):
                gaiola_input = st.text_input(
                    f"Código da Gaiola para **{uploaded_file.name}**", 
                    key=f"gaiola_{i}",
                    value=st.session_state.get(f"gaiola_value_{i}", f"G{i+1}"),
                    max_chars=10
                )
                st.session_state[f"gaiola_value_{i}"] = gaiola_input
                
            submitted = st.form_submit_button("Confirmar Gaiolas e Iniciar Carga")
            
            if submitted: # Só processa a carga se o botão for clicado
                st.markdown("---")
                # Lógica de processamento de múltiplos arquivos
                
                for i, uploaded_file in enumerate(uploaded_files_pre):
                    gaiola_code = st.session_state[f"gaiola_value_{i}"].strip()
                    
                    if not gaiola_code:
                        st.warning(f"O arquivo '{uploaded_file.name}' não tem código de gaiola definido. Por favor, preencha.")
                        gaiolas_ok = False
                        break

                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df_input_pre = pd.read_csv(uploaded_file, encoding='utf-8')
                        else:
                            df_input_pre = pd.read_excel(uploaded_file, sheet_name=0)
                        
                        colunas_basicas = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
                        for col in colunas_basicas:
                            if col not in df_input_pre.columns:
                                raise KeyError(f"A coluna '{col}' está faltando no arquivo '{uploaded_file.name}'.")
                        
                        # --- INSERÇÃO DA NOVA COLUNA 'Gaiola' E CRIAÇÃO DO ID ÚNICO ---
                        df_input_pre[COLUNA_GAIOLA] = gaiola_code
                        # Cria o ID único que será o Order ID do Circuit
                        df_input_pre[COLUNA_ID_UNICO] = df_input_pre[COLUNA_GAIOLA].astype(str) + '-' + df_input_pre[COLUNA_SEQUENCE].astype(str)
                        df_list.append(df_input_pre)
                        
                    except KeyError as ke:
                        st.error(f"Erro de Coluna no arquivo '{uploaded_file.name}': {ke}")
                        gaiolas_ok = False
                        break
                    except Exception as e:
                        st.error(f"Erro ao carregar o arquivo '{uploaded_file.name}'. Erro: {e}")
                        gaiolas_ok = False
                        break

                if gaiolas_ok and df_list:
                    # CONCATENAÇÃO FINAL
                    df_unificado = pd.concat(df_list, ignore_index=True)
                    
                    # Usa o comprimento total do DF unificado como um 'hash' para resetar as marcações
                    current_hash = len(df_unificado)
                    if st.session_state.get('last_uploaded_hash') != current_hash:
                         st.session_state['volumoso_ids'] = set()
                         st.session_state['last_uploaded_hash'] = current_hash
                         
                    st.session_state['df_original'] = df_unificado.copy()
                    st.success(f"**{len(df_list)}** planilhas unificadas! Total de **{len(df_unificado)}** registros carregados.")
                    gaiolas_unificadas = sorted(list(df_unificado[COLUNA_GAIOLA].unique()))
                    st.caption(f"Gaiola(s) unificada(s): **{', '.join(gaiolas_unificadas)}**")
                else:
                    st.session_state['df_original'] = None

    
    # Limpa a sessão se o arquivo for removido
    elif uploaded_files_pre is None and st.session_state.get('df_original') is not None:
        st.session_state['df_original'] = None
        st.session_state['volumoso_ids'] = set()
        st.session_state['last_uploaded_name'] = None
        st.session_state['last_uploaded_hash'] = None
        st.rerun() 
        

    
    # ----------------------------------------------------------------------------------
    # RESTANTE DA LÓGICA (1.2 e 1.3)
    # ----------------------------------------------------------------------------------
    if st.session_state.get('df_original') is not None:
        
        st.markdown("---")
        st.subheader("1.2 Marcar Pacotes Volumosos (Volumosos = *)")
        
        df_temp = st.session_state['df_original'].copy()
        
        # Lista os IDs ÚNICOS (Gaiola-Sequence) para a marcação
        ordens_unicas_sorted = df_temp[COLUNA_ID_UNICO].astype(str).unique()
        
        # Função de ordenação customizada para o checkbox: ordena pelo número da Sequence dentro do ID_UNICO
        def sort_key_custom(id_unico):
            try:
                # Extrai apenas a parte numérica da Sequence (depois do '-')
                sequence_part = id_unico.split('-')[-1]
                # Remove não-dígitos para garantir a conversão
                num_part = re.sub(r'[^\d]', '', sequence_part)
                return int(num_part)
            except:
                return float('inf')

        ordens_unicas_sorted = sorted(ordens_unicas_sorted, key=sort_key_custom)
        # ----------------------------------------------------------------
        
        
        # Função de callback para atualizar o set de IDs volumosos
        # O ID volumoso agora é o ID_UNICO (Ex: 'A1-1')
        def update_volumoso_ids(id_unico, is_checked):
            if is_checked:
                st.session_state['volumoso_ids'].add(id_unico)
            elif id_unico in st.session_state['volumoso_ids']:
                st.session_state['volumoso_ids'].remove(id_unico)

        st.caption("Marque os **IDs Únicos (Gaiola-Sequência)** das ordens de serviço que são volumosas (serão marcadas com *):")

        # Container para os checkboxes
        with st.container(height=300):
            # Itera pela lista ordenada e exibe um checkbox por ID ÚNICO
            for id_unico in ordens_unicas_sorted:
                
                is_checked = id_unico in st.session_state['volumoso_ids']
                
                st.checkbox(
                    str(id_unico), 
                    value=is_checked, 
                    key=f"vol_{id_unico}",
                    on_change=update_volumoso_ids, 
                    args=(id_unico, not is_checked) 
                )


        st.info(f"**{len(st.session_state['volumoso_ids'])}** pacotes marcados como volumosos (ID Único).")
        
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
        
        
        if st.button("🚀 Iniciar Corretor e Agrupamento", key="btn_pre_final_run"):
            
            # 1. Aplicar a marcação * no DF antes de processar
            df_para_processar = st.session_state['df_original'].copy()
            
            # Garante que a coluna ID_UNICO seja string para manipulação
            df_para_processar[COLUNA_ID_UNICO] = df_para_processar[COLUNA_ID_UNICO].astype(str)
            
            # Aplica o * nos IDs ÚNICOS que estão no set
            for id_volumoso in st.session_state['volumoso_ids']:
                str_id_volumoso = str(id_volumoso)
                
                # Filtra a coluna ID_UNICO para garantir que apenas o ID exato seja marcado
                df_para_processar.loc[
                    df_para_processar[COLUNA_ID_UNICO] == str_id_volumoso, 
                    COLUNA_ID_UNICO
                ] = str_id_volumoso + '*'

            # 2. Iniciar o processamento e agrupamento
            df_circuit, df_processado_completo = processar_e_corrigir_dados(
                df_para_processar, 
                limite_similaridade_ajustado
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
                
                # --- SAÍDA 1: ARQUIVO PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                st.subheader("Arquivo para Roteirização (Circuit)")
                st.dataframe(df_circuit, use_container_width=True)
                
                # Download Circuit
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                    df_circuit.to_excel(writer, index=False, sheet_name='Circuit Import')
                buffer_circuit.seek(0)
                
                st.download_button(
                    label="📥 Baixar ARQUIVO PARA CIRCUIT",
                    data=buffer_circuit,
                    file_name="Circuit_Import_FINAL_MARCADO.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_circuit"
                )
                
                # --- SAÍDA 2: PLANILHA DE VOLUMOSOS SEPARADA ---
                # Filtra o DataFrame completo processado (que inclui o '*' no ID_UNICO)
                df_volumosos = df_processado_completo[
                    df_processado_completo[COLUNA_ID_UNICO].astype(str).str.contains(r'\*', regex=True, na=False)
                ].copy()
                
                # Ordena pelo número de sequência (sem o '*')
                df_volumosos['Sort_Key'] = df_volumosos[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
                df_volumosos['Sort_Key'] = pd.to_numeric(df_volumosos['Sort_Key'], errors='coerce')
                df_volumosos = df_volumosos.sort_values(by='Sort_Key').drop(columns=['Sort_Key'])

                if not df_volumosos.empty:
                    st.markdown("---")
                    st.subheader("Planilha de APENAS Volumosos (Pacotes com *)")
                    st.caption(f"Contém **{len(df_volumosos)}** itens marcados com *.")

                    # Seleciona as colunas relevantes para o motorista/logística
                    df_vol_export = df_volumosos[[
                        COLUNA_ID_UNICO, 
                        COLUNA_GAIOLA, 
                        COLUNA_SEQUENCE, # Mantendo o número original do pacote para conferência
                        COLUNA_ENDERECO, 
                        'Bairro', 
                        'City', 
                        'Zipcode/Postal code',
                        'Endereco_Corrigido'
                    ]].copy()
                    
                    df_vol_export.columns = [
                        'ID Único (Gaiola-Seq*)', 
                        'Gaiola',
                        'Nº da Sequência Original',
                        'Endereço Original', 
                        'Bairro', 
                        'Cidade', 
                        'CEP', 
                        'Endereço Corrigido/Agrupado'
                    ]

                    st.dataframe(df_vol_export, use_container_width=True)
                    
                    # Download Volumosos
                    buffer_vol = io.BytesIO()
                    with pd.ExcelWriter(buffer_vol, engine='openpyxl') as writer:
                        df_vol_export.to_excel(writer, index=False, sheet_name='Volumosos')
                    buffer_vol.seek(0)
                    
                    st.download_button(
                        label="📥 Baixar PLANILHA APENAS VOLUMOSOS",
                        data=buffer_vol,
                        file_name="Volumosos_Marcados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_volumosos"
                    )


# ----------------------------------------------------------------------------------
# ABA 2: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
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
    
    df_raw_pos = None 
    df_extracted = None 
    copia_data = "Nenhum arquivo carregado ou nenhum dado válido encontrado após o processamento."

    # Campo para o usuário especificar o nome da aba, útil para arquivos .xlsx
    if uploaded_file_pos is not None and uploaded_file_pos.name.endswith('.xlsx'):
        sheet_name = st.text_input(
            "Seu arquivo é um Excel (.xlsx). Digite o nome da aba com os dados da rota (ex: Table 3):", 
            value=st.session_state.get('sheet_name_pos', sheet_name_default),
            key="sheet_name_pos_input"
        )
        st.session_state['sheet_name_pos'] = sheet_name # Salva o valor para persistência
    
    # --- Lógica de Carregamento e Processamento Inicial ---
    if uploaded_file_pos is not None:
        try:
            current_sheet_name = sheet_name if uploaded_file_pos.name.endswith('.xlsx') else None

            if uploaded_file_pos.name.endswith('.csv'):
                df_raw_pos = pd.read_csv(uploaded_file_pos, encoding='utf-8')
            else:
                df_raw_pos = pd.read_excel(uploaded_file_pos, sheet_name=current_sheet_name)
            
            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_raw_pos)}** registros.")
            
            df_extracted = extract_circuit_info(df_raw_pos)

            if df_extracted is not None and not df_extracted.empty:
                st.markdown("---")
                st.subheader("2.2 Lista Completa (Para Cópia/Impressão)")
                st.caption("A lista abaixo contém *TODOS* os itens da rota (IDs Únicos Agrupados).")
                
                df_visualizacao = df_extracted[['#', 'Lista de Impressão', 'Address', 'Estimated Arrival Time']].copy()
                df_visualizacao.columns = ['# Parada', 'ID(s) Agrupado - Anotações', 'Endereço da Parada', 'Chegada Estimada']
                st.dataframe(df_visualizacao, use_container_width=True)

                copia_data = '\n'.join(df_extracted['Lista de Impressão'].astype(str).tolist())
            
            else:
                 copia_data = "O arquivo foi carregado, mas a coluna 'Notes' estava vazia ou o processamento não gerou resultados. Verifique o arquivo de rota do Circuit."


        except KeyError as ke:
            if "Table 3" in str(ke) or "Sheet" in str(ke): 
                st.error(f"Erro de Aba: A aba **'{current_sheet_name}'** não foi encontrada no arquivo Excel. Verifique o nome da aba.")
            elif 'notes' in str(ke) or '#' in str(ke):
                 st.error(f"Erro de Coluna: O arquivo da rota deve conter as colunas **#** (Sequência de Parada) e **Notes**. Verifique o arquivo de rota.")
            else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Erro: {e}")
            
    
    # --- 2.3 Área de Cópia e Download da Lista COMPLETA ---
    if uploaded_file_pos is not None:
        st.markdown("### 2.3 Copiar Lista Completa para a Área de Transferência")
        st.info("Para copiar: **Selecione todo o texto** abaixo (Ctrl+A / Cmd+A) e pressione **Ctrl+C / Cmd+C**.")
        
        st.text_area(
            "Conteúdo da Lista de Impressão (Alinhado à Esquerda):", 
            copia_data, 
            height=300,
            key="text_area_completa"
        )

        if df_extracted is not None and not df_extracted.empty:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer: 
                df_extracted[['Lista de Impressão']].to_excel(writer, index=False, sheet_name='Lista Impressao')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Baixar Lista Limpa COMPLETA (Excel)",
                data=buffer,
                file_name="Lista_Ordem_Impressao_COMPLETA.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Baixe este arquivo. Contém todos os itens da rota.",
                key="download_list_completa"
            )


    # ----------------------------------------------------------------------------------
    # 2.4 FILTRO DE VOLUMOSOS (NOVA FUNCIONALIDADE)
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    st.header("📦 2.4 Filtrar Apenas Volumosos (Mantendo a Sequência)")

    if df_extracted is not None and not df_extracted.empty:
        
        # O botão que o usuário deve clicar para ver a lista de volumosos
        if st.button("✨ Mostrar APENAS Pacotes Volumosos (*)", key="btn_filtro_volumosos"):
            
            # FILTRAGEM: O Order ID (que é o ID_UNICO agrupado) tem que conter o * (asterisco)
            df_volumosos = df_extracted[
                df_extracted['Ordem ID'].astype(str).str.contains(r'\*', regex=True, na=False)
            ].copy() 
            
            if not df_volumosos.empty:
                st.success(f"Filtro aplicado! Encontrados **{len(df_volumosos)}** paradas com itens volumosos.")

                copia_data_volumosos = '\n'.join(df_volumosos['Lista de Impressão'].astype(str).tolist())
                
                st.subheader("Lista de Volumosos Filtrada (Sequência do Circuit)")
                st.caption("A tabela abaixo mostra apenas as paradas que contêm pacotes marcados com *. A coluna **# Parada** mostra a sequência original do Circuit.")

                df_vol_visualizacao = df_volumosos[['#', 'Lista de Impressão', 'Address', 'Estimated Arrival Time']].copy()
                df_vol_visualizacao.columns = ['# Parada', 'ID(s) Agrupado - Anotações', 'Endereço da Parada', 'Chegada Estimada']
                st.dataframe(
                    df_vol_visualizacao, 
                    use_container_width=True
                )

                st.markdown("### Copiar Lista de Volumosos")
                st.text_area(
                    "Conteúdo da Lista de Volumosos (Alinhado à Esquerda):", 
                    copia_data_volumosos, 
                    height=200,
                    key="text_area_volumosos"
                )

                # Download Volumosos
                buffer_vol = io.BytesIO()
                with pd.ExcelWriter(buffer_vol, engine='openpyxl') as writer: 
                    df_volumosos[['Lista de Impressão']].to_excel(writer, index=False, sheet_name='Lista Volumosos')
                buffer_vol.seek(0)
                
                st.download_button(
                    label="📥 Baixar Lista de Volumosos FILTRADA (Excel)",
                    data=buffer_vol,
                    file_name="Lista_Ordem_Volumosos_FILTRADA.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Baixe este arquivo. Contém apenas os itens volumosos, mantendo a sequência da rota.",
                    key="download_list_volumosos"
                )
            
            else:
                st.warning("Nenhuma parada na rota contém pacotes marcados com * (volumosos).")

    else:
        st.info("Carregue e processe um arquivo de rota do Circuit na seção 2.1 para habilitar o filtro.")
