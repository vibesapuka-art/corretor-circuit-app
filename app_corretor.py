# -*- coding: utf-8 -*-
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

# --- CSS DE ÚLTIMO RECURSO (Simplificado e em CHAMADAS SEPARADAS) ---
# Este bloco foi reescrito para evitar o erro "TypeError" no st.markdown('''...''')
st.markdown("<style>", unsafe_html=True)
st.markdown("""
/* Alinha o texto dentro do st.text_area para a esquerda */
textarea {
    text-align: left !important;
    font-family: monospace;
    white-space: pre-wrap;
}
""", unsafe_html=True)
st.markdown("""
/* Garante que títulos e outros elementos fiquem alinhados à esquerda */
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}
""", unsafe_html=True)
st.markdown("</style>", unsafe_html=True)
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
# NOVAS COLUNAS
COLUNA_GAIOLA = 'Gaiola' 
COLUNA_ID_UNICO = 'ID_UNICO' # ID temporário: Gaiola-Sequence (Ex: A1-1, G3-1)

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
    return x_limpo.mode().iloc[0]


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e o agrupamento.
    A chamada do process.extract foi forçada a ser em uma linha para evitar o SyntaxError.
    """
    # Adicionando tratamento para o caso de a coluna ID_UNICO ainda não existir (apenas para segurança)
    colunas_essenciais_base = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code', COLUNA_GAIOLA]
    for col in colunas_essenciais_base:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada. Verifique se o DataFrame foi carregado e as gaiolas foram confirmadas.")
            return None, None 

    # O ID_UNICO deve ter sido criado na seção 1.3
    if COLUNA_ID_UNICO not in df_entrada.columns:
        st.error(f"Erro interno: A coluna temporária '{COLUNA_ID_UNICO}' não foi gerada. Verifique se o botão 'UNIFICAR, CORRIGIR E AGRUPAR' foi clicado corretamente.")
        return None, None


    df = df_entrada.copy()
    
    # Preenchimento e Garantia de Tipos (Essencial)
    df['Bairro'] = df['Bairro'].astype(str).replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    df[COLUNA_GAIOLA] = df[COLUNA_GAIOLA].astype(str).replace('nan', '', regex=False)
    
    # Cria a coluna numérica para a ORDENAÇÃO. Aqui, usamos a SEQUENCE original do pacote.
    # A coluna ID_UNICO já deve vir com o * do volumoso, se houver.
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace(r'\*|\s', '', regex=True)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)


    # 1. Limpeza e Normalização (Fuzzy Matching)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching para Agrupamento
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos > 0:
        for i, end_principal in enumerate(enderecos_unicos):
            if end_principal not in mapa_correcao:
                # CORREÇÃO DE SINTAXE: Chamada em uma linha
                matches = process.extract(end_principal, enderecos_unicos, scorer=fuzz.WRatio, limit=None)
                
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
    else:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")


    # 3. Aplicação do Endereço Corrigido
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # 4. Agrupamento (POR ENDEREÇO CORRIGIDO E CIDADE)
    colunas_agrupamento = ['Endereco_Corrigido', 'City'] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        # Agrupa os IDs ÚNICOS (Gaiola-Sequence) que já contêm o '*'
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
        'Order ID': df_agrupado['Sequences_Agrupadas'], # IDs ÚNICOS Agrupados (com *)
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'], 
        'Longitude': df_agrupado['Longitude'], 
        'Notes': notas_completas
    })
    
    # Limpa o cache do Streamlit para evitar que dados antigos sejam reutilizados
    st.cache_data.clear()
    
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

# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

st.title("🗺️ Flow Completo Circuit (Pré e Pós-Roteirização)")

# CRIAÇÃO DAS ABAS (ADICIONANDO A TERCEIRA ABA)
tab1, tab2, tab3 = st.tabs(["🚀 Pré-Roteirização (Importação)", "📦 Marcação de Volumosos", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CARGA E PROCESSAMENTO FINAL)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa unifica rotas, corrige erros de digitação e agrupa pedidos. **A marcação de volumosos foi movida para a aba 'Marcação de Volumosos'.**")

    # Inicializa o estado para armazenar a lista de DataFrames carregados (com gaiola e nome)
    if 'loaded_dfs' not in st.session_state:
        st.session_state['loaded_dfs'] = []
    
    st.markdown("---")
    st.subheader("1.1 Carregar Planilhas Originais e Definir Gaiolas")

    uploaded_files_pre = st.file_uploader(
        "Arraste e solte os arquivos originais (CSV/Excel) aqui:", 
        type=['csv', 'xlsx'],
        accept_multiple_files=True, 
        key="file_pre"
    )

    
    if uploaded_files_pre:
        # Verifica se a lista de arquivos mudou, se sim, limpa o estado
        current_file_names = {f.name for f in uploaded_files_pre}
        loaded_file_names = {item['file_name'] for item in st.session_state['loaded_dfs']}

        if current_file_names != loaded_file_names:
            st.session_state['loaded_dfs'] = []
            
            # Inicializa o estado com os novos arquivos
            for i, uploaded_file in enumerate(uploaded_files_pre):
                 # Adiciona um placeholder para a gaiola
                st.session_state['loaded_dfs'].append({
                    'file_name': uploaded_file.name,
                    'file_object': uploaded_file,
                    'gaiola': f"G{i+1}", 
                    'df': None, # O DataFrame bruto
                    'volumosos': set() # Set de Sequences originais marcadas como volumosas
                })
            
            # Limpa qualquer input antigo
            st.session_state['df_unificado_final'] = None

        
        st.markdown("#### Defina o Código da Gaiola para cada arquivo e Inicie a Carga:")
        
        # Usa um form para submeter todas as entradas de gaiola e iniciar a carga
        with st.form("gaiola_form"):
            for i, item in enumerate(st.session_state['loaded_dfs']):
                gaiola_input = st.text_input(
                    f"Código da Gaiola para **{item['file_name']}**", 
                    key=f"gaiola_input_{i}",
                    value=item['gaiola'],
                    max_chars=10
                )
                # Atualiza o item na lista com o valor digitado (temporariamente)
                item['gaiola'] = gaiola_input
                
            submitted = st.form_submit_button("Confirmar Gaiolas e Iniciar Processamento Individual")
            
            if submitted: 
                df_list = []
                gaiolas_ok = True
                
                st.markdown("---")
                st.subheader("Processando Arquivos Individualmente...")

                for i, item in enumerate(st.session_state['loaded_dfs']):
                    gaiola_code = item['gaiola'].strip()
                    uploaded_file = item['file_object']
                    
                    if not gaiola_code:
                        st.warning(f"O arquivo '{uploaded_file.name}' não tem código de gaiola definido. Por favor, preencha.")
                        gaiolas_ok = False
                        break

                    try:
                        # 1. Carregar DataFrame
                        if uploaded_file.name.endswith('.csv'):
                            df_input_pre = pd.read_csv(uploaded_file, encoding='utf-8')
                        else:
                            df_input_pre = pd.read_excel(uploaded_file, sheet_name=0)
                        
                        # 2. Validação e Preparação
                        colunas_basicas = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
                        for col in colunas_basicas:
                            if col not in df_input_pre.columns:
                                raise KeyError(f"A coluna '{col}' está faltando no arquivo '{uploaded_file.name}'.")
                        
                        # 3. Adicionar Gaiola e Coluna de Sequência
                        df_input_pre[COLUNA_GAIOLA] = gaiola_code
                        df_input_pre[COLUNA_SEQUENCE] = df_input_pre[COLUNA_SEQUENCE].astype(str) # Garante que a sequência é string
                        
                        # 4. Salvar o DF processado (com a coluna Gaiola) no estado
                        item['df'] = df_input_pre.copy() 
                        st.session_state['loaded_dfs'][i] = item # Atualiza o item no state

                        st.info(f"✅ Arquivo **{uploaded_file.name}** (Gaiola: **{gaiola_code}**) carregado com **{len(df_input_pre)}** pacotes.")
                        
                    except KeyError as ke:
                        st.error(f"Erro de Coluna no arquivo '{uploaded_file.name}': {ke}")
                        gaiolas_ok = False
                        break
                    except Exception as e:
                        st.error(f"Erro ao carregar o arquivo '{uploaded_file.name}'. Erro: {e}")
                        gaiolas_ok = False
                        break

                if gaiolas_ok:
                    st.success("Carga dos arquivos concluída. Prossiga para a marcação de volumosos na aba 'Marcação de Volumosos'.")
                    st.session_state['df_unificado_final'] = None # Limpa o resultado final para forçar o recálculo
                else:
                    st.session_state['loaded_dfs'] = [] # Se falhar, reseta a lista de arquivos carregados
                    st.session_state['df_unificado_final'] = None


    # Limpa a sessão se o arquivo for removido
    elif uploaded_files_pre is None and st.session_state.get('loaded_dfs'):
        st.session_state['loaded_dfs'] = []
        st.session_state['df_unificado_final'] = None
        st.rerun() 
        

    
    # ----------------------------------------------------------------------------------
    # 1.2 UNIFICAÇÃO, CORREÇÃO E PROCESSAMENTO FINAL
    # ----------------------------------------------------------------------------------
    
    # Verifica se há DFs carregados
    dfs_prontos_para_processar = [item for item in st.session_state['loaded_dfs'] if item.get('df') is not None]

    if dfs_prontos_para_processar:
        st.markdown("---")
        st.subheader("1.2 Unificar e Processar Rotas")
        st.warning("⚠️ **Verifique a aba 'Marcação de Volumosos' antes de prosseguir!**")
        
        limite_similaridade_ajustado = st.slider(
            'Ajuste a Precisão do Corretor (Fuzzy Matching):',
            min_value=80,
            max_value=100,
            value=100, 
            step=1,
            help="Use 100% para garantir que endereços na mesma rua com números diferentes não sejam agrupados (recomendado)."
        )
        st.info(f"O limite de similaridade para agrupamento está em **{limite_similaridade_ajustado}%**.")
        
        
        if st.button("🚀 UNIFICAR, CORRIGIR E AGRUPAR PARA CIRCUIT", key="btn_pre_final_run"):
            
            df_final_list = []
            
            # 1. Unificação e Aplicação do Asterisco (*) e ID_UNICO
            for item in st.session_state['loaded_dfs']:
                # Pula se o DF não foi carregado corretamente
                if item['df'] is None:
                    continue
                    
                df_proc = item['df'].copy()
                gaiola = item['gaiola']
                volumosos = item['volumosos']
                
                # Cria a coluna ID_UNICO sem o asterisco inicial
                df_proc[COLUNA_ID_UNICO] = df_proc[COLUNA_GAIOLA].astype(str) + '-' + df_proc[COLUNA_SEQUENCE].astype(str)

                # Aplica o * (asterisco) no ID_UNICO se a Sequence original estiver no set de volumosos
                for seq_volumoso in volumosos:
                    str_seq_volumoso = str(seq_volumoso)
                    
                    # Filtra os registros que correspondem àquela SEQUENCE e GAIOLA
                    df_proc.loc[
                        (df_proc[COLUNA_SEQUENCE] == str_seq_volumoso) & (df_proc[COLUNA_GAIOLA] == gaiola), 
                        COLUNA_ID_UNICO
                    ] = df_proc[COLUNA_ID_UNICO] + '*'

                df_final_list.append(df_proc)
                
            if not df_final_list:
                st.error("Não há planilhas válidas e processadas para unificar. Carregue os arquivos e clique em 'Confirmar Gaiolas'.")
                # Se falhar aqui, não prossegue
            else:
                # CONCATENAÇÃO FINAL: Junta todos os DataFrames (com ID_UNICO já marcado)
                df_unificado = pd.concat(df_final_list, ignore_index=True)
                st.session_state['df_unificado_final'] = df_unificado.copy()
                
                # 2. Iniciar o processamento e agrupamento (Fuzzy Matching, Agrupamento e Ordenação)
                df_circuit, df_processado_completo = processar_e_corrigir_dados(
                    st.session_state['df_unificado_final'], 
                    limite_similaridade_ajustado
                )
                
                if df_circuit is not None:
                    st.markdown("---")
                    st.header("✅ Resultado Concluído!")
                    
                    total_entradas = len(st.session_state['df_unificado_final'])
                    total_agrupados = len(df_circuit)
                    
                    st.metric(
                        label="Endereços Únicos Agrupados",
                        value=total_agrupados,
                        delta=f"-{total_entradas - total_agrupados} pacotes agrupados"
                    )
                    
                    # --- SAÍDA 1: ARQUIVO PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                    st.subheader("Arquivo para Roteirização (Circuit)")
                    st.dataframe(df_circuit, use_container_width=True)
                    
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
                    df_volumosos = df_processado_completo[
                        df_processado_completo[COLUNA_ID_UNICO].astype(str).str.contains(r'\*', regex=True, na=False)
                    ].copy()
                    
                    df_volumosos['Sort_Key'] = df_volumosos[COLUNA_SEQUENCE].astype(str).str.replace(r'\*|\s', '', regex=True)
                    df_volumosos['Sort_Key'] = pd.to_numeric(df_volumosos['Sort_Key'], errors='coerce')
                    df_volumosos = df_volumosos.sort_values(by=['Gaiola', 'Sort_Key']).drop(columns=['Sort_Key'])

                    if not df_volumosos.empty:
                        st.markdown("---")
                        st.subheader("Planilha de APENAS Volumosos (Pacotes com *)")
                        st.caption(f"Contém **{len(df_volumosos)}** itens marcados com *. Ordenado por Gaiola e Sequência Original.")

                        df_vol_export = df_volumosos[[
                            COLUNA_ID_UNICO, 
                            COLUNA_GAIOLA, 
                            COLUNA_SEQUENCE, 
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

    else:
        # Se não houver DFs prontos, garantir que o resultado final seja limpo
        st.session_state['df_unificado_final'] = None


# ----------------------------------------------------------------------------------
# ABA 2: MARCAÇÃO INDIVIDUAL DE VOLUMOSOS (EXPANDER ISOLADO)
# ----------------------------------------------------------------------------------
with tab2:
    st.header("2. Marcar Pacotes Volumosos por Gaiola (*)")
    st.caption("Marque os números de sequência (Sequence) que são volumosos para cada gaiola. Eles receberão um `*` no ID e serão usados na aba 'Pré-Roteirização'.")

    # Verifica se há DFs carregados e prontos para a marcação
    dfs_prontos_para_marcar = [item for item in st.session_state['loaded_dfs'] if item.get('df') is not None]
    
    if dfs_prontos_para_marcar:
        st.info(f"Arquivos carregados: **{len(dfs_prontos_para_marcar)}** gaiola(s) prontas para marcação.")
        
        # Itera sobre os DFs prontos e cria a interface de marcação em CONTAINERS separados
        for i, item in enumerate(dfs_prontos_para_marcar):
            
            # Usando st.expander para isolar e permitir colapsar a interface de cada gaiola
            with st.expander(f"📦 Gaiola: {item['gaiola']} ({item['file_name']})", expanded=True):
                
                df_current = item['df'].copy()
                gaiola_code = item['gaiola']
                
                st.markdown(f"#### Total de **{len(df_current)}** pacotes na Gaiola **{gaiola_code}**")
                
                # --- Preparação da lista de Sequences Originais ---
                # Garante que as sequências são ordenáveis (para ordenar o checkbox)
                df_current['Sort_Key'] = pd.to_numeric(df_current[COLUNA_SEQUENCE].astype(str).str.replace(r'\*|\s', '', regex=True), errors='coerce').fillna(float('inf'))
                sequences_sorted = df_current.sort_values('Sort_Key')[COLUNA_SEQUENCE].astype(str).unique()
                
                # Armazena os IDs volumosos (Sequences originais) desta gaiola no state
                volumosos_set = item['volumosos']
                
                # Callback para o checkbox
                def update_volumoso_set(seq_id, is_checked, item_index):
                    # Usamos o `session_state` diretamente
                    if is_checked:
                        st.session_state['loaded_dfs'][item_index]['volumosos'].add(seq_id)
                    elif seq_id in st.session_state['loaded_dfs'][item_index]['volumosos']:
                        st.session_state['loaded_dfs'][item_index]['volumosos'].remove(seq_id)
                    
                    # Força a limpeza do resultado final para garantir que o * seja recalculado
                    st.session_state['df_unificado_final'] = None

                st.info(f"Pacotes já marcados como volumosos: **{len(volumosos_set)}** de **{len(sequences_sorted)}**.")

                st.markdown("---")
                st.markdown("##### 2.1 Marcação em Faixa")

                # Colunas para o layout de faixa
                col_start, col_end, col_button_mark, col_button_unmark = st.columns([1.5, 1.5, 2, 2])
                
                # --- Marcação em Faixa ---
                # Garante valores padrão se a lista de sequências for vazia
                start_default = sequences_sorted[0] if len(sequences_sorted) > 0 else "1"
                end_default = sequences_sorted[-1] if len(sequences_sorted) > 0 else "1"
                
                with col_start:
                    # Chave única garantida pelo índice da iteração (i)
                    start_seq = st.text_input(f"Início da Faixa (Seq)", value=start_default, key=f"start_seq_vol_{i}")
                with col_end:
                    # Chave única garantida pelo índice da iteração (i)
                    end_seq = st.text_input(f"Fim da Faixa (Seq)", value=end_default, key=f"end_seq_vol_{i}")
                
                
                # Função de helper para encontrar sequências numéricas entre o range (mesmo que sejam strings)
                def get_sequences_in_range(df, col, start, end):
                    # Tenta converter para numérico para a comparação de faixa
                    df['Temp_Num'] = pd.to_numeric(df[col].astype(str).str.replace(r'\*|\s', '', regex=True), errors='coerce')
                    try:
                        start_num = pd.to_numeric(start, errors='coerce')
                        end_num = pd.to_numeric(end, errors='coerce')
                    except:
                        return []
                    
                    if pd.isna(start_num) or pd.isna(end_num): return []
                    
                    # Filtra usando a coluna Sequence original (que é a chave do volumoso set)
                    sequences_in_range = df[
                        (df['Temp_Num'] >= start_num) & (df['Temp_Num'] <= end_num)
                    ][col].astype(str).unique().tolist()
                    
                    return sequences_in_range
                    
                
                with col_button_mark:
                    # Chave única garantida pelo índice da iteração (i)
                    if st.button("Marcar Faixa", key=f"btn_mark_range_{i}"):
                        sequences_to_mark = get_sequences_in_range(df_current, COLUNA_SEQUENCE, start_seq, end_seq)
                        for seq in sequences_to_mark:
                            st.session_state['loaded_dfs'][i]['volumosos'].add(seq)
                        st.session_state['df_unificado_final'] = None # Força recálculo no processamento
                        st.rerun() # Adiciona rerun para atualizar a contagem de volumosos no info

                with col_button_unmark:
                    # Chave única garantida pelo índice da iteração (i)
                    if st.button("Limpar Faixa", key=f"btn_unmark_range_{i}"):
                        sequences_to_unmark = get_sequences_in_range(df_current, COLUNA_SEQUENCE, start_seq, end_seq)
                        for seq in sequences_to_unmark:
                            if seq in st.session_state['loaded_dfs'][i]['volumosos']:
                                st.session_state['loaded_dfs'][i]['volumosos'].remove(seq)
                        st.session_state['df_unificado_final'] = None # Força recálculo no processamento
                        st.rerun() # Adiciona rerun para atualizar a contagem de volumosos no info


                st.markdown("---")
                st.markdown("##### 2.2 Marcação Individual")
                
                with st.container(height=250):
                    # Marcação individual por checkbox
                    for seq_id in sequences_sorted:
                        is_checked = seq_id in volumosos_set
                        # Chave única garantida pela gaiola e sequence (gaiola_code, seq_id)
                        st.checkbox(
                            f"Seq: {seq_id}", 
                            value=is_checked, 
                            key=f"vol_{gaiola_code}_{seq_id}",
                            on_change=update_volumoso_set, 
                            args=(seq_id, not is_checked, i) 
                        )
    
    else:
        st.warning("⚠️ **Nenhum arquivo carregado.** Por favor, vá para a aba 'Pré-Roteirização' (Seção 1.1), carregue os arquivos e clique em 'Confirmar Gaiolas' para habilitar a marcação.")



# ----------------------------------------------------------------------------------
# ABA 3: PÓS-ROTEIRIZAÇÃO (LIMPEZA P/ IMPRESSÃO)
# ----------------------------------------------------------------------------------

with tab3:
    st.header("3. Limpar Saída do Circuit para Impressão")
    st.warning("⚠️ Atenção: Use o arquivo CSV/Excel que foi gerado *após a conversão* do PDF da rota do Circuit.")

    st.markdown("---")
    st.subheader("3.1 Carregar Arquivo da Rota")

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

    if uploaded_file_pos is not None and uploaded_file_pos.name.endswith('.xlsx'):
        sheet_name = st.text_input(
            "Seu arquivo é um Excel (.xlsx). Digite o nome da aba com os dados da rota (ex: Table 3):", 
            value=st.session_state.get('sheet_name_pos', sheet_name_default),
            key="sheet_name_pos_input"
        )
        st.session_state['sheet_name_pos'] = sheet_name 
    
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
                st.subheader("3.2 Lista Completa (Para Cópia/Impressão)")
                
                # CORREÇÃO: Removendo 'Estimated Arrival Time' para evitar KeyError
                df_visualizacao = df_extracted[['#', 'Lista de Impressão', 'Address']].copy()
                df_visualizacao.columns = ['# Parada', 'ID(s) Agrupado - Anotações', 'Endereço da Parada']
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
            
    
    if uploaded_file_pos is not None:
        st.markdown("### 3.3 Copiar Lista Completa para a Área de Transferência")
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


    st.markdown("---")
    st.header("📦 3.4 Filtrar Apenas Volumosos (Mantendo a Sequência)")

    if df_extracted is not None and not df_extracted.empty:
        
        if st.button("✨ Mostrar APENAS Pacotes Volumosos (*)", key="btn_filtro_volumosos"):
            
            df_volumosos = df_extracted[
                df_extracted['Ordem ID'].astype(str).str.contains(r'\*', regex=True, na=False)
            ].copy() 
            
            if not df_volumosos.empty:
                st.success(f"Filtro aplicado! Encontrados **{len(df_volumosos)}** paradas com itens volumosos.")

                copia_data_volumosos = '\n'.join(df_volumosos['Lista de Impressão'].astype(str).tolist())
                
                st.subheader("Lista de Volumosos Filtrada (Sequência do Circuit)")

                # CORREÇÃO: Removendo 'Estimated Arrival Time' para evitar KeyError
                df_vol_visualizacao = df_volumosos[['#', 'Lista de Impressão', 'Address']].copy()
                df_vol_visualizacao.columns = ['# Parada', 'ID(s) Agrupado - Anotações', 'Endereço da Parada']
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
        st.info("Carregue e processe um arquivo de rota do Circuit na seção 3.1 para habilitar o filtro.")
