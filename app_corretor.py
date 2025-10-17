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

# --- CSS Simplificado para evitar TypeError e garantir alinhamento ---
st.markdown("""
<style>
/* Força o alinhamento à esquerda no campo de texto principal */
div[data-testid="stTextarea"] textarea {
    text-align: left !important;
    font-family: monospace;
    white-space: pre-wrap;
}
/* Alinha os títulos e outros elementos em geral */
h1, h2, h3, h4, .stMarkdown {
    text-align: left !important;
}
</style>
""", unsafe_html=True)
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
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code', COLUNA_GAIOLA, COLUNA_ID_UNICO]
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada. Verifique se o DataFrame foi carregado corretamente.")
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

# CRIAÇÃO DAS ABAS 
tab1, tab2 = st.tabs(["🚀 Pré-Roteirização (Importação)", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa unifica rotas de diferentes gaiolas, corrige erros de digitação, marca volumes e agrupa pedidos.")

    # Inicializa o estado para armazenar a lista de DataFrames carregados (com gaiola e nome)
    if 'loaded_dfs' not in st.session_state:
        st.session_state['loaded_dfs'] = []
    
    st.markdown("---")
    st.subheader("1.1 Carregar Planilhas Originais e Definir Gaiolas")
    st.info("Carregue **todas** as planilhas. A marcação de volumosos será feita para cada planilha individualmente na próxima etapa.")

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
                    st.success("Carga dos arquivos concluída. Prossiga para a marcação de volumosos.")
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
    # 1.2 MARCAÇÃO INDIVIDUAL DE VOLUMOSOS
    # ----------------------------------------------------------------------------------
    
    # Verifica se há DFs carregados e prontos para a marcação
    dfs_prontos_para_marcar = [item for item in st.session_state['loaded_dfs'] if item.get('df') is not None]
    
    if dfs_prontos_para_marcar:
        st.markdown("---")
        st.subheader("1.2 Marcar Pacotes Volumosos por Planilha (Volumosos = *)")
        st.info("Marque individualmente os números de sequência (Sequence) que são volumosos em cada gaiola. Eles serão marcados com `*`.")
        
        # Cria as abas para cada arquivo carregado
        tab_titles = [f"{item['gaiola']} ({item['file_name']})" for item in dfs_prontos_para_marcar]
        tabs = st.tabs(tab_titles)

        # Itera sobre os DFs prontos e cria a interface de marcação
        for i, item in enumerate(dfs_prontos_para_marcar):
            with tabs[i]:
                df_current = item['df'].copy()
                gaiola_code = item['gaiola']
                
                st.markdown(f"#### Gaiola **{gaiola_code}** (Total de **{len(df_current)}** pacotes)")
                
                # --- Preparação da lista de Sequences Originais ---
                df_current['Sort_Key'] = pd.to_numeric(df_current[COLUNA_SEQUENCE], errors='coerce').fillna(float('inf'))
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

                # Colunas para o layout de faixa
                col_start, col_end, col_button_mark, col_button_unmark = st.columns([1.5, 1.5, 2, 2])
                
                # --- Marcação em Faixa ---
                # Garante valores padrão se a lista de sequências for vazia
                start_default = sequences_sorted[0] if len(sequences_sorted) > 0 else "1"
                end_default = sequences_sorted[-1] if len(sequences_sorted) > 0 else "1"
                
                with col_start:
                    start_seq = st.text_input(f"Início da Faixa (Seq)", value=start_default, key=f"start_seq_vol_{i}")
                with col_end:
                    end_seq = st.text_input(f"Fim da Faixa (Seq)", value=end_default, key=f"end_seq_vol_{i}")
                
                
                # Função de helper para encontrar sequências numéricas entre o range (mesmo que sejam strings)
                def get_sequences_in_range(df, col, start, end):
                    # Tenta converter para numérico para a comparação de faixa
                    df['Temp_Num'] = pd.to_numeric(df[col], errors='coerce')
                    try:
                        start_num = pd.to_numeric(start, errors='coerce')
                        end_num = pd.to_numeric(end, errors='coerce')
                    except:
                        return []
                    
                    if pd.isna(start_num) or pd.isna(end_num): return []
                    
                    return df[
                        (df['Temp_Num'] >= start_num) & (df['Temp_Num'] <= end_num)
                    ][col].astype(str).unique().tolist()
                    
                
                with col_button_mark:
                    if st.button("Marcar Faixa", key=f"btn_mark_range_{i}"):
                        sequences_to_mark = get_sequences_in_range(df_current, COLUNA_SEQUENCE, start_seq, end_seq)
                        for seq in sequences_to_mark:
                            st.session_state['loaded_dfs'][i]['volumosos'].add(seq)
                        st.rerun() # Adiciona rerun para atualizar a contagem de volumosos no info

                with col_button_unmark:
                    if st.button("Limpar Faixa", key=f"btn_unmark_range_{i}"):
                        sequences_to_unmark = get_sequences_in_range(df_current, COLUNA_SEQUENCE, start_seq, end_seq)
                        for seq in sequences_to_unmark:
                            if seq in st.session_state['loaded_dfs'][i]['volumosos']:
                                st.session_state['loaded_dfs'][i]['volumosos'].remove(seq)
                        st.rerun() # Adiciona rerun para atualizar a contagem de volumosos no info

                st.info(f"**{len(volumosos_set)}** de **{len(sequences_sorted)}** pacotes marcados como volumosos nesta gaiola.")
                
                st.markdown("##### Marcação Individual")
                with st.container(height=250):
                    # Marcação individual por checkbox
                    for seq_id in sequences_sorted:
                        is_checked = seq_id in volumosos_set
                        st.checkbox(
                            f"Seq: {seq_id}", 
                            value=is_checked, 
                            key=f"vol_{gaiola_code}_{seq_id}",
                            on_change=update_volumoso_set, 
                            args=(seq_id, not is_checked, i) 
                        )
        
        # ----------------------------------------------------------------------------------
        # 1.3 UNIFICAÇÃO, CORREÇÃO E PROCESSAMENTO FINAL
        # ----------------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("1.3 Unificar e Processar Rotas")
        
        limite_similaridade_ajustado = st.slider(
            'Ajuste a Precisão do Corretor (Fuzzy Matching):',
            min_value=80,
            max_value=100,
            value=100, 
            step=1,
            help="Use 100% para garantir que endereços na mesma rua com números diferentes não sejam agrupados (recomendado)."
        )
        st.info(f"O limite de similaridade está em **{limite_similaridade_ajustado}%**.")
        
        
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
                        
                        st.download_button()
                            label="📥 Baixar PLANILHA APENAS VOLUMOSOS",
                            data

