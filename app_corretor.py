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
COLUNA_SEQUENCE_GLOBAL = 'Sequence_Global' # Novo! Sequência Contínua para seleção de Volumosos (1, 2, 3...)

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
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace(r'\*|\s', '', regex=True)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)


    # 1. Limpeza e Normalização (Fuzzy Matching)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching para Agrupamento
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching...")
    total_unicos = len(enderecos_unicos)
    
    # Processa apenas se houver endereços
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
        # Usamos uma ordenação customizada para garantir que o número original do pacote seja respeitado
        Sequences_Agrupadas=(COLUNA_ID_UNICO, 
                             # Ordena os IDs agrupados (Ex: [G3-1*, A1-10]) pelo número da Sequence original (1, 10)
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
    if 'volumoso_sequence_global_ids' not in st.session_state: # Novo set: armazena os Sequence_Global
        st.session_state['volumoso_sequence_global_ids'] = set() 
    if 'df_mapa' not in st.session_state: # Novo DF: armazena o mapeamento Sequence_Global <-> ID_UNICO
        st.session_state['df_mapa'] = None

    
    st.markdown("---")
    st.subheader("1.1 Carregar Planilhas Originais e Definir Gaiolas")
    st.info("Carregue **todas** as planilhas. A numeração dos pacotes (Sequence) será combinada com a Gaiola para criar IDs únicos e uma **Sequência Global (1, 2, 3...)** para a seleção de volumosos.")

    uploaded_files_pre = st.file_uploader(
        "Arraste e solte os arquivos originais (CSV/Excel) aqui:", 
        type=['csv', 'xlsx'],
        accept_multiple_files=True, 
        key="file_pre"
    )

    df_list = [] 
    gaiolas_ok = True
    
    if uploaded_files_pre:
        st.markdown("#### Defina o Código da Gaiola para cada arquivo:")
        
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
            
            if submitted: 
                st.markdown("---")
                
                sequence_global_counter = 1 # Inicializa o contador global
                
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
                        
                        # Garante a ordem original de subida do pacote (Sequence)
                        df_input_pre = df_input_pre.sort_values(by=COLUNA_SEQUENCE, key=lambda x: pd.to_numeric(x, errors='coerce')).reset_index(drop=True)

                        # --- CRIAÇÃO DOS IDs E DA SEQUÊNCIA GLOBAL ---
                        df_input_pre[COLUNA_GAIOLA] = gaiola_code
                        # ID ÚNICO (Para o agrupamento)
                        df_input_pre[COLUNA_ID_UNICO] = df_input_pre[COLUNA_GAIOLA].astype(str) + '-' + df_input_pre[COLUNA_SEQUENCE].astype(str)
                        
                        # SEQUÊNCIA GLOBAL (Para a seleção de volumosos)
                        df_input_pre[COLUNA_SEQUENCE_GLOBAL] = range(sequence_global_counter, sequence_global_counter + len(df_input_pre))
                        sequence_global_counter += len(df_input_pre) # Atualiza o contador para a próxima planilha
                        
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
                    df_unificado = pd.concat(df_list, ignore_index=True)
                    
                    current_hash = len(df_unificado)
                    if st.session_state.get('last_uploaded_hash') != current_hash:
                         st.session_state['volumoso_sequence_global_ids'] = set() # Reseta a seleção
                         st.session_state['last_uploaded_hash'] = current_hash
                         
                    st.session_state['df_original'] = df_unificado.copy()
                    
                    # Cria e salva o mapa para uso posterior na marcação
                    st.session_state['df_mapa'] = df_unificado[[COLUNA_SEQUENCE_GLOBAL, COLUNA_ID_UNICO, COLUNA_GAIOLA, COLUNA_SEQUENCE]].copy()
                    
                    st.success(f"**{len(df_list)}** planilhas unificadas! Total de **{len(df_unificado)}** registros (pacotes) carregados.")
                    gaiolas_unificadas = sorted(list(df_unificado[COLUNA_GAIOLA].unique()))
                    st.caption(f"Gaiola(s) unificada(s): **{', '.join(gaiolas_unificadas)}**. Sequência Global de **1** a **{len(df_unificado)}**.")
                else:
                    st.session_state['df_original'] = None

    
    # Limpa a sessão se o arquivo for removido
    elif uploaded_files_pre is None and st.session_state.get('df_original') is not None:
        st.session_state['df_original'] = None
        st.session_state['volumoso_sequence_global_ids'] = set()
        st.session_state['df_mapa'] = None
        st.session_state['last_uploaded_hash'] = None
        st.rerun() 
        

    
    # ----------------------------------------------------------------------------------
    # RESTANTE DA LÓGICA (1.2 e 1.3)
    # ----------------------------------------------------------------------------------
    if st.session_state.get('df_original') is not None:
        
        st.markdown("---")
        st.subheader("1.2 Marcar Pacotes Volumosos (Volumosos = *)")
        
        df_mapa = st.session_state['df_mapa']
        
        # Lista os IDs GLOBAIS (1, 2, 3...) para a marcação
        global_sequences_sorted = df_mapa[COLUNA_SEQUENCE_GLOBAL].unique()
        
        
        # Função de callback para atualizar o set de IDs volumosos (Sequence_Global)
        def update_volumoso_ids(sequence_global_id, is_checked):
            if is_checked:
                st.session_state['volumoso_sequence_global_ids'].add(sequence_global_id)
            elif sequence_global_id in st.session_state['volumoso_sequence_global_ids']:
                st.session_state['volumoso_sequence_global_ids'].remove(sequence_global_id)

        st.caption("Marque os pacotes volumosos usando a **Sequência Global Contínua (1, 2, 3...)**.")

        # Container para os checkboxes
        with st.container(height=300):
            # Itera pela lista ordenada e exibe um checkbox por Sequência Global
            for seq_global in global_sequences_sorted:
                
                # Exibe a gaiola e a sequence original para referência
                map_info = df_mapa[df_mapa[COLUNA_SEQUENCE_GLOBAL] == seq_global].iloc[0]
                gaiola_info = map_info[COLUNA_GAIOLA]
                seq_original = map_info[COLUNA_SEQUENCE]
                
                display_label = f"**{seq_global}** | Gaiola: {gaiola_info}, Seq: {seq_original}"
                
                is_checked = seq_global in st.session_state['volumoso_sequence_global_ids']
                
                st.checkbox(
                    display_label, 
                    value=is_checked, 
                    key=f"vol_global_{seq_global}",
                    on_change=update_volumoso_ids, 
                    args=(seq_global, not is_checked) 
                )
                
        # Adicionar a opção de marcar uma faixa
        st.markdown("##### Marcação em Faixa")
        col_start, col_end, col_button = st.columns([1, 1, 1])
        with col_start:
            start_seq = st.number_input("Início da Faixa Global", min_value=1, max_value=len(df_mapa), value=1, step=1, key="start_seq_vol")
        with col_end:
            end_seq = st.number_input("Fim da Faixa Global", min_value=1, max_value=len(df_mapa), value=len(df_mapa), step=1, key="end_seq_vol")
        
        with col_button:
            st.text("") # Espaço para alinhar o botão
            if st.button("Marcar Faixa como Volumoso", key="btn_mark_range"):
                for seq in range(start_seq, end_seq + 1):
                    if seq in global_sequences_sorted:
                        st.session_state['volumoso_sequence_global_ids'].add(seq)
                st.rerun()
            
            if st.button("Limpar Faixa de Volumosos", key="btn_unmark_range"):
                 for seq in range(start_seq, end_seq + 1):
                    if seq in global_sequences_sorted and seq in st.session_state['volumoso_sequence_global_ids']:
                        st.session_state['volumoso_sequence_global_ids'].remove(seq)
                 st.rerun()
        
        st.info(f"**{len(st.session_state['volumoso_sequence_global_ids'])}** pacotes marcados como volumosos (Sequência Global).")
        
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
            
            # Mapeamento do Sequence_Global para o ID_UNICO para aplicar o *
            mapa_volumosos = df_mapa[df_mapa[COLUNA_SEQUENCE_GLOBAL].isin(st.session_state['volumoso_sequence_global_ids'])][COLUNA_ID_UNICO].tolist()
            
            # Aplica o * nos IDs ÚNICOS mapeados como volumosos
            for id_volumoso_unico in mapa_volumosos:
                str_id_volumoso = str(id_volumoso_unico)
                
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
        st.session_state['sheet_name_pos'] = sheet_name 
    
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
        
        if st.button("✨ Mostrar APENAS Pacotes Volumosos (*)", key="btn_filtro_volumosos"):
            
            df_volumosos = df_extracted[
                df_extracted['Ordem ID'].astype(str).str.contains(r'\*', regex=True, na=False)
            ].copy() 
            
            if not df_volumosos.empty:
                st.success(f"Filtro aplicado! Encontrados **{len(df_volumosos)}** paradas com itens volumosos.")

                copia_data_volumosos = '\n'.join(df_volumosos['Lista de Impressão'].astype(str).tolist())
                
                st.subheader("Lista de Volumosos Filtrada (Sequência do Circuit)")

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
