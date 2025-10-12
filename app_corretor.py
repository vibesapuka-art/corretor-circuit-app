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

# --- CSS para alinhar texto à esquerda (Corrige a centralização do Streamlit) ---
st.markdown("""
<style>
.stTextArea [data-baseweb="base-input"] {
    text-align: left;
    font-family: monospace;
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


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e o agrupamento.
    A coluna Sequence já estará ajustada com '*' se necessário.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None

    df = df_entrada.copy()
    
    # Cria uma coluna numérica temporária para a ordenação (ignorando o *)
    # st.session_state['volumoso_ids'] é uma lista de IDs.
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace('*', '', regex=False)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(0).astype(int)

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
            grupo_matches = [
                match[0] for match in matches 
                if match[1] >= limite_similaridade
            ]
            
            df_grupo = df[df['Endereco_Limpo'].isin(grupo_matches)]
            endereco_oficial_original = df_grupo[COLUNA_ENDERECO].mode()[0]
            
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
        # Agrupa as sequências (que já contêm o *)
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x)))), 
        Total_Pacotes=('Sequence_Num', 'count'), # Usa a contagem da coluna numérica (ignora o *)
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        Bairro_Agrupado=('Bairro', lambda x: x.mode()[0]),
        Zipcode_Agrupado=('Zipcode/Postal code', lambda x: x.mode()[0]),
        
        # Captura o menor número de sequência original (sem *) para ordenação
        Min_Sequence=('Sequence_Num', 'min') 
        
    ).reset_index()

    # 5. ORDENAÇÃO: Ordena o DataFrame pelo menor número de sequência.
    df_agrupado = df_agrupado.sort_values(by='Min_Sequence').reset_index(drop=True)
    
    # 6. Formatação do DF para o CIRCUIT 
    endereco_completo_circuit = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro_Agrupado']
    )
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(str) + 
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
    
    return df_circuit


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
    
    # 2. Separar a coluna Notes: Parte antes do ';' é o Order ID
    df[coluna_notes_lower] = df[coluna_notes_lower].str.strip('"')
    
    # Divide a coluna na primeira ocorrência de ';'
    df_split = df[coluna_notes_lower].str.split(';', n=1, expand=True)
    df['Ordem ID'] = df_split[0].str.strip()
    df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
    
    
    # 3. Formatação Final da Tabela (APENAS ID e ANOTAÇÕES)
    colunas_finais = ['Ordem ID', 'Anotações Completas']
    
    df_final = df[colunas_finais]
    
    return df_final


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
    st.caption("Esta etapa corrige erros de digitação, marca volumes e agrupa pedidos.")

    # Inicializa o estado para armazenar o DataFrame e as ordens marcadas
    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = None
    if 'volumoso_ids' not in st.session_state:
        st.session_state['volumoso_ids'] = []
    
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
        
        df_display = st.session_state['df_original'].copy()
        df_display['Exibir'] = (
            '#' + df_display[COLUNA_SEQUENCE].astype(str) + 
            ' - ' + df_display[COLUNA_ENDERECO].astype(str)
        )
        
        ordens_originais = df_display[COLUNA_SEQUENCE].unique()
        
        # A caixa de seleção de múltiplas escolhas permite ao usuário selecionar as ordens volumosas
        st.session_state['volumoso_ids'] = st.multiselect(
            'Selecione as Ordens de Serviço (do original) que são Volumosas (Serão marcadas com *):',
            options=ordens_originais,
            default=st.session_state['volumoso_ids'],
            format_func=lambda x: df_display[df_display[COLUNA_SEQUENCE] == x]['Exibir'].iloc[0]
        )
        
        st.markdown(f"**{len(st.session_state['volumoso_ids'])}** pacotes marcados como volumosos.")
        st.caption("A ordem de exibição é apenas para facilitar a marcação.")
        
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
            
            # Converte a coluna para string para aplicar o *
            df_para_processar[COLUNA_SEQUENCE] = df_para_processar[COLUNA_SEQUENCE].astype(str)
            
            for id_volumoso in st.session_state['volumoso_ids']:
                # Encontra a linha da ordem e adiciona o *
                df_para_processar.loc[
                    df_para_processar[COLUNA_SEQUENCE] == str(id_volumoso), 
                    COLUNA_SEQUENCE
                ] = str(id_volumoso) + '*'

            # 2. Iniciar o processamento e agrupamento
            df_circuit = processar_e_corrigir_dados(df_para_processar, limite_similaridade_ajustado)
            
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
                
                # --- SAÍDA PARA CIRCUIT (ROTEIRIZAÇÃO) ---
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

    # Limpa a sessão se o arquivo for removido
    elif uploaded_file_pre is None:
        st.session_state['df_original'] = None
        st.session_state['volumoso_ids'] = []
        

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
            
            # --- CORREÇÃO ESSENCIAL: PADRONIZAÇÃO DE COLUNAS ---
            df_input_pos.columns = df_input_pos.columns.str.strip() 
            df_input_pos.columns = df_input_pos.columns.str.lower()
            # ---------------------------------------------------

            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_input_pos)}** registros.")
            
            # Processa os dados
            df_final_pos = processar_rota_para_impressao(df_input_pos)
            
            if df_final_pos is not None and not df_final_pos.empty:
                st.markdown("---")
                st.subheader("2.2 Resultado Final (Ordem ID e Anotações)")
                st.caption("A tabela abaixo é apenas para visualização. Use a área de texto para cópia rápida.")
                
                # Exibe a tabela
                st.dataframe(df_final_pos, use_container_width=True)

                # --- LÓGICA DE COPIA PERSONALIZADA (ID - ANOTAÇÕES) ---
                
                # Combina as duas colunas com o separador " - "
                df_final_pos['Linha Impressão'] = (
                    df_final_pos['Ordem ID'].astype(str) + 
                    ' - ' + 
                    df_final_pos['Anotações Completas'].astype(str)
                )
                
                # Converte para string sem cabeçalho e sem índice, com quebras de linha
                copia_data = df_final_pos['Linha Impressão'].to_string(index=False, header=False)
                
                st.markdown("### 2.3 Copiar para a Área de Transferência (ID - Anotações)")
                
                st.info("Para copiar: **Selecione todo o texto** abaixo (Ctrl+A / Cmd+A) e pressione **Ctrl+C / Cmd+C**.")
                
                # Área de texto para visualização e cópia
                st.text_area(
                    "Conteúdo da Lista de Impressão (Alinhado à Esquerda):", 
                    copia_data, 
                    height=300
                )

                # Download como Excel 
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_final_pos[['Ordem ID', 'Anotações Completas']].to_excel(writer, index=False, sheet_name='Lista Impressao')
                buffer.seek(0)
                
                st.download_button(
                    label="📥 Baixar Lista Limpa (Excel)",
                    data=buffer,
                    file_name="Lista_Ordem_Impressao.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_list"
                )

        except KeyError as ke:
             # Captura erros de coluna ou aba
            if "Table 3" in str(ke):
                st.error(f"Erro de Aba: A aba **'{sheet_name}'** não foi encontrada no arquivo Excel. Verifique o nome da aba.")
            elif 'notes' in str(ke):
                 st.error(f"Erro de Coluna: A coluna 'Notes' não foi encontrada. Verifique se o arquivo da rota está correto.")
            else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o arquivo da rota (PDF convertido) está no formato CSV ou Excel. Erro: {e}")
