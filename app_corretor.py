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
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None

    df = df_entrada.copy()

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
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x)))),
        Total_Pacotes=(COLUNA_SEQUENCE, 'count'),
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        Bairro_Agrupado=('Bairro', lambda x: x.mode()[0]),
        Zipcode_Agrupado=('Zipcode/Postal code', lambda x: x.mode()[0])
        
    ).reset_index()

    # 5. Formatação do DF para o CIRCUIT 
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
    
    NOTA: Esta função assume que as colunas de df_input JÁ FORAM PADRONIZADAS 
    para minúsculas antes de serem passadas.
    """
    coluna_notes_lower = 'notes'
    coluna_address_lower = 'address'

    # 1. Verificar se as colunas essenciais existem (em minúsculas)
    if coluna_notes_lower not in df_input.columns:
        # Este erro é capturado e reformatado no bloco try/except da interface
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
    
    
    # 3. Formatação Final da Tabela
    colunas_finais = ['Ordem ID']
    
    # Se a coluna de endereço existir, renomeie-a e inclua
    if coluna_address_lower in df_input.columns:
        df = df.rename(columns={coluna_address_lower: 'Endereço'})
        colunas_finais.append('Endereço')
    else:
        st.warning("A coluna de Endereço não foi encontrada para inclusão na lista de impressão.")
    
    colunas_finais.append('Anotações Completas')
    
    df_final = df[colunas_finais]
    
    return df_final


# ===============================================
# INTERFACE PRINCIPAL
# ===============================================

st.title("🗺️ Flow Completo Circuit (Pré e Pós-Roteirização)")

# CRIAÇÃO DAS ABAS (ESSENCIAL PARA EVITAR NameError)
tab1, tab2 = st.tabs(["🚀 Pré-Roteirização (Importação)", "📋 Pós-Roteirização (Impressão/Cópia)"])


# ----------------------------------------------------------------------------------
# ABA 1: PRÉ-ROTEIRIZAÇÃO (CORREÇÃO E IMPORTAÇÃO)
# ----------------------------------------------------------------------------------

with tab1:
    st.header("1. Gerar Arquivo para Importar no Circuit")
    st.caption("Esta etapa corrige erros de digitação e agrupa pedidos no mesmo endereço.")

    st.markdown("---")
    st.subheader("Configurações:")
    limite_similaridade_ajustado = st.slider(
        'Ajuste a Precisão do Corretor (Fuzzy Matching):',
        min_value=80,
        max_value=100,
        value=100, 
        step=1,
        help="Use 100% para garantir que endereços na mesma rua com números diferentes não sejam agrupados (recomendado)."
    )
    st.info(f"O limite de similaridade está em **{limite_similaridade_ajustado}%**.")


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
            
            st.success(f"Arquivo '{uploaded_file_pre.name}' carregado! Total de **{len(df_input_pre)}** registros.")
            
            st.markdown("---")
            st.subheader("1.2 Processar")
            
            if st.button("🚀 Iniciar Corretor e Agrupamento", key="btn_pre"):
                df_circuit = processar_e_corrigir_dados(df_input_pre, limite_similaridade_ajustado)
                
                if df_circuit is not None:
                    st.markdown("---")
                    st.header("✅ Resultado Concluído!")
                    
                    total_entradas = len(df_input_pre)
                    total_agrupados = len(df_circuit)
                    
                    st.metric(
                        label="Endereços Únicos Agrupados",
                        value=total_agrupados,
                        delta=f"-{total_entradas - total_agrupados} agrupados"
                    )
                    
                    # --- SAÍDA PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                    st.subheader("Arquivo para Roteirização (Circuit)")
                    st.dataframe(df_circuit.head(5), use_container_width=True)
                    
                    # Download Circuit
                    buffer_circuit = io.BytesIO()
                    with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                        df_circuit.to_excel(writer, index=False, sheet_name='Circuit Import')
                    buffer_circuit.seek(0)
                    
                    st.download_button(
                        label="📥 Baixar ARQUIVO PARA CIRCUIT",
                        data=buffer_circuit,
                        file_name="Circuit_Import_FINAL.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_circuit"
                    )

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato e as colunas. Erro: {e}")


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
            # Converte todos os nomes de colunas para minúsculas e remove espaços extras
            df_input_pos.columns = df_input_pos.columns.str.strip() 
            df_input_pos.columns = df_input_pos.columns.str.lower()
            # ---------------------------------------------------

            st.success(f"Arquivo '{uploaded_file_pos.name}' carregado! Total de **{len(df_input_pos)}** registros.")
            
            # Processa os dados
            df_final_pos = processar_rota_para_impressao(df_input_pos)
            
            if df_final_pos is not None and not df_final_pos.empty:
                st.markdown("---")
                st.subheader("2.2 Resultado Final (Ordem ID + Anotações)")
                
                # Exibe a tabela
                st.dataframe(df_final_pos, use_container_width=True)

                # Opção de Copiar para a Área de Transferência
                csv_data = df_final_pos.to_csv(index=False, header=False, sep='\t')
                
                st.markdown("### 2.3 Copiar para a Área de Transferência")
                st.info("Para copiar para o Excel/Word/etc., selecione todo o texto abaixo (Ctrl+A) e pressione Ctrl+C.")
                
                st.text_area(
                    "Conteúdo da Tabela (Separação por Tabulação):", 
                    csv_data, 
                    height=300
                )

                # Download como Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_final_pos.to_excel(writer, index=False, sheet_name='Lista Impressao')
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
                 st.error(f"Erro de Coluna: A coluna 'Notes' não foi encontrada após a padronização. Verifique se o arquivo da rota está correto.")
            else:
                 st.error(f"Ocorreu um erro de coluna ou formato. Erro: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se o arquivo da rota (PDF convertido) está no formato CSV ou Excel. Erro: {e}")
