import streamlit as st
import pandas as pd
import uuid
from io import StringIO
import os # Mantido por boa prática, embora não use mais o módulo os para o DB

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Otimizador de Rotas com Correção Manual")

# Variáveis globais simuladas
APP_ID = "rota_flow_simulacao" 
app_id = APP_ID

# --- 1. Inicialização do Estado de Sessão (Substitui o DB) ---
def initialize_session_state():
    """
    Inicializa o dicionário de correções no estado de sessão do Streamlit.
    Este dicionário só persiste enquanto o aplicativo estiver aberto.
    """
    if 'fixed_coords' not in st.session_state:
        # { "Endereço Exato (Destination Address)": {"lat": 12.345, "lng": -56.789} }
        st.session_state['fixed_coords'] = {}
        st.info("Dicionário de correções iniciado. Os dados não serão salvos permanentemente.")

# --- 2. Funções de Manipulação do Dicionário Fixo (Usando Session State) ---
# NOTE: Não há mais funções save/get para o banco de dados.
# O dicionário st.session_state['fixed_coords'] é a fonte de verdade local.

# --- 3. Função de Processamento de Dados (Foco na Correção) ---
def process_data(df: pd.DataFrame, fixed_coords_dict: dict):
    """
    Processa o DataFrame, aplica correções manuais usando 'Destination Address' como chave.
    """
    st.header("1. Detalhes do Processamento")

    # 1. Validação de Colunas Mínimas
    required_cols = ['Destination Address', 'Latitude', 'Longitude']
    if not all(col in df.columns for col in required_cols):
        st.error(f"O arquivo deve conter as colunas: {required_cols}.")
        st.info(f"Colunas encontradas: {df.columns.tolist()}")
        return None

    # Garantir que Lat/Lng sejam numéricos (força coercão para NaN se houver erro)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    st.subheader("APLICAÇÃO DE CORREÇÕES MANUAIS (Ponto Crítico)")
    st.warning("A correspondência é feita usando a coluna **'Destination Address'** como chave EXATA (caracter por caracter).")

    # --- LÓGICA DE CORREÇÃO ---
    
    # 1. Mapeamento: Aplica o dicionário de correção na coluna de endereço.
    df[['Fixed_Lat', 'Fixed_Lng']] = df['Destination Address'].map(fixed_coords_dict).apply(
        lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series([None, None])
    )
    
    # 2. Priorizar a correção manual sobre os valores existentes (combine_first)
    df['Latitude'] = df['Fixed_Lat'].combine_first(df['Latitude'])
    df['Longitude'] = df['Fixed_Lng'].combine_first(df['Longitude'])
    
    # --- FIM DA LÓGICA DE CORREÇÃO ---
    
    # Contar quantas correções foram aplicadas
    correcoes_aplicadas = df['Fixed_Lat'].notnull().sum()
    st.success(f"✅ {correcoes_aplicadas} correções manuais do Dicionário Fixo aplicadas com sucesso.")

    # 3. Identificar pontos sem Lat/Lng válida
    invalid_rows = df[df['Latitude'].isna() | df['Longitude'].isna()]
    if not invalid_rows.empty:
        st.error(f"🚨 {len(invalid_rows)} registros permanecem com Lat/Lng inválida após a correção.")
        st.dataframe(invalid_rows[['Destination Address', 'Latitude', 'Longitude']].head(5))
        st.info("Estes endereços precisam ser corrigidos manualmente na Aba 2 para serem roteirizados.")
    
    # 4. Preparar o DataFrame para Download/Próxima Etapa
    df = df.drop(columns=['Fixed_Lat', 'Fixed_Lng'])
    
    if 'Routing_ID' not in df.columns:
        df['Routing_ID'] = [str(uuid.uuid4()) for _ in range(len(df))]

    st.subheader("Saída Final")
    st.dataframe(df.head())
    
    # Botão de Download
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Dados Processados (CSV)",
        data=csv,
        file_name='dados_processados_com_correcao.csv',
        mime='text/csv',
    )
    
    return df

# --- 4. Interface do Streamlit (Função Principal) ---

def main():
    st.title("🗺️ Pré-Roteirizador & Dicionário Fixo")
    
    # Inicializa o dicionário de correções (sempre executa no início)
    initialize_session_state()

    # O código começa a renderizar a interface aqui.
    tab1, tab2 = st.tabs(["1. Processar Planilha (Correção)", "2. Gerenciar Dicionário Fixo"])

    # --- TAB 1: Processamento ---
    with tab1:
        st.header("1. Upload e Processamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo CSV/Excel para processar.",
            type=["csv", "xlsx"]
        )
        st.info("O arquivo deve conter as colunas: **'Destination Address'**, **'Latitude'** e **'Longitude'**.")

        if uploaded_file:
            try:
                # Determinar o tipo de arquivo
                if uploaded_file.name.endswith('.csv'):
                    data = uploaded_file.getvalue().decode('utf-8')
                    df = pd.read_csv(StringIO(data))
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Arquivo '{uploaded_file.name}' lido com sucesso. Total de {len(df)} linhas.")

                # Executa o processamento
                process_data(df.copy(), st.session_state['fixed_coords'])
                
            except Exception as e:
                st.error(f"Erro ao ler ou processar o arquivo. Verifique se o formato e as colunas estão corretos. Erro: {e}")

    # --- TAB 2: Gerenciamento do Dicionário Fixo ---
    with tab2:
        st.header("2. Gerenciar Dicionário Fixo de Lat/Lng")
        
        # 2.1 Adicionar/Atualizar Correção
        st.subheader("2.1 Adicionar Nova Correção")
        st.info("Use o conteúdo **EXATO** da coluna 'Destination Address' como chave para garantir a correspondência.")
        with st.form("form_add_correction"):
            new_address = st.text_input("Endereço Exato (Valor da coluna 'Destination Address')", placeholder="Ex: Rua Tal, 123, Bairro")
            new_lat = st.text_input("Latitude Corrigida", placeholder="Use ponto '.' como separador. Ex: -23.5505")
            new_lng = st.text_input("Longitude Corrigida", placeholder="Use ponto '.' como separador. Ex: -46.6333")
            submitted = st.form_submit_button("Adicionar Correção") # Removido 'e Salvar' pois não há DB

            if submitted:
                if not new_address:
                    st.warning("Preencha o Endereço para adicionar a correção.")
                else:
                    try:
                        # 1. Limpeza: Remove espaços e substitui vírgula por ponto (para flexibilidade)
                        cleaned_lat = new_lat.strip().replace(',', '.')
                        cleaned_lng = new_lng.strip().replace(',', '.')

                        # 2. Conversão: Tenta converter para float
                        lat_val = float(cleaned_lat)
                        lng_val = float(cleaned_lng)
                        
                        # 3. Sucesso: Adiciona/Atualiza o dicionário no estado de sessão
                        st.session_state['fixed_coords'][new_address] = {'lat': lat_val, 'lng': lng_val}
                        
                        st.success(f"Correção salva LOCALMENTE com sucesso para: '{new_address}'")
                        st.experimental_rerun() # Recarrega para atualizar a visualização da tabela

                    except ValueError:
                        # 4. Falha: Exibe a mensagem de erro
                        st.error("Latitude e Longitude devem ser números válidos. Verifique se usou ponto (.) ou se há caracteres estranhos.")

        # 2.2 Visualizar/Excluir Dicionário
        st.subheader("2.2 Visualizar Dicionário Atual (Local)")
        st.info("Este dicionário é LOCAL e será perdido se você fechar o navegador.")
        
        fixed_coords = st.session_state['fixed_coords']
        if fixed_coords:
            # Converte para DataFrame para visualização fácil
            data_list = []
            for address, coords in fixed_coords.items():
                data_list.append({
                    "Endereço (Chave Exata - 'Destination Address')": address,
                    "Latitude Fixa": coords['lat'],
                    "Longitude Fixa": coords['lng'],
                })
            
            df_fixed = pd.DataFrame(data_list)
            st.dataframe(df_fixed, use_container_width=True, height=300)

            # Opção de Excluir
            address_to_delete = st.selectbox(
                "Selecione o endereço para remover (opcional):",
                [""] + list(fixed_coords.keys())
            )
            
            if st.button("Remover Correção Selecionada"):
                if address_to_delete and address_to_delete in fixed_coords:
                    del st.session_state['fixed_coords'][address_to_delete]
                    st.success(f"Correção para '{address_to_delete}' removida com sucesso. ")
                    st.experimental_rerun()
                elif address_to_delete:
                    st.warning("Endereço não encontrado no dicionário.")
        else:
            st.info("O dicionário de correções fixas está vazio.")
            
if __name__ == '__main__':
    main()
