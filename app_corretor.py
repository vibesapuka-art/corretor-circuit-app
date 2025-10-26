import streamlit as st
import pandas as pd
import json
import uuid
import os 

# Importação direta do cliente Google Cloud Firestore
from google.cloud import firestore
from google.cloud.firestore_v1.base_client import BaseClient
from io import StringIO

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Otimizador de Rotas com Correção Manual")

# Variáveis globais simuladas
APP_ID = "rota_flow_simulacao" 

# Configuração MOCK (apenas o ID do projeto)
MOCK_FIREBASE_CONFIG = {
    "projectId": "mock-project-id" 
}

app_id = APP_ID

# --- 1. Inicialização do Firestore ---
@st.cache_resource
def initialize_firestore():
    """
    Inicializa o cliente Firestore, definindo a variável de ambiente GOOGLE_CLOUD_PROJECT 
    para forçar o reconhecimento do Project ID.
    """
    # Se já estiver no estado da sessão (cache), retorna.
    if 'db_client' in st.session_state and isinstance(st.session_state['db_client'], BaseClient):
        return st.session_state['db_client']
    
    try:
        # Define a variável de ambiente GOOGLE_CLOUD_PROJECT
        os.environ['GOOGLE_CLOUD_PROJECT'] = MOCK_FIREBASE_CONFIG['projectId']
        
        # Inicializa o cliente Firestore
        db = firestore.Client()
        st.session_state['db_client'] = db # Armazena no estado da sessão
        return db
        
    except Exception as e:
        # Retorna None em caso de falha de inicialização
        st.error(f"Erro ao inicializar o Firestore: {e}. O aplicativo não pode funcionar sem a conexão com o banco de dados.")
        return None

# --- 2. Funções de Manipulação do Dicionário Fixo ---
def get_fixed_coords(db: BaseClient, app_id: str):
    """Carrega o dicionário de Lat/Lng fixas do Firestore."""
    if not db:
        return {} 
    try:
        doc_ref = db.collection('artifacts').document(app_id).collection('public').document('correcoes_fixas')
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return data.get('fixed_coords', {})
        return {}
    except Exception as e:
        # AQUI garantimos que, se falhar, o app não trave.
        st.error(f"⚠️ Erro ao tentar carregar o dicionário fixo do Banco de Dados: {e}. Usando dicionário vazio local.")
        return {}

def save_fixed_coords(db: BaseClient, app_id: str, fixed_coords_dict: dict):
    """Salva o dicionário de Lat/Lng fixas no Firestore."""
    if not db:
        st.error("Falha ao salvar. Conexão com o Firestore indisponível.")
        return False
    try:
        doc_ref = db.collection('artifacts').document(app_id).collection('public').document('correcoes_fixas')
        doc_ref.set({'fixed_coords': fixed_coords_dict})
        return True
    except Exception as e:
        st.error(f"Erro ao salvar o dicionário fixo (Banco de Dados Indisponível): {e}")
        return False

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
    
    # 1. Mapeamento
    df[['Fixed_Lat', 'Fixed_Lng']] = df['Destination Address'].map(fixed_coords_dict).apply(
        lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series([None, None])
    )
    
    # 2. Priorizar a correção manual sobre os valores existentes
    df['Latitude'] = df['Fixed_Lat'].combine_first(df['Latitude'])
    df['Longitude'] = df['Fixed_Lng'].combine_first(df['Longitude'])
    
    # --- FIM DA LÓGICA DE CORREÇÃO ---
    
    # Contar quantas correções foram aplicadas
    correcoes_aplicadas = df['Fixed_Lat'].notnull().sum()
    st.success(f"✅ {correcoes_aplicadas} correções manuais do Dicionário Fixo aplicadas com sucesso.")

    # 3. Identificar pontos sem Lat/Lng válida (que não foram corrigidos e vieram nulos/inválidos)
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
    
    # 1. Tenta inicializar o DB. Se falhar, exibe a mensagem de erro e sai.
    db_instance = initialize_firestore() 
    if db_instance is None:
        return # Sai se a inicialização falhou

    # 2. Se o DB estiver pronto, carrega o dicionário para o estado da sessão (se ainda não estiver lá)
    if 'fixed_coords' not in st.session_state:
        # Tentativa de carregar o dicionário. Se falhar, apenas usa um dicionário vazio.
        st.session_state['fixed_coords'] = get_fixed_coords(db_instance, app_id)

    # --- Ponto de Verificação CRÍTICO ---
    # Se você vir esta mensagem, o app está carregando 100% da interface.
    # Caso contrário, o crash é na linha acima.
    st.info(f"Conexão OK. Dicionário carregado com {len(st.session_state['fixed_coords'])} correções.")

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
                if uploaded_file.name.endswith('.csv'):
                    data = uploaded_file.getvalue().decode('utf-8')
                    df = pd.read_csv(StringIO(data))
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"Arquivo '{uploaded_file.name}' lido com sucesso. Total de {len(df)} linhas.")

                process_data(df.copy(), st.session_state['fixed_coords'])
                
            except Exception as e:
                st.error(f"Erro ao ler ou processar o arquivo. Verifique se o formato e as colunas estão corretos. Erro: {e}")

    # --- TAB 2: Gerenciamento do Dicionário Fixo ---
    with tab2:
        st.header("2. Gerenciar Dicionário Fixo de Lat/Lng")
        
        st.subheader("2.1 Adicionar Nova Correção")
        st.info("Use o conteúdo **EXATO** da coluna 'Destination Address' como chave para garantir a correspondência.")
        with st.form("form_add_correction"):
            new_address = st.text_input("Endereço Exato (Valor da coluna 'Destination Address')", placeholder="Ex: Rua Tal, 123, Bairro")
            new_lat = st.text_input("Latitude Corrigida", placeholder="Use ponto '.' como separador. Ex: -23.5505")
            new_lng = st.text_input("Longitude Corrigida", placeholder="Use ponto '.' como separador. Ex: -46.6333")
            submitted = st.form_submit_button("Adicionar Correção e Salvar")

            if submitted:
                if not new_address:
                    st.warning("Preencha o Endereço para adicionar a correção.")

                try:
                    cleaned_lat = new_lat.strip().replace(',', '.')
                    cleaned_lng = new_lng.strip().replace(',', '.')

                    lat_val = float(cleaned_lat)
                    lng_val = float(cleaned_lng)
                    
                    st.session_state['fixed_coords'][new_address] = {'lat': lat_val, 'lng': lng_val}
                    
                    if save_fixed_coords(db_instance, app_id, st.session_state['fixed_coords']):
                        st.success(f"Correção salva com sucesso para: '{new_address}'")
                    else:
                        st.error("Falha ao salvar a correção no banco de dados.")

                except ValueError:
                    st.error("Latitude e Longitude devem ser números válidos. Verifique se usou ponto (.) ou se há caracteres estranhos.")

        # 2.2 Visualizar/Excluir Dicionário
        st.subheader("2.2 Visualizar Dicionário Atual")
        
        fixed_coords = st.session_state['fixed_coords']
        if fixed_coords:
            data_list = []
            for address, coords in fixed_coords.items():
                data_list.append({
                    "Endereço (Chave Exata - 'Destination Address')": address,
                    "Latitude Fixa": coords['lat'],
                    "Longitude Fixa": coords['lng'],
                })
            
            df_fixed = pd.DataFrame(data_list)
            st.dataframe(df_fixed, use_container_width=True, height=300)

            address_to_delete = st.selectbox(
                "Selecione o endereço para remover (opcional):",
                [""] + list(fixed_coords.keys())
            )
            
            if st.button("Remover Correção Selecionada"):
                if address_to_delete and address_to_delete in fixed_coords:
                    del st.session_state['fixed_coords'][address_to_delete]
                    if save_fixed_coords(db_instance, app_id, st.session_state['fixed_coords']):
                        st.success(f"Correção para '{address_to_delete}' removida com sucesso. Recarregue a página para atualizar a visualização.")
                        st.rerun()
                    else:
                        st.error("Falha ao remover a correção no banco de dados.")
                elif address_to_delete:
                    st.warning("Endereço não encontrado no dicionário.")
        else:
            st.info("O dicionário de correções fixas está vazio.")
            
if __name__ == '__main__':
    main()
