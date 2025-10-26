import streamlit as st
import pandas as pd
import json
import uuid
from firebase_admin import credentials, initialize_app, firestore
from google.cloud.firestore_v1.base_client import BaseClient
from io import StringIO
import os # Importação adicionada para usar variáveis de ambiente

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Otimizador de Rotas com Correção Manual")

# Variáveis globais simuladas (IMPORTANTE: No ambiente real, use as variáveis __ fornecidas)
APP_ID = "rota_flow_simulacao" # ID fixo para o app de roteirização
# Configuração MOCK simplificada. No Canvas real, o SDK de Admin não é usado diretamente
# O Firebase Config para inicialização básica do SDK Web (Client SDK) é o que seria usado no browser.
MOCK_FIREBASE_CONFIG = {
    "apiKey": "mock-api-key",
    "authDomain": "mock-project-id.firebaseapp.com",
    "projectId": "mock-project-id", # Chave essencial
    "storageBucket": "mock-project-id.appspot.com",
    "messagingSenderId": "mock-sender-id",
    "appId": "mock-app-id"
}

# Use variáveis globais ou mocks
app_id = APP_ID

# --- 1. Inicialização do Firebase e Firestore ---
@st.cache_resource
def initialize_firestore():
    """
    Inicializa o Firebase e o cliente Firestore.
    
    Adicionado 'project=...' na chamada a firestore.client() para resolver o erro de Project ID.
    """
    if 'db' in st.session_state and isinstance(st.session_state['db'], BaseClient):
        return st.session_state['db']
    
    try:
        if not initialize_app():
            # Cria uma credencial básica que será aceita pelo initialize_app
            cred = credentials.Certificate(MOCK_FIREBASE_CONFIG)
            initialize_app(cred)

        # CORREÇÃO CRÍTICA: Passa o projectId explicitamente
        db = firestore.client(project=MOCK_FIREBASE_CONFIG['projectId'])
        st.session_state['db'] = db
        return db
        
    except Exception as e:
        # Se falhar, exibe uma mensagem genérica sem vazar detalhes da chave
        st.error(f"Erro ao inicializar o Firestore: {e}. Verifique se a variável MOCK_FIREBASE_CONFIG está correta para este ambiente.")
        return None

db: BaseClient = initialize_firestore()

# --- 2. Funções de Manipulação do Dicionário Fixo ---
def get_fixed_coords(db: BaseClient, app_id: str):
    """Carrega o dicionário de Lat/Lng fixas do Firestore."""
    if not db:
        return {}
    try:
        # Caminho onde o dicionário é salvo
        doc_ref = db.collection('artifacts').document(app_id).collection('public').document('correcoes_fixas')
        doc = doc_ref.get()
        if doc.exists:
            # O dicionário é armazenado como um campo dentro do documento
            data = doc.to_dict()
            return data.get('fixed_coords', {})
        return {}
    except Exception as e:
        st.error(f"Erro ao carregar o dicionário fixo: {e}")
        return {}

def save_fixed_coords(db: BaseClient, app_id: str, fixed_coords_dict: dict):
    """Salva o dicionário de Lat/Lng fixas no Firestore."""
    if not db:
        return False
    try:
        # Caminho onde o dicionário é salvo
        doc_ref = db.collection('artifacts').document(app_id).collection('public').document('correcoes_fixas')
        # Salva o dicionário completo em um único campo 'fixed_coords'
        doc_ref.set({'fixed_coords': fixed_coords_dict})
        return True
    except Exception as e:
        st.error(f"Erro ao salvar o dicionário fixo: {e}")
        return False

# --- 3. Função de Processamento de Dados (Foco na Correção) ---
def process_data(df: pd.DataFrame, fixed_coords_dict: dict):
    """
    Processa o DataFrame, aplica correções manuais usando 'Destination Address' como chave.
    """
    st.header("1. Detalhes do Processamento")

    # 1. Validação de Colunas Mínimas
    # REQUIRED_COL deve ser as colunas que você PRECISA para o processamento de correção e geolocalização.
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

    # --- LÓGICA DE CORREÇÃO ATUALIZADA ---
    
    # 1. Mapeamento
    # O PANDAS procura o valor da coluna 'Destination Address' como chave em fixed_coords_dict.
    df[['Fixed_Lat', 'Fixed_Lng']] = df['Destination Address'].map(fixed_coords_dict).apply(
        # Aplica uma função para extrair 'lat' e 'lng' se o mapeamento for bem-sucedido (for um dicionário)
        lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series([None, None])
    )
    
    # 2. Priorizar a correção manual sobre os valores existentes
    # .combine_first() usa o valor da primeira série (Fixed) se não for nulo, caso contrário, usa o valor da segunda (Original).
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
    
    # Remover colunas temporárias
    df = df.drop(columns=['Fixed_Lat', 'Fixed_Lng'])
    
    # Adicionar coluna de ID de Roteirização (se não existir)
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

# --- 4. Interface do Streamlit ---

if 'fixed_coords' not in st.session_state:
    st.session_state['fixed_coords'] = get_fixed_coords(db, app_id)

def main():
    # Verifica se a inicialização do banco de dados foi bem-sucedida antes de continuar
    if db is None:
        st.warning("O aplicativo não pode funcionar sem a conexão com o Firestore.")
        return

    st.title("🗺️ Pré-Roteirizador & Dicionário Fixo")

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
                    # Tenta ler o CSV, usando inferência de encoding
                    data = uploaded_file.getvalue().decode('utf-8')
                    df = pd.read_csv(StringIO(data))
                else:
                    # Ler Excel (apenas a primeira aba)
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
            submitted = st.form_submit_button("Adicionar Correção e Salvar")

            if submitted:
                # --- NOVO BLOCO DE LIMPEZA E VALIDAÇÃO ---
                if not new_address:
                    st.warning("Preencha o Endereço para adicionar a correção.")

                try:
                    # 1. Limpeza: Remove espaços e substitui vírgula por ponto
                    cleaned_lat = new_lat.strip().replace(',', '.')
                    cleaned_lng = new_lng.strip().replace(',', '.')

                    # 2. Conversão: Tenta converter para float
                    lat_val = float(cleaned_lat)
                    lng_val = float(cleaned_lng)
                    
                    # 3. Sucesso: Adiciona/Atualiza o dicionário
                    st.session_state['fixed_coords'][new_address] = {'lat': lat_val, 'lng': lng_val}
                    
                    # Salva no Firestore
                    if save_fixed_coords(db, app_id, st.session_state['fixed_coords']):
                        st.success(f"Correção salva com sucesso para: '{new_address}'")
                    else:
                        st.error("Falha ao salvar a correção no banco de dados.")

                except ValueError:
                    # 4. Falha: Exibe a mensagem de erro
                    st.error("Latitude e Longitude devem ser números válidos. Verifique se usou ponto (.) ou se há caracteres estranhos.")
                # --- FIM DO NOVO BLOCO ---

        # 2.2 Visualizar/Excluir Dicionário
        st.subheader("2.2 Visualizar Dicionário Atual")
        
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
                    if save_fixed_coords(db, app_id, st.session_state['fixed_coords']):
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
