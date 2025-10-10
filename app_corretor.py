import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os
from geopy.geocoders import Nominatim # <--- NOVA BIBLIOTECA
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# --- Configurações da Página ---
st.set_page_config(
    page_title="Corretor de Endereços Circuit (Geocodificado)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configurações Principais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'

# Inicializa o geocodificador UMA VEZ
# O Nominatim é gratuito, mas é lento e pode ter limites de uso por hora.
@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="circuit_address_corrector_app")

geolocator = get_geolocator()

def geocode_address(full_address, geolocator):
    """Tenta obter Latitude e Longitude para um endereço corrigido."""
    try:
        location = geolocator.geocode(full_address, timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except GeocoderTimedOut:
        return None, None # Tenta novamente ou falha silenciosamente
    except GeocoderServiceError:
        return None, None
    except Exception:
        return None, None


def limpar_endereco(endereco):
    """Normaliza o texto do endereço para melhor comparação."""
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    endereco = re.sub(r'[^\w\s]', '', endereco)
    endereco = re.sub(r'\s+', ' ', endereco)
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    return endereco


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e a NOVA GEROCODIFICAÇÃO.
    """
    colunas_essenciais = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code']
    for col in colunas_essenciais:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada na sua planilha.")
            return None, None

    df = df_entrada.copy()

    # 1. Limpeza e Normalização (Fuzzy Matching)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching para Agrupamento (Omitido para brevidade, mas está no código final)
    # ... Lógica do Fuzzy Matching anterior ...
    
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

    # 4. Agrupamento (Mantém Bairro, City, Zipcode)
    colunas_agrupamento = ['Endereco_Corrigido', 'Bairro', 'City', 'Zipcode/Postal code']
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x)))),
        Total_Pacotes=(COLUNA_SEQUENCE, 'count'),
        # MANTÉM LAT/LON ORIGINAIS POR ENQUANTO
        Latitude_Original=(COLUNA_LATITUDE, 'first'),
        Longitude_Original=(COLUNA_LONGITUDE, 'first')
    ).reset_index()

    # 5. --- GERAÇÃO DO NOVO ENDEREÇO E GEOCODIFICAÇÃO ---
    st.subheader("⚠️ Geocodificação em Andamento...")
    st.caption("Esta etapa pode ser lenta, pois consulta a localização exata de cada endereço único.")
    
    df_agrupado['Endereco_Completo_Padrao'] = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro'] + ', ' +
        df_agrupado['City']
    )
    
    # Prepara novas colunas
    df_agrupado['Latitude'] = None
    df_agrupado['Longitude'] = None
    
    total_enderecos = len(df_agrupado)
    for i, row in df_agrupado.iterrows():
        # Geocodifica SOMENTE se a Lat/Lon corrigida ainda não existir
        if row['Latitude'] is None:
            lat, lon = geocode_address(row['Endereco_Completo_Padrao'], geolocator)
            df_agrupado.at[i, 'Latitude'] = lat
            df_agrupado.at[i, 'Longitude'] = lon
        
        # Usa o Lat/Lon original se a Geocodificação falhar
        if df_agrupado.at[i, 'Latitude'] is None:
            df_agrupado.at[i, 'Latitude'] = row['Latitude_Original']
            df_agrupado.at[i, 'Longitude'] = row['Longitude_Original']
        
        st.progress((i + 1) / total_enderecos, text=f"Geocodificando {i+1} de {total_enderecos} endereços...")

    # Remove colunas auxiliares
    df_agrupado = df_agrupado.drop(columns=['Latitude_Original', 'Longitude_Original', 'Endereco_Completo_Padrao'])

    # 6. --- Formatação do DF para o CIRCUIT (COM NOVAS COORDENADAS) ---
    endereco_completo_circuit = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro']
    )
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode/Postal code']
    )

    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'], 
        'Address': endereco_completo_circuit, 
        'Latitude': df_agrupado['Latitude'],  # <--- AGORA USA A COORDENADA CORRIGIDA
        'Longitude': df_agrupado['Longitude'], # <--- AGORA USA A COORDENADA CORRIGIDA
        'Notes': notas_completas
    })

    # 7. --- Formatação do DF para IMPRESSÃO (3 COLUNAS) ---
    endereco_imprimir_simples = df_agrupado['Endereco_Corrigido']
    separador = pd.Series(['-'] * len(df_agrupado))
    
    df_impressao = pd.DataFrame({
        'Ordem ID': df_agrupado['Sequences_Agrupadas'],
        'Separador': separador,
        'Endereco_Simples': endereco_imprimir_simples
    })
    
    return df_circuit, df_impressao

# --- Interface Streamlit (mantida igual) ---

st.title("🗺️ Corretor de Endereços para Circuit (Avançado)")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("⚙️ Configurações de Correção")
limite_similaridade_ajustado = st.sidebar.slider(
    'Ajuste a Precisão do Corretor (Fuzzy Matching):',
    min_value=80,
    max_value=100,
    value=90,
    step=1,
    help="Valores maiores (ex: 95) agrupam apenas endereços quase idênticos."
)
st.sidebar.info(f"O limite de similaridade atual é: **{limite_similaridade_ajustado}%**")


# --- CORPO PRINCIPAL DO APP ---

st.markdown("---")
st.subheader("1. Carregar Planilha")

uploaded_file = st.file_uploader(
    "Arraste e solte o arquivo aqui:", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file, sheet_name=0)
        
        st.success(f"Arquivo '{uploaded_file.name}' carregado! Total de **{len(df_input)}** registros.")
        
        st.markdown("---")
        st.subheader("2. Corrigir e Gerar Arquivos")
        
        if st.button("🚀 Iniciar Corretor e Geocodificação"):
            # Chama a função principal com as novas coordenadas
            df_circuit, df_impressao = processar_e_corrigir_dados(df_input, limite_similaridade_ajustado)
            
            if df_circuit is not None:
                st.markdown("---")
                st.header("✅ Processamento Concluído!")
                
                total_entradas = len(df_input)
                total_agrupados = len(df_circuit)
                
                st.metric(
                    label="Endereços Únicos Corrigidos",
                    value=total_agrupados,
                    delta=f"-{total_entradas - total_agrupados} agrupados"
                )
                
                # --- SAÍDA PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                st.subheader("3A. Arquivo para Roteirização (Circuit)")
                st.caption("Contém as **NOVAS** coordenadas geocodificadas para maior precisão.")
                st.dataframe(df_circuit.head(5), use_container_width=True)
                
                # Download Circuit
                buffer_circuit = io.BytesIO()
                with pd.ExcelWriter(buffer_circuit, engine='openpyxl') as writer:
                    df_circuit.to_excel(writer, index=False, sheet_name='Circuit Import')
                buffer_circuit.seek(0)
                
                st.download_button(
                    label="📥 Baixar ARQUIVO PARA CIRCUIT",
                    data=buffer_circuit,
                    file_name="Circuit_Import_GEOCODIFICADO.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_circuit"
                )
                
                # --- SAÍDA PARA IMPRESSÃO (LISTA DE 3 COLUNAS) ---
                st.markdown("---")
                st.subheader("3B. Arquivo para Impressão (3 Colunas Separadas)")
                st.dataframe(df_impressao.head(5), use_container_width=True)
                
                # Download Impressão
                buffer_impressao = io.BytesIO()
                with pd.ExcelWriter(buffer_impressao, engine='openpyxl') as writer:
                    df_impressao.to_excel(writer, index=False, sheet_name='Lista 3 Colunas')
                buffer_impressao.seek(0)
                
                st.download_button(
                    label="📄 Baixar ARQUIVO PARA IMPRESSÃO",
                    data=buffer_impressao,
                    file_name="Lista_Impressao_3Colunas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_print"
                )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato e as colunas (Destination Address, Sequence, etc.). Erro: {e}")
