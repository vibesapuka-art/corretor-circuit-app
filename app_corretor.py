import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os

# --- Configurações da Página ---
st.set_page_config(
    page_title="Corretor de Endereços Circuit (Final)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configurações Principais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'


def limpar_endereco(endereco):
    """Normaliza o texto do endereço para melhor comparação, MANTENDO O NÚMERO E VÍRGULAS."""
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    
    # 1. A ÚNICA MUDANÇA: remover caracteres que NÃO são alfanuméricos (\w), espaço (\s) OU VÍRGULA (,)
    # Isso torna a diferença de número (ex: 100 vs 101) mais significativa no score final,
    # forçando o agrupamento a diferenciar endereços por número.
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    
    # 2. Substitui múltiplos espaços por um único
    endereco = re.sub(r'\s+', ' ', endereco)
    
    # 3. Substitui abreviações comuns
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return endereco


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e o agrupamento, usando as Lat/Lon originais.
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
            # Usa o Endereço original mais frequente como Endereço Oficial
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
        
        # USA AS LATITUDE E LONGITUDE ORIGINAIS DO PRIMEIRO PEDIDO DO GRUPO
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        # Mantemos o Bairro e Zipcode mais frequentes para o resultado final
        Bairro_Agrupado=('Bairro', lambda x: x.mode()[0]),
        Zipcode_Agrupado=('Zipcode/Postal code', lambda x: x.mode()[0])
        
    ).reset_index()

    # 5. Formatação do DF para o CIRCUIT (USANDO LAT/LON ORIGINAIS)
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
        'Latitude': df_agrupado['Latitude'],  # <--- COORDENADA ORIGINAL
        'Longitude': df_agrupado['Longitude'], # <--- COORDENADA ORIGINAL
        'Notes': notas_completas
    })

    # 6. Formatação do DF para IMPRESSÃO (3 COLUNAS)
    ordem_id_impressao = df_agrupado['Sequences_Agrupadas']
    separador = pd.Series(['-'] * len(df_agrupado))
    endereco_imprimir_simples = df_agrupado['Endereco_Corrigido']
    
    df_impressao = pd.DataFrame({
        'Ordem ID': ordem_id_impressao,
        'Separador': separador,
        'Endereco_Simples': endereco_imprimir_simples
    })
    
    return df_circuit, df_impressao

# --- Interface Streamlit ---

st.title("🗺️ Corretor de Endereços para Circuit (Final)")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("⚙️ Configurações de Correção")

# Slider de Similaridade 
limite_similaridade_ajustado = st.sidebar.slider(
    'Ajuste a Precisão do Corretor (Fuzzy Matching):',
    min_value=80,
    max_value=100,
    value=90,
    step=1,
    help="Valores maiores (ex: 95) agrupam apenas endereços quase idênticos."
)
st.sidebar.info(f"O limite de similaridade é **{limite_similaridade_ajustado}%**. Se tiver agrupamento errado, aumente este valor para 95% ou mais.")


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
        
        if st.button("🚀 Iniciar Corretor e Agrupamento"):
            # Chama a função principal
            df_circuit, df_impressao = processar_e_corrigir_dados(df_input, limite_similaridade_ajustado)
            
            if df_circuit is not None:
                st.markdown("---")
                st.header("✅ Processamento Concluído!")
                
                total_entradas = len(df_input)
                total_agrupados = len(df_circuit)
                
                st.metric(
                    label="Endereços Únicos Agrupados",
                    value=total_agrupados,
                    delta=f"-{total_entradas - total_agrupados} agrupados"
                )
                
                # --- SAÍDA PARA CIRCUIT (ROTEIRIZAÇÃO) ---
                st.subheader("3A. Arquivo para Roteirização (Circuit)")
                st.caption("Contém as coordenadas **originais** da sua planilha de entrada.")
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
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato e as colunas. Erro: {e}")

### 🚀 Próximos Passos

1.  **Atualize seu `app_corretor.py`** com o código acima.
2.  **Faça o deploy** no Streamlit Cloud.
3.  **Teste a Planilha:**
    * **Primeiro teste:** Tente rodar o processamento com o slider em **90%** (valor padrão). Se o problema de agrupamento incorreto for resolvido, ótimo.
    * **Segundo teste (Se o problema persistir):** Suba o slider para **95%**. Isso fará com que o agrupamento seja muito mais rigoroso, garantindo que a diferença no número da casa impeça a junção.
