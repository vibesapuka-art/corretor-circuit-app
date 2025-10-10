import pandas as pd
import re
from rapidfuzz import process, fuzz
import io # Para manipulação de arquivos na memória
import streamlit as st # Biblioteca para criar a interface web

# --- Configurações Principais ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
LIMITE_SIMILARIDADE = 90 


def limpar_endereco(endereco):
    """Normaliza o texto do endereço para melhor comparação."""
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    endereco = re.sub(r'[^\w\s]', '', endereco)
    endereco = re.sub(r'\s+', ' ', endereco)
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    return endereco


@st.cache_data # Cache para otimizar a execução de funções pesadas
def processar_e_corrigir_dados(df_entrada):
    """
    Função principal que aplica toda a lógica de correção, agrupamento e formatação.
    """
    df = df_entrada.copy()

    # 1. Limpeza e Normalização
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching e Geração do Mapa de Correção
    for end_principal in enderecos_unicos:
        if end_principal not in mapa_correcao:
            matches = process.extract(
                end_principal, 
                enderecos_unicos, 
                scorer=fuzz.WRatio, 
                limit=None
            )
            grupo_matches = [
                match[0] for match in matches 
                if match[1] >= LIMITE_SIMILARIDADE
            ]
            
            df_grupo = df[df['Endereco_Limpo'].isin(grupo_matches)]
            endereco_oficial_original = df_grupo[COLUNA_ENDERECO].mode()[0]
            
            for end_similar in grupo_matches:
                mapa_correcao[end_similar] = endereco_oficial_original

    # 3. Aplicação do Endereço Corrigido
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # 4. Agrupamento e Concatenação dos Dados
    colunas_agrupamento = ['Endereco_Corrigido', 'Bairro', 'City', 'Zipcode/Postal code']
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        Sequences_Agrupadas=(COLUNA_SEQUENCE, lambda x: ','.join(map(str, sorted(x)))),
        Total_Pacotes=(COLUNA_SEQUENCE, 'count'),
        Latitude=('Latitude', 'first'),
        Longitude=('Longitude', 'first')
    ).reset_index()

    # 5. Formatação para o Circuit
    endereco_completo = (
        df_agrupado['Endereco_Corrigido'] + ', ' + 
        df_agrupado['Bairro']
    )
    
    notas_completas = (
        'Pacotes: ' + df_agrupado['Total_Pacotes'].astype(str) + 
        ' | Cidade: ' + df_agrupado['City'] + 
        ' | CEP: ' + df_agrupado['Zipcode/Postal code']
    )

    # CRIAÇÃO DO DATAFRAME FINAL NA ORDEM CORRETA
    df_circuit = pd.DataFrame({
        'Order ID': df_agrupado['Sequences_Agrupadas'],  # PRIMEIRA COLUNA
        'Address': endereco_completo,
        'Latitude': df_agrupado['Latitude'],
        'Longitude': df_agrupado['Longitude'],
        'Notes': notas_completas
    })
    
    return df_circuit

# --- Interface Streamlit ---

st.title("🗺️ Corretor de Endereços para Circuit")
st.markdown("---")
st.markdown("Esta ferramenta usa Fuzzy Matching para agrupar endereços semelhantes e gerar um arquivo otimizado para importação no Circuit.")

# Widget para fazer upload do arquivo
uploaded_file = st.file_uploader(
    "1. Arraste e solte sua planilha (.csv ou .xlsx)", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    # 2. Carregar o arquivo
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            # Assume que é XLSX se não for CSV
            df_input = pd.read_excel(uploaded_file)
        
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso! Total de {len(df_input)} registros.")

        # Botão para iniciar o processamento
        if st.button("2. Processar e Corrigir Endereços"):
            with st.spinner('Corrigindo e agrupando endereços...'):
                df_resultado = processar_e_corrigir_dados(df_input)
            
            st.markdown("---")
            st.header("✅ Processamento Concluído!")
            st.write(f"Endereços agrupados: **{len(df_resultado)}** (de {len(df_input)} entradas originais).")
            
            # Exibir amostra
            st.subheader("Amostra do Arquivo Final")
            st.dataframe(df_resultado.head(10))
            
            # 3. Botão para Download
            # Converte o DataFrame para o formato Excel na memória
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_resultado.to_excel(writer, index=False, sheet_name='Circuit Import')
            buffer.seek(0)
            
            st.download_button(
                label="3. Baixar Arquivo Circuit_Import_FINAL.xlsx",
                data=buffer,
                file_name="Circuit_Import_FINAL.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique se as colunas 'Destination Address' e 'Sequence' estão presentes. Erro: {e}")