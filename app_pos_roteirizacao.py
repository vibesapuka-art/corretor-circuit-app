import pandas as pd
import io
import streamlit as st
import os

# --- Configurações da Página ---
st.set_page_config(
    page_title="Processar Rota Circuit para Impressão",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 Processador Pós-Roteirização Circuit")
st.subheader("Separa Ordem ID e prepara para cópia/impressão.")

st.info("Instrução: Use o arquivo CSV gerado após converter o PDF da rota do Circuit para Excel/CSV (geralmente é o arquivo com o maior número de colunas/dados da rota).")

# --- CORPO PRINCIPAL DO APP ---

uploaded_file = st.file_uploader(
    "Arraste e solte o arquivo da rota do Circuit aqui (CSV/Excel):", 
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    try:
        # Carregar o arquivo
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            # Tenta ler a primeira aba do Excel
            df_input = pd.read_excel(uploaded_file, sheet_name=0)
        
        st.success(f"Arquivo '{uploaded_file.name}' carregado! Total de **{len(df_input)}** registros.")

        # --- PROCESSAMENTO ---
        
        # 1. Encontrar a coluna "Notes" (o nome pode variar um pouco após a conversão)
        coluna_notes = None
        for col in df_input.columns:
            if 'notes' in col.lower():
                coluna_notes = col
                break
        
        if coluna_notes is None:
            st.error("Erro: A coluna 'Notes' (Anotações) não foi encontrada no seu arquivo. Verifique se o arquivo da rota foi gerado corretamente.")
        else:
            df = df_input.copy()
            df = df.dropna(subset=[coluna_notes]) # Remove linhas sem anotações
            
            # 2. Separar a coluna Notes: Parte antes do ';' é o Order ID
            # Remove aspas duplas iniciais/finais que são comuns em CSVs
            df[coluna_notes] = df[coluna_notes].astype(str).str.strip('"')
            
            # Divide a coluna na primeira ocorrência de ';'
            df[['Ordem ID', 'Anotações Completas']] = df[coluna_notes].str.split(';', n=1, expand=True)
            
            # Limpa espaços em branco e remove caracteres indesejados nas novas colunas
            df['Ordem ID'] = df['Ordem ID'].str.strip()
            df['Anotações Completas'] = df['Anotações Completas'].str.strip()
            
            # 3. Formatação Final da Tabela
            
            # Tenta incluir a coluna 'Address' se existir
            colunas_finais = ['Ordem ID']
            coluna_endereco = None
            for col in df_input.columns:
                if 'address' in col.lower():
                    colunas_finais.append(col) # Inclui o nome real da coluna de Endereço
                    coluna_endereco = col
                    break
            
            colunas_finais.append('Anotações Completas')
            
            # Renomeia o Address para 'Endereço'
            if coluna_endereco:
                df = df.rename(columns={coluna_endereco: 'Endereço'})
            
            df_final = df[colunas_finais]

            st.markdown("---")
            st.subheader("Resultado Final para Cópia e Impressão")
            
            # 4. Exibir e Opção de Copiar
            
            # Exibe a tabela
            st.dataframe(df_final, use_container_width=True)

            # Opção de Cópia para a Área de Transferência (CSV formatado)
            
            # Converte o DataFrame para CSV para cópia, removendo o cabeçalho e índice
            csv_data = df_final.to_csv(index=False, header=False, sep='\t')
            
            st.text_area(
                "Área de Transferência (Selecione o texto e Copie - Ctrl+C)", 
                csv_data, 
                height=300
            )

            # Download como CSV (para quem prefere)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_final.to_excel(writer, index=False, sheet_name='Lista Impressao')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Baixar Lista Limpa (Excel)",
                data=buffer,
                file_name="Lista_Ordem_Impressao.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_list"
            )

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato e as colunas. Erro: {e}")
