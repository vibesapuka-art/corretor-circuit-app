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
st.subheader("Separa Ordem ID da coluna de Notas e prepara para cópia.")

st.warning("⚠️ Atenção: Este aplicativo NÃO lê PDF. Use o arquivo CSV/Excel GERADO PELO SEU CONVERSOR de PDF (após roteirizar no Circuit).")

# --- CORPO PRINCIPAL DO APP ---

uploaded_file = st.file_uploader(
    "1. Arraste e solte o arquivo da rota do Circuit aqui (CSV/Excel):", 
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
        
        # 1. Encontrar a coluna "Notes" (o nome pode variar um pouco)
        coluna_notes = None
        for col in df_input.columns:
            if 'notes' in col.lower():
                coluna_notes = col
                break
        
        if coluna_notes is None:
            st.error("Erro: A coluna 'Notes' (Anotações) não foi encontrada no seu arquivo. Verifique se o arquivo da rota foi gerado corretamente.")
        else:
            df = df_input.copy()
            # Garante que a coluna Notas seja tratada como string
            df[coluna_notes] = df[coluna_notes].astype(str)
            df = df.dropna(subset=[coluna_notes]) 
            
            # 2. Separar a coluna Notes: Parte antes do ';' é o Order ID
            # Remove aspas duplas iniciais/finais que são comuns em CSVs
            df[coluna_notes] = df[coluna_notes].str.strip('"')
            
            # O separador que você usa é ';'
            df_split = df[coluna_notes].str.split(';', n=1, expand=True)
            df['Ordem ID'] = df_split[0].str.strip()
            df['Anotações Completas'] = df_split[1].str.strip() if 1 in df_split.columns else ""
            
            
            # 3. Formatação Final da Tabela
            
            colunas_finais = ['Ordem ID']
            coluna_endereco = None
            
            # Inclui a coluna 'Address'
            for col in df_input.columns:
                if 'address' in col.lower():
                    colunas_finais.append(col) 
                    coluna_endereco = col
                    break
            
            colunas_finais.append('Anotações Completas')
            
            # Renomeia o Address para 'Endereço'
            if coluna_endereco:
                df = df.rename(columns={coluna_endereco: 'Endereço'})
            
            df_final = df[colunas_finais]

            st.markdown("---")
            st.subheader("2. Resultado Final para Cópia e Impressão")
            st.caption("A coluna 'Ordem ID' contém o valor antes do primeiro ponto-e-vírgula.")
            
            # Exibe a tabela
            st.dataframe(df_final, use_container_width=True)

            # 4. Opção de Copiar para a Área de Transferência
            
            # Converte o DataFrame para um formato simples (tab-separated) para cópia
            csv_data = df_final.to_csv(index=False, header=False, sep='\t')
            
            st.markdown("### 3. Copiar para a Área de Transferência")
            st.info("Para copiar para o Excel/Word/etc., selecione todo o texto abaixo (Ctrl+A) e pressione Ctrl+C.")
            
            st.text_area(
                "Conteúdo da Tabela (Separação por Tabulação):", 
                csv_data, 
                height=300
            )

            # Download como Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
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
        st.error(f"Ocorreu um erro ao processar o arquivo. Verifique o formato, as colunas ('Notes'/'Address') e se o arquivo foi convertido corretamente do PDF. Erro: {e}")
