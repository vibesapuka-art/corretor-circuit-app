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
                
                # Botão de Copiar (CORRIGIDO com st.clipboard)
                st.clipboard(
                    label="📋 Copiar Lista de Impressão",
                    text=copia_data,
                )

                st.info("O botão acima copia o texto automaticamente. O campo abaixo é apenas para visualização e verificação do alinhamento.")
                
                # Área de texto para visualização
                st.text_area(
                    "Conteúdo da Lista de Impressão (ID - Anotações):", 
                    copia_data, 
                    height=300
                )

                # Download como Excel (mantém o formato tabulado, caso o usuário queira importar)
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
