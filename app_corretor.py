# ... O código da ABA 1 e as FUNÇÕES de Pré/Pós-Roteirização permanecem os mesmos ...

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

    sheet_name = 'Table 3' # Valor padrão para o campo de texto
    
    # Campo para o usuário especificar o nome da aba, útil para arquivos .xlsx
    if uploaded_file_pos is not None and uploaded_file_pos.name.endswith('.xlsx'):
        sheet_name = st.text_input(
            "Seu arquivo é um Excel (.xlsx). Digite o nome da aba com os dados da rota (ex: Table 3, Planilha1):", 
            value="Table 3" # Sugere o nome que você indicou
        )

    if uploaded_file_pos is not None:
        try:
            if uploaded_file_pos.name.endswith('.csv'):
                df_input_pos = pd.read_csv(uploaded_file_pos)
            else:
                # Agora usa o nome da aba fornecido pelo usuário (ou o default 'Table 3')
                df_input_pos = pd.read_excel(uploaded_file_pos, sheet_name=sheet_name)
            
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

        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo. Certifique-se de que a aba **'{sheet_name}'** existe e que o formato do arquivo está correto. Erro: {e}")
