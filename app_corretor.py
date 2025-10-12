# ... (O código anterior permanece o mesmo até aqui) ...

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
                
                # NOVO BOTÃO DE COPIA COM st.clipboard
                st.clipboard(
                    label="📋 Copiar Lista de Impressão",
                    text=copia_data, # Agora usa o argumento 'text'
                )

                st.info("O botão acima copia o texto automaticamente. O campo abaixo é apenas para visualização e verificação do alinhamento.")
                
                # Área de texto para visualização
                st.text_area(
                    "Conteúdo da Lista de Impressão (ID - Anotações):", 
                    copia_data, 
                    height=300
                )
                
# ... (O restante do bloco with tab2, incluindo o download, permanece o mesmo) ...
