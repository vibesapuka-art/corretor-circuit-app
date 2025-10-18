# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import os

# --- Configurações Iniciais da Página ---
st.set_page_config(
    page_title="Circuit Flow Completo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS REMOVIDO PARA EVITAR ERRO DE INICIALIZAÇÃO (TypeError) ---
# A remoção deste bloco garante que o app inicialize corretamente e mostre as opções.
# --------------------------------------------------------------------------------------


# --- Configurações Globais (Colunas) ---
COLUNA_ENDERECO = 'Destination Address'
COLUNA_SEQUENCE = 'Sequence'
COLUNA_LATITUDE = 'Latitude'
COLUNA_LONGITUDE = 'Longitude'
# NOVAS COLUNAS
COLUNA_GAIOLA = 'Gaiola' 
COLUNA_ID_UNICO = 'ID_UNICO' # ID temporário: Gaiola-Sequence (Ex: A1-1, G3-1)

# ===============================================
# FUNÇÕES DE PRÉ-ROTEIRIZAÇÃO (CORREÇÃO/AGRUPAMENTO)
# ===============================================

def limpar_endereco(endereco):
    """
    Normaliza o texto do endereço para melhor comparação.
    MANTÉM NÚMEROS e VÍRGULAS (,) para que endereços com números diferentes
    não sejam agrupados.
    """
    if pd.isna(endereco):
        return ""
    endereco = str(endereco).lower().strip()
    
    # Remove caracteres que NÃO são alfanuméricos (\w), espaço (\s) OU VÍRGULA (,)
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    
    # Substitui múltiplos espaços por um único
    endereco = re.sub(r'\s+', ' ', endereco)
    
    # Substitui abreviações comuns para padronização
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    
    return endereco


# Função auxiliar para lidar com valores vazios no mode()
def get_most_common_or_empty(x):
    """
    Retorna o valor mais comum de uma Série Pandas ou uma string vazia se todos forem NaN.
    """
    x_limpo = x.dropna()
    if x_limpo.empty:
        return ""
    return x_limpo.mode().iloc[0]


@st.cache_data
def processar_e_corrigir_dados(df_entrada, limite_similaridade):
    """
    Função principal que aplica a correção e o agrupamento.
    """
    # Adicionando tratamento para o caso de a coluna ID_UNICO ainda não existir (apenas para segurança)
    colunas_essenciais_base = [COLUNA_ENDERECO, COLUNA_SEQUENCE, COLUNA_LATITUDE, COLUNA_LONGITUDE, 'Bairro', 'City', 'Zipcode/Postal code', COLUNA_GAIOLA]
    for col in colunas_essenciais_base:
        if col not in df_entrada.columns:
            st.error(f"Erro: A coluna essencial '{col}' não foi encontrada. Verifique se o DataFrame foi carregado e as gaiolas foram confirmadas.")
            return None, None 

    # O ID_UNICO deve ter sido criado na seção 1.3
    if COLUNA_ID_UNICO not in df_entrada.columns:
        st.error(f"Erro interno: A coluna temporária '{COLUNA_ID_UNICO}' não foi gerada. Verifique se o botão 'UNIFICAR, CORRIGIR E AGRUPAR' foi clicado corretamente.")
        return None, None


    df = df_entrada.copy()
    
    # Preenchimento e Garantia de Tipos (Essencial)
    df['Bairro'] = df['Bairro'].astype(str).replace('nan', '', regex=False)
    df['City'] = df['City'].astype(str).replace('nan', '', regex=False)
    df['Zipcode/Postal code'] = df['Zipcode/Postal code'].astype(str).replace('nan', '', regex=False)
    df[COLUNA_GAIOLA] = df[COLUNA_GAIOLA].astype(str).replace('nan', '', regex=False)
    
    # Cria a coluna numérica para a ORDENAÇÃO. Aqui, usamos a SEQUENCE original do pacote.
    # A coluna ID_UNICO já deve vir com o * do volumoso, se houver.
    df['Sequence_Num'] = df[COLUNA_SEQUENCE].astype(str).str.replace(r'\*|\s', '', regex=True)
    df['Sequence_Num'] = pd.to_numeric(df['Sequence_Num'], errors='coerce').fillna(float('inf')).astype(float)


    # 1. Limpeza e Normalização (Fuzzy Matching)
    df['Endereco_Limpo'] = df[COLUNA_ENDERECO].apply(limpar_endereco)
    enderecos_unicos = df['Endereco_Limpo'].unique()
    mapa_correcao = {}
    
    # 2. Fuzzy Matching para Agrupamento
    progresso_bar = st.progress(0, text="Iniciando Fuzzy Matching...")
    total_unicos = len(enderecos_unicos)
    
    if total_unicos > 0:
        for i, end_principal in enumerate(enderecos_unicos):
            if end_principal not in mapa_correcao:
                # CORREÇÃO DE SINTAXE: Chamada em uma linha
                matches = process.extract(end_principal, enderecos_unicos, scorer=fuzz.WRatio, limit=None)
                
                grupo_matches = [match[0] for match in matches if match[1] >= limite_similaridade]
                
                df_grupo = df[df['Endereco_Limpo'].isin(grupo_matches)]
                endereco_oficial_original = get_most_common_or_empty(df_grupo[COLUNA_ENDERECO])
                if not endereco_oficial_original:
                    endereco_oficial_original = end_principal 
                
                for end_similar in grupo_matches:
                    mapa_correcao[end_similar] = endereco_oficial_original
                    
                progresso_bar.progress((i + 1) / total_unicos, text=f"Processando {i+1} de {total_unicos} endereços únicos...")
        
        progresso_bar.empty()
        st.success("Fuzzy Matching concluído!")
    else:
        progresso_bar.empty()
        st.warning("Nenhum endereço encontrado para processar.")


    # 3. Aplicação do Endereço Corrigido
    df['Endereco_Corrigido'] = df['Endereco_Limpo'].map(mapa_correcao)

    # 4. Agrupamento (POR ENDEREÇO CORRIGIDO E CIDADE)
    colunas_agrupamento = ['Endereco_Corrigido', 'City'] 
    
    df_agrupado = df.groupby(colunas_agrupamento).agg(
        # Agrupa os IDs ÚNICOS (Gaiola-Sequence) que já contêm o '*'
        Sequences_Agrupadas=(COLUNA_ID_UNICO, 
                             lambda x: ','.join(map(str, sorted(x, key=lambda y: int(re.sub(r'[^\d]', '', str(y).split('-')[-1])) if re.sub(r'[^\d]', '', str(y).split('-')[-1]).isdigit() else float('inf'))))
                            ), 
        Total_Pacotes=('Sequence_Num', lambda x: (x != float('inf')).sum()), 
        Latitude=(COLUNA_LATITUDE, 'first'),
        Longitude=(COLUNA_LONGITUDE, 'first'),
        
        # Agrupa as informações comuns
        Bairro_Agrupado=('Bairro', get_most_common_or_empty),
        Zipcode_Agrupado=('Zipcode/Postal code', get_most_common_or_empty),
        
        # Agrupa as gaiolas (mantém TODAS as gaiolas únicas daquele endereço)
        Gaiola_Ag
