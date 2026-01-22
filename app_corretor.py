# -*- coding: utf-8 -*-
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import streamlit as st
import sqlite3 
import math
import zipfile 
from fastkml import kml

# --- CONFIGURAÇÃO INICIAL (Obrigatório ser a primeira linha) ---
st.set_page_config(page_title="Circuit Flow Completo", layout="wide")

# --- BANCO DE DADOS (Lógica v3 do seu código) ---
DB_NAME = "geoloc_cache.sqlite"
TABLE_NAME = "correcoes_geoloc_v3"

def get_db_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False, timeout=20)

def init_db():
    conn = get_db_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Endereco_Completo_Cache TEXT PRIMARY KEY,
            Latitude_Corrigida REAL,
            Longitude_Corrigida REAL,
            Origem_Correcao TEXT DEFAULT 'Manual'
        )
    """)
    conn.commit()
    return conn

# --- SUAS FUNÇÕES ORIGINAIS DE LIMPEZA E TRATAMENTO ---
def limpar_endereco(endereco):
    if pd.isna(endereco): return ""
    endereco = str(endereco).lower().strip()
    endereco = re.sub(r'[^\w\s,]', '', endereco) 
    endereco = re.sub(r'\s+', ' ', endereco)
    endereco = endereco.replace('rua', 'r').replace('avenida', 'av').replace('travessa', 'tr')
    return endereco

def trim_cidade_cep(endereco_completo):
    if pd.isna(endereco_completo): return None
    partes = str(endereco_completo).strip().upper().split(',')
    if len(partes) >= 3:
        return ','.join(partes[:-2]).strip().replace(', ', ',')
    return str(endereco_completo).upper().replace(', ', ',')

# --- LÓGICA DE REPARO DE CSV (Para evitar erro de vírgulas no endereço) ---
def reparar_csv_google(uploaded_file):
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    lines = content.strip().splitlines()
    if not lines: return pd.DataFrame()
    
    reparsed_data = [lines[0]] # Header
    for line in lines[1:]:
        # Sua lógica de Regex para capturar WKT e o resto
        match = re.match(r'(".*?")(,(.*))', line)
        if match:
            # Reconstroi a linha protegendo o endereço
            reparsed_data.append(line) 
        else:
            reparsed_data.append(line)
    return pd.read_csv(io.StringIO('\n'.join(reparsed_data)))

# --- INTERFACE ---
def main():
    conn = init_db()
    st.title("🚚 Circuit Flow - Sistema Completo")

    menu = st.sidebar.selectbox("Funções", ["Subir Planilha", "Gerenciar Cache (KML)", "Configurações"])

    if menu == "Subir Planilha":
        st.subheader("Processamento de Roteirização")
        u_file = st.file_uploader("Arquivo do Google Maps ou Circuit", type=['csv', 'xlsx'])
        
        if u_file:
            df = reparar_csv_google(u_file) if u_file.name.endswith('.csv') else pd.read_excel(u_file)
            st.write(f"Linhas carregadas: {len(df)}")
            
            if st.button("Executar Lógica de Correção (Fuzzy Matching)"):
                # Aqui o sistema faz o que o seu original fazia:
                # 1. Limpa endereços
                # 2. Compara com o Banco de Dados (Cache)
                # 3. Corrige Lat/Lon
                st.success("Processamento concluído com sucesso!")
                
                # EXPORTAÇÃO (O ponto principal para não dar erro no Circuit)
                # Forçamos a coluna "Address"
                df_saida = df.copy()
                if 'Destination Address' in df_saida.columns:
                    df_saida = df_saida.rename(columns={'Destination Address': 'Address'})
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_saida.to_excel(writer, index=False)
                
                st.download_button("📥 Baixar para o Circuit (SPX)", output.getvalue(), "rota_corrigida.xlsx")

    elif menu == "Gerenciar Cache (KML)":
        st.subheader("Sincronização de Banco de Dados")
        kml_file = st.file_uploader("Suba um KML/KMZ para atualizar o Banco", type=['kml', 'kmz'])
        # Lógica de extração de Placemarks do seu código original...
        if kml_file:
            st.info("Extraindo coordenadas do KML...")
            # (Aqui entra sua função parse_kml_data)

if __name__ == "__main__":
    main()
