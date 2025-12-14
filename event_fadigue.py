import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# =============================
# Funções principais
# =============================

def carregar_dados(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Converter datas para tz-naive (evita erros)
    df['data_criacao'] = pd.to_datetime(df['data_criacao'], errors='coerce', utc=True).dt.tz_localize(None)
    df['data_encerramento'] = pd.to_datetime(df['data_encerramento'], errors='coerce', utc=True).dt.tz_localize(None)
    return df

def calcular_metricas_basicas(df):
    # Frequência total de alertas por host+trigger
    df['frequencia'] = df.groupby(['u_host_host','u_item_name','u_trigger_description'])['number'].transform('count')
    # Quantos hosts únicos foram afetados por trigger
    df['escopo'] = df.groupby(['u_item_name','u_trigger_description'])['u_host_host'].transform('nunique')
    return df

def gerar_embeddings(df, model_name='all-MiniLM-L6-v2', device='cpu', batch_size=512):
    model = SentenceTransformer(model_name, device=device)
    embeddings = []
    texts = df['u_trigger_description'].astype(str).tolist()
    for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float16')
    return embeddings

def calcular_genericidade(df, embeddings, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    df['genericidade'] = 1 - distances.mean(axis=1)
    return df

def normalizar_metricas(df):
    scaler = MinMaxScaler()
    df[['frequencia','escopo','genericidade']] = scaler.fit_transform(df[['frequencia','escopo','genericidade']])
    return df

def calcular_efetividade_temporal(df):
    df['dia_alerta'] = df['data_criacao'].dt.date

    # Dias totais por host+trigger
    dias_por_host_trigger = df.groupby(['u_host_host','u_item_name','u_trigger_description'])['dia_alerta'].nunique().reset_index()
    dias_por_host_trigger.rename(columns={'dia_alerta':'dias_alerta'}, inplace=True)

    # Total de dias da base
    dias_base = (df['data_criacao'].max() - df['data_criacao'].min()).days + 1

    dias_por_host_trigger['repeticao_50pct'] = dias_por_host_trigger['dias_alerta'] / dias_base >= 0.5

    df = df.merge(dias_por_host_trigger[['u_host_host','u_item_name','u_trigger_description','dias_alerta','repeticao_50pct']],
                  on=['u_host_host','u_item_name','u_trigger_description'], how='left')

    # Percentual de dias que cada host alertou
    dias_totais_host = df.groupby('u_host_host')['dia_alerta'].nunique().reset_index()
    dias_totais_host.rename(columns={'dia_alerta':'dias_totais'}, inplace=True)
    dias_alerta_host = df[['u_host_host','dia_alerta']].drop_duplicates().groupby('u_host_host')['dia_alerta'].count().reset_index()
    dias_alerta_host.rename(columns={'dia_alerta':'dias_com_alerta'}, inplace=True)

    df_host = pd.merge(dias_totais_host, dias_alerta_host, on='u_host_host')
    df_host['pct_dias_alerta'] = df_host['dias_com_alerta'] / df_host['dias_totais']
    df_host['efetivo_host'] = df_host['pct_dias_alerta'] >= 0.5

    df = df.merge(df_host[['u_host_host','pct_dias_alerta','efetivo_host']], on='u_host_host', how='left')
    return df

def calcular_tempo_resolucao(df, limite_minutos=5):
    # Garantir que datas sejam tz-naive
    df['data_criacao'] = pd.to_datetime(df['data_criacao'], utc=True).dt.tz_localize(None)
    df['data_encerramento'] = pd.to_datetime(df['data_encerramento'], utc=True).dt.tz_localize(None)

    df['tempo_resolucao'] = (df['data_encerramento'] - df['data_criacao']).dt.total_seconds() / 60
    df['resolucao_rapida'] = df['tempo_resolucao'] < limite_minutos
    return df

def calcular_score(df):
    df['alert_quality_score'] = 1 - (0.3*df['frequencia'] + 0.3*df['escopo'] + 0.2*df['genericidade'])
    df['alert_quality_score'] += 0.2 * df['pct_dias_alerta']
    df['alert_quality_score'] -= 0.2 * df['repeticao_50pct']   # penaliza repetição alta
    df['alert_quality_score'] -= 0.2 * df['resolucao_rapida']  # penaliza resolução rápida
    df['inefetivo'] = df['alert_quality_score'] < 0.4
    return df

def gerar_motivo(df):
    def explicar(row):
        motivos = []
        if row['frequencia'] > 0.6:
            motivos.append("frequência alta")
        if row['escopo'] > 0.6:
            motivos.append("escopo grande")
        if row['genericidade'] > 0.6:
            motivos.append("trigger genérico")
        if row['pct_dias_alerta'] < 0.5:
            motivos.append("alerta pouco frequente no tempo")
        if row['repeticao_50pct']:
            motivos.append("repetição >50% do período")
        if row['resolucao_rapida']:
            motivos.append("resolução muito rápida")
        return ", ".join(motivos)

    df['motivo'] = df.apply(explicar, axis=1)
    return df

def salvar_resultados(df, path="results/alertas_ineficazes.csv"):
    alertas_ineficazes = df[df['inefetivo']]
    alertas_ineficazes.to_csv(path, index=False)
    print(f"Arquivo '{path}' gerado com sucesso!")
    print(f"Total de alertas ineficazes: {len(alertas_ineficazes)}")
    print(alertas_ineficazes[['number','u_host_host','u_item_name','u_trigger_description','motivo']].head(10))

# =============================
# Fluxo principal
# =============================

if __name__ == "__main__":
    df = carregar_dados("data/alertas.csv")
    df = calcular_metricas_basicas(df)
    embeddings = gerar_embeddings(df, device='cpu')  # use 'cuda' se GPU disponível
    df = calcular_genericidade(df, embeddings)
    df = normalizar_metricas(df)
    df = calcular_efetividade_temporal(df)
    df = calcular_tempo_resolucao(df, limite_minutos=5)
    df = calcular_score(df)
    df = gerar_motivo(df)
    salvar_resultados(df)
