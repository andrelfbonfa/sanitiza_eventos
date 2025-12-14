import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# --------------------------------------------------
# Carrega CSV e já trata o básico pra não dar treta
# --------------------------------------------------
def carregar_dados(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Garantindo datas sem timezone pra evitar erro chato do pandas
    df['data_criacao'] = pd.to_datetime(
        df['data_criacao'], errors='coerce', utc=True
    ).dt.tz_localize(None)

    df['data_encerramento'] = pd.to_datetime(
        df['data_encerramento'], errors='coerce', utc=True
    ).dt.tz_localize(None)

    return df


# --------------------------------------------------
# Métricas simples: quantas vezes o alerta aparece e em quantos hosts
# --------------------------------------------------
def calcular_metricas_basicas(df):
    df['frequencia'] = (
        df.groupby(['u_host_host', 'u_item_name', 'u_trigger_description'])['number']
        .transform('count')
    )

    df['escopo'] = (
        df.groupby(['u_item_name', 'u_trigger_description'])['u_host_host']
        .transform('nunique')
    )

    return df


# --------------------------------------------------
# Gera embeddings do texto do trigger (parte mais pesada)
# --------------------------------------------------
def gerar_embeddings(df, model_name='all-MiniLM-L6-v2', device='cpu', batch_size=512):
    model = SentenceTransformer(model_name, device=device)
    textos = df['u_trigger_description'].astype(str).tolist()

    embeddings = []

    for i in tqdm(range(0, len(textos), batch_size), desc="Gerando embeddings"):
        batch = textos[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)

    return np.vstack(embeddings).astype('float16')


# --------------------------------------------------
# Mede o quão genérico é o alerta comparando com outros textos parecidos
# --------------------------------------------------
def calcular_genericidade(df, embeddings, k=10):
    vizinhos = NearestNeighbors(n_neighbors=k, metric='cosine')
    vizinhos.fit(embeddings)

    distancias, _ = vizinhos.kneighbors(embeddings)

    # quanto mais parecido com vários outros, mais genérico
    df['genericidade'] = 1 - distancias.mean(axis=1)
    return df


# --------------------------------------------------
# Normaliza tudo pra ficar na mesma escala
# --------------------------------------------------
def normalizar_metricas(df):
    scaler = MinMaxScaler()
    df[['frequencia', 'escopo', 'genericidade']] = scaler.fit_transform(
        df[['frequencia', 'escopo', 'genericidade']]
    )
    return df


# --------------------------------------------------
# Analisa repetição no tempo (ideia forte da Netflix + regra dos 50%)
# --------------------------------------------------
def calcular_efetividade_temporal(df):
    df['dia_alerta'] = df['data_criacao'].dt.date

    dias_alerta = (
        df.groupby(['u_host_host', 'u_item_name', 'u_trigger_description'])['dia_alerta']
        .nunique()
        .reset_index(name='dias_alerta')
    )

    dias_base = (df['data_criacao'].max() - df['data_criacao'].min()).days + 1

    dias_alerta['repeticao_50pct'] = dias_alerta['dias_alerta'] / dias_base >= 0.5

    df = df.merge(
        dias_alerta,
        on=['u_host_host', 'u_item_name', 'u_trigger_description'],
        how='left'
    )

    # Agora olhando por host: alerta em dia demais geralmente é ruído
    dias_host = df.groupby('u_host_host')['dia_alerta'].nunique().reset_index(name='dias_host')
    dias_com_alerta = (
        df[['u_host_host', 'dia_alerta']]
        .drop_duplicates()
        .groupby('u_host_host')
        .size()
        .reset_index(name='dias_com_alerta')
    )

    host_df = dias_host.merge(dias_com_alerta, on='u_host_host')
    host_df['pct_dias_alerta'] = host_df['dias_com_alerta'] / host_df['dias_host']

    df = df.merge(
        host_df[['u_host_host', 'pct_dias_alerta']],
        on='u_host_host',
        how='left'
    )

    return df


# --------------------------------------------------
# Tempo entre abertura e fechamento (alerta fecha rápido demais é suspeito)
# --------------------------------------------------
def calcular_tempo_resolucao(df, limite_minutos=5):
    df['tempo_resolucao'] = (
        (df['data_encerramento'] - df['data_criacao'])
        .dt.total_seconds() / 60
    )

    df['resolucao_rapida'] = df['tempo_resolucao'] < limite_minutos
    return df


# --------------------------------------------------
# Score final juntando tudo
# --------------------------------------------------
def calcular_score(df):
    df['alert_quality_score'] = 1 - (
        0.3 * df['frequencia'] +
        0.3 * df['escopo'] +
        0.2 * df['genericidade']
    )

    df['alert_quality_score'] += 0.2 * df['pct_dias_alerta']
    df['alert_quality_score'] -= 0.2 * df['repeticao_50pct']
    df['alert_quality_score'] -= 0.2 * df['resolucao_rapida']

    df['inefetivo'] = df['alert_quality_score'] < 0.4
    return df


# --------------------------------------------------
# Explica o motivo (isso aqui é top pra justificar decisão)
# --------------------------------------------------
def gerar_motivo(df):
    def explicar(row):
        motivos = []
        if row['frequencia'] > 0.6:
            motivos.append("dispara demais")
        if row['escopo'] > 0.6:
            motivos.append("impacta muitos hosts")
        if row['genericidade'] > 0.6:
            motivos.append("mensagem genérica")
        if row['repeticao_50pct']:
            motivos.append("repete em mais de 50% do período")
        if row['resolucao_rapida']:
            motivos.append("fecha rápido demais")

        return ", ".join(motivos)

    df['motivo'] = df.apply(explicar, axis=1)
    return df


# --------------------------------------------------
# Salva resultado final
# --------------------------------------------------
def salvar_resultados(df, path="results/alertas_ineficazes.csv"):
    ineficazes = df[df['inefetivo']]
    ineficazes.to_csv(path, index=False)

    print(f"Arquivo gerado: {path}")
    print(f"Total de alertas ineficazes: {len(ineficazes)}")
    print(
        ineficazes[
            ['number', 'u_host_host', 'u_item_name', 'u_trigger_description', 'motivo']
        ].head(10)
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    df = carregar_dados("data/alertas.csv")
    df = calcular_metricas_basicas(df)

    embeddings = gerar_embeddings(df, device='cpu')
    df = calcular_genericidade(df, embeddings)

    df = normalizar_metricas(df)
    df = calcular_efetividade_temporal(df)
    df = calcular_tempo_resolucao(df)

    df = calcular_score(df)
    df = gerar_motivo(df)

    salvar_resultados(df)
