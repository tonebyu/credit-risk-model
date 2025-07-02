import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def assign_high_risk_labels(input_csv_path: str, output_csv_path: str, snapshot_date: str = "2023-01-01"):
    df = pd.read_csv(input_csv_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    rfm_df = df[['CustomerId', 'TransactionStartTime', 'Amount']].copy()

    snapshot_date = pd.to_datetime(snapshot_date)
    rfm = rfm_df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby('Cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_summary.sum(axis=1).idxmin()

    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    rfm = rfm[['is_high_risk']].reset_index()

    df_final = df.merge(rfm, on='CustomerId', how='left')
    df_final.to_csv(output_csv_path, index=False)
    print(f"✅ High-risk labels added and saved to: {output_csv_path}")
