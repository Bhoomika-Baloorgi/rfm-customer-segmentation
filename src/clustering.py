import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def find_optimal_k(rfm_scaled, k_range=range(2, 10)):
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, km.labels_))

    # Plot elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, marker='o', color='#7F77DD')
    ax1.set_title('Elbow method — choose where curve bends')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')

    ax2.plot(list(k_range), silhouettes, marker='o', color='#1D9E75')
    ax2.set_title('Silhouette score — higher is better')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette score')

    plt.tight_layout()
    plt.savefig('outputs/elbow_silhouette.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/elbow_silhouette.png")

    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"Suggested k by silhouette: {best_k}")
    return best_k


def run_kmeans(rfm, k=4):
    features = rfm[['Recency', 'Frequency', 'Monetary']].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Find optimal k
    best_k = find_optimal_k(scaled)
    k = best_k  # override with data-driven choice

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm['Cluster'] = km.fit_predict(scaled)

    print(f"\nCluster sizes:\n{rfm['Cluster'].value_counts().sort_index()}")

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    rfm['PCA1'] = components[:, 0]
    rfm['PCA2'] = components[:, 1]

    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    return rfm, scaler, km


def plot_clusters(rfm):
    colors = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#378ADD']
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, cluster in enumerate(sorted(rfm['Cluster'].unique())):
        mask = rfm['Cluster'] == cluster
        segment = rfm.loc[mask, 'Segment'].mode()[0] if 'Segment' in rfm.columns else f"Cluster {cluster}"
        ax.scatter(rfm.loc[mask, 'PCA1'], rfm.loc[mask, 'PCA2'],
                   c=colors[i % len(colors)], label=segment, alpha=0.6, s=30)

    ax.set_title('Customer segments — PCA visualization', fontsize=14)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('outputs/cluster_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/cluster_pca.png")
