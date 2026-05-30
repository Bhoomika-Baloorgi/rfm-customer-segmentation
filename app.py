"""
Streamlit dashboard for RFM Customer Segmentation
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Dashboard")
st.caption("RFM Analysis + K-Means Clustering · UCI Online Retail Dataset")

@st.cache_data
def load_segments():
    return pd.read_csv("outputs/customer_segments.csv")

if not os.path.exists("outputs/customer_segments.csv"):
    st.warning("Run `python main.py` first to generate the segments.")
    st.stop()

rfm = load_segments()

# --- Metrics row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(rfm):,}")
col2.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
col3.metric("Avg Order Frequency", f"{rfm['Frequency'].mean():.1f}")
col4.metric("Avg Spend (£)", f"{rfm['Monetary'].mean():,.0f}")

st.divider()

# --- Segment breakdown ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Customers per segment")
    counts = rfm['Segment'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#7F77DD','#1D9E75','#D85A30','#D4537E','#378ADD','#BA7517']
    ax.bar(counts.index, counts.values,
           color=colors[:len(counts)], alpha=0.85)
    ax.set_ylabel("Customers")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Average RFM by segment")
    hm = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(hm.T, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- Cluster scatter ---
if 'PCA1' in rfm.columns:
    st.subheader("Cluster visualization (PCA)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, seg in enumerate(rfm['Segment'].unique()):
        mask = rfm['Segment'] == seg
        ax.scatter(rfm.loc[mask,'PCA1'], rfm.loc[mask,'PCA2'],
                   label=seg, alpha=0.5, s=20, color=colors[i % len(colors)])
    ax.legend(loc='best')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- Segment filter ---
st.divider()
st.subheader("Explore a segment")
seg = st.selectbox("Select segment", sorted(rfm['Segment'].unique()))
filtered = rfm[rfm['Segment'] == seg]
st.write(f"{len(filtered)} customers in **{seg}**")
st.dataframe(filtered[['CustomerID','Recency','Frequency','Monetary','RFM_Score']].head(20),
             use_container_width=True)

recs = {
    'Champions':          'Reward with loyalty perks and early access. Ask for referrals.',
    'Loyal Customers':    'Upsell higher-value products. They already trust the brand.',
    'New Customers':      'Strong onboarding sequence. Guide through first 3 purchases.',
    'Potential Loyalists':'Targeted offers to increase purchase frequency.',
    'At-Risk':            'Urgent win-back campaign with a time-limited discount.',
    'Lost':               'Low priority. Only re-engage with a strong offer.'
}
if seg in recs:
    st.info(f"**Recommended action:** {recs[seg]}")
