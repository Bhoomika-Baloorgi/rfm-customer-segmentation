import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean(filepath):
    df = pd.read_excel(filepath, dtype={'CustomerID': str})
    print(f"Raw shape: {df.shape}")

    # Drop nulls in key columns
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)

    # Remove cancelled orders
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Remove negative/zero quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Add revenue column
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Parse date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print(f"Clean shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    return df


def compute_rfm(df):
    # Snapshot date: 1 day after last transaction
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('InvoiceNo',   'nunique'),
        Monetary  = ('Revenue',     'sum')
    ).reset_index()

    print(f"\nRFM summary:\n{rfm.describe().round(2)}")
    return rfm


def score_rfm(rfm):
    # Score 1-5 (5 = best), recency is inverted (lower days = better)
    rfm['R_Score'] = pd.qcut(rfm['Recency'],   q=5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],  q=5, labels=[1,2,3,4,5], duplicates='drop')

    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    return rfm


def assign_segment(rfm):
    def label(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At-Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Potential Loyalists'

    rfm['Segment'] = rfm.apply(label, axis=1)
    print(f"\nSegment distribution:\n{rfm['Segment'].value_counts()}")
    return rfm
