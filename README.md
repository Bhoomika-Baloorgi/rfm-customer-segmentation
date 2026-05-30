# Customer Segmentation using RFM Analysis + K-Means Clustering

## Problem
Businesses struggle to treat all customers the same way — a first-time buyer and a loyal high-spender need completely different marketing strategies. This project segments customers based on actual purchase behavior, so marketing teams can target each group with the right action.

## Approach
**RFM Framework** — every customer is scored on three dimensions:
- **Recency** — how recently did they buy?
- **Frequency** — how often do they buy?
- **Monetary** — how much have they spent?

K-Means clustering then groups customers into behavioral segments automatically, validated with the Silhouette score.

## Dataset
[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) — 541,909 real transactions from a UK-based online retailer (2010–2011).

## Segments Identified
| Segment | Description | Recommended Action |
|---|---|---|
| Champions | High R, F, M | Reward, ask for referrals |
| Loyal Customers | High F & M, moderate R | Upsell premium products |
| New Customers | High R, low F | Strong onboarding sequence |
| Potential Loyalists | Good R & F, growing M | Nurture with targeted offers |
| At-Risk | Low R, used to buy often | Win-back campaign urgently |
| Lost | Low R, F, M | Low priority re-engagement |

## Key Findings
- Champions make up ~X% of customers but drive ~Y% of revenue
- At-Risk segment has highest average order value — high business cost if lost
- Elbow method + Silhouette score both suggested k=4 as optimal

## Tech Stack
- `pandas` — data cleaning and RFM feature engineering
- `scikit-learn` — K-Means clustering, PCA, StandardScaler
- `matplotlib` / `seaborn` — EDA and segment visualizations
- `streamlit` — interactive dashboard

## Project Structure
```
rfm-customer-segmentation/
├── data/                    # Raw dataset (not committed)
├── src/
│   ├── rfm_analysis.py      # Data cleaning + RFM computation
│   ├── clustering.py        # K-Means + PCA visualization
│   └── visualizations.py    # EDA plots + segment charts
├── outputs/                 # Generated charts and CSV
├── main.py                  # Run full pipeline
├── app.py                   # Streamlit dashboard
└── requirements.txt
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset → save as data/online_retail.xlsx

# 3. Run the full pipeline
python main.py

# 4. Launch the dashboard
streamlit run app.py
```

## Results
All output charts are saved to `outputs/`:
- `elbow_silhouette.png` — choosing optimal k
- `cluster_pca.png` — 2D cluster visualization
- `segment_counts.png` — customer distribution
- `segment_heatmap.png` — avg RFM per segment
- `customer_segments.csv` — final labelled dataset
