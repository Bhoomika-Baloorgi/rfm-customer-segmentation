"""
Customer Segmentation using RFM Analysis + K-Means Clustering
Dataset: UCI Online Retail — https://archive.ics.uci.edu/dataset/352/online+retail

Run:
    python main.py

Or open notebooks/rfm_analysis.ipynb for the interactive version.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.rfm_analysis import load_and_clean, compute_rfm, score_rfm, assign_segment
from src.clustering import run_kmeans, plot_clusters
from src.visualizations import run_eda, plot_segment_summary, print_business_recommendations


DATA_PATH = "data/online_retail.xlsx"


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 50)
    print("  CUSTOMER SEGMENTATION — RFM + K-MEANS")
    print("=" * 50)

    # Step 1: Load and clean
    print("\n[1/5] Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)

    # Step 2: EDA
    print("\n[2/5] Running exploratory data analysis...")
    run_eda(df)

    # Step 3: Compute RFM
    print("\n[3/5] Computing RFM scores...")
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = assign_segment(rfm)

    # Step 4: K-Means clustering
    print("\n[4/5] Running K-Means clustering...")
    rfm, scaler, model = run_kmeans(rfm)
    plot_clusters(rfm)

    # Step 5: Business insights
    print("\n[5/5] Generating business insights...")
    plot_segment_summary(rfm)
    print_business_recommendations(rfm)

    # Save final output
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    print("\nFinal segments saved to outputs/customer_segments.csv")
    print("\nDone. Check the outputs/ folder for all charts.")


if __name__ == "__main__":
    main()
