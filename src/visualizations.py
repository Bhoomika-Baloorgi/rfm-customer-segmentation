import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

sns.set_style("whitegrid")
COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#378ADD']


def run_eda(df):
    os_import = __import__('os')
    os_import.makedirs('outputs', exist_ok=True)

    _plot_revenue_by_country(df)
    _plot_monthly_revenue(df)
    _plot_top_products(df)
    _plot_order_value_dist(df)
    print("EDA plots saved to outputs/")


def _plot_revenue_by_country(df):
    top = (df.groupby('Country')['Revenue']
             .sum()
             .sort_values(ascending=False)
             .head(10))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=COLORS[0], alpha=0.85)
    ax.set_title('Top 10 countries by revenue', fontsize=13)
    ax.set_xlabel('Total revenue (£)')
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + 1000, bar.get_y() + bar.get_height()/2,
                f'£{val:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/revenue_by_country.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_monthly_revenue(df):
    monthly = (df.set_index('InvoiceDate')
                 .resample('M')['Revenue']
                 .sum())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, marker='o', color=COLORS[1], linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.1, color=COLORS[1])
    ax.set_title('Monthly revenue trend', fontsize=13)
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (£)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/monthly_revenue.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_top_products(df):
    top = (df.groupby('Description')['Quantity']
             .sum()
             .sort_values(ascending=False)
             .head(10))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=COLORS[2], alpha=0.85)
    ax.set_title('Top 10 products by quantity sold', fontsize=13)
    ax.set_xlabel('Total quantity')
    plt.tight_layout()
    plt.savefig('outputs/top_products.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_order_value_dist(df):
    order_vals = df.groupby('InvoiceNo')['Revenue'].sum()
    order_vals = order_vals[order_vals < order_vals.quantile(0.99)]  # remove outliers

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(order_vals, bins=50, color=COLORS[3], alpha=0.8, edgecolor='white')
    ax.set_title('Distribution of order values (excl. top 1% outliers)', fontsize=13)
    ax.set_xlabel('Order value (£)')
    ax.set_ylabel('Number of orders')
    ax.axvline(order_vals.median(), color='black', linestyle='--', linewidth=1.2,
               label=f'Median: £{order_vals.median():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/order_value_dist.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_segment_summary(rfm):
    # Segment size bar chart
    counts = rfm['Segment'].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[i % len(COLORS)] for i in range(len(counts))], alpha=0.85)
    ax.set_title('Customer count per segment', fontsize=13)
    ax.set_ylabel('Number of customers')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                str(val), ha='center', fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('outputs/segment_counts.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Average RFM per segment heatmap
    heatmap_data = (rfm.groupby('Segment')[['Recency','Frequency','Monetary']]
                      .mean()
                      .round(1))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Average RFM values per segment', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/segment_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Segment plots saved.")


def print_business_recommendations(rfm):
    recommendations = {
        'Champions':          '🏆 Reward them. Early access, loyalty perks, ask for reviews.',
        'Loyal Customers':    '💛 Upsell premium products. They trust you already.',
        'New Customers':      '👋 Onboard well. Send welcome series, guide first 3 purchases.',
        'Potential Loyalists':'📈 Nurture with targeted offers. They have the habit, increase value.',
        'At-Risk':            '⚠️  Win-back campaign urgently. Discount + "we miss you" email.',
        'Lost':               '🔴 Low priority. Only re-engage with a strong offer, don\'t spam.'
    }

    print("\n" + "="*55)
    print("  BUSINESS RECOMMENDATIONS PER SEGMENT")
    print("="*55)
    for seg, rec in recommendations.items():
        count = len(rfm[rfm['Segment'] == seg])
        pct = count / len(rfm) * 100
        print(f"\n{seg} ({count} customers, {pct:.1f}%)")
        print(f"  → {rec}")
    print("="*55)
