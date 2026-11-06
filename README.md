# E-commerce Customer Segmentation (RFM + Clustering)

Segmenting ~4,300 customers of a UK online retailer into actionable marketing groups, using RFM feature engineering and four clustering algorithms compared on cluster quality.

**Stack:** Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn

---

## Problem

A business can't treat every customer the same, but raw transaction logs don't tell you who is loyal, who is about to churn, and who is worth a discount. The goal here is to turn 541,909 raw transaction rows into a small number of customer segments that a marketing team can actually act on.

## Data

[Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI Machine Learning Repository) — transactions from a UK-based online gift retailer, Dec 2010 to Dec 2011.

| | |
|---|---|
| Raw rows | 541,909 |
| Rows after cleaning | 392,732 |
| Customers after aggregation | ~4,300 |
| Columns | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

The CSV is not committed to this repo (see [Running it](#running-it)).

## Approach

**1. Cleaning**
- Dropped rows with missing `CustomerID` or `Description` (customer-level analysis needs an ID).
- Removed exact duplicate rows.
- Removed cancelled transactions (invoice numbers starting with `C`), which carry negative quantities and would distort spend totals.
- Parsed `InvoiceDate` to datetime and derived `TotalAmount = Quantity × UnitPrice`.

**2. EDA**
- Correlation of `Quantity`, `UnitPrice`, `TotalAmount` — confirmed `TotalAmount` is driven almost entirely by quantity (r = 0.99), while unit price is independent of basket size.
- Customers by country — the UK dominates with 3,950 unique customers; Germany, France and Spain form a smaller international base.
- Monthly revenue for the top 5 countries — a clear upward trend through the year with a sharp Q4 holiday spike.

**3. Feature engineering (RFM)**

Aggregated transactions to one row per customer:

| Feature | Definition |
|---|---|
| **Recency** | Days since the customer's last purchase (lower = more engaged) |
| **Frequency** | Number of distinct invoices |
| **Monetary** | Total amount spent |

All three were heavily right-skewed (skew of 1.25, 12.10 and 19.34). A `log1p` transform brought them close to normal (-0.38, 1.21, 0.36). Features were then scaled with `RobustScaler`, chosen over `StandardScaler` because it uses the median and IQR and so is not dragged around by the high-spending outliers that a retail dataset always has — and those outliers are real customers worth keeping, not errors. Winsorizing was tried first and discarded because it flattened genuine high-value behaviour.

**4. Models**

Four algorithms were fit on the same scaled RFM features so they could be compared directly:

| Model | Why it was tried |
|---|---|
| K-Means | Baseline centroid method; `k` chosen via elbow plot and silhouette sweep |
| Hierarchical (Ward) | No need to assume spherical clusters; dendrogram shows merge structure |
| DBSCAN | Density-based, labels outliers as noise instead of forcing them into a cluster |
| Gaussian Mixture | Soft assignment; allows elliptical clusters with different covariances |

## Results

| Model | Silhouette score |
|---|---|
| **K-Means (k = 4)** | **0.334** |
| Hierarchical (k = 4) | 0.301 |
| GMM (4 components) | 0.158 |
| DBSCAN (eps = 0.2, min_samples = 20) | 0.146 |

K-Means gave the cleanest separation and was selected. A silhouette sweep from k = 2 to k = 10 scored k = 2 highest (0.430), but two clusters only split customers into "active" and "lapsed" — not enough resolution to build different campaigns on. **k = 4 was chosen as the trade-off between statistical separation and business usefulness**, which is the more honest framing for a segmentation whose output is a marketing plan.

Two caveats worth stating: DBSCAN's score is computed with noise points excluded, so it isn't strictly comparable to the others, and silhouette favours compact spherical clusters, which structurally advantages K-Means over DBSCAN and GMM.

### The four segments

Profiled in original units (days, order counts, £) rather than log-scaled values:

| Segment | Customers | Median recency | Median orders | Median spend | Share of revenue |
|---|---|---|---|---|---|
| **Champions** | 730 (17%) | 9 days | 10 | £3,739 | **65.8%** |
| **Regulars** | 1,226 (28%) | 59 days | 4 | £1,299 | 23.6% |
| **Recent, low-value** | 837 (19%) | 17 days | 2 | £480 | 5.2% |
| **Lapsed / one-time** | 1,546 (36%) | 178 days | 1 | £279 | 5.4% |

**The headline finding: 17% of customers generate 66% of revenue.** The 730 Champions have bought a median of 10 times and spent a median of £3,739 each, while the largest group by headcount — 1,546 lapsed one-time buyers — contributes just 5.4% of revenue between them.

That single fact reframes the marketing budget. Retention spend on the top segment protects two-thirds of revenue; broad reactivation campaigns aimed at the lapsed group are chasing a twentieth of it.

| Segment | Action |
|---|---|
| Champions | Retain — loyalty perks, early access, referral asks. Highest priority. |
| Regulars | Grow — cross-sell and bundles to raise order frequency toward Champion behaviour |
| Recent, low-value | Nurture — recent buyers with low basket size; onboarding and upsell sequences |
| Lapsed / one-time | Low-cost reactivation only; don't over-invest in a segment worth 5% of revenue |

## What I'd do next

- Add a CLTV estimate per segment to size the future revenue at stake, not just historical spend.
- Validate stability by refitting on a held-out time window — segments that reshuffle month to month aren't safe to build campaigns on.
- Export the labelled customer table to CSV as a deliverable for a CRM.

## Running it

```bash
git clone https://github.com/Pierre-Aziz/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
pip install -r requirements.txt
```

Download `data.csv` from the [UCI Online Retail page](https://archive.ics.uci.edu/dataset/352/online+retail) and place it in the project root, then open the notebook:

```bash
jupyter notebook notebooks/customer_segmentation.ipynb
```

## Repo structure

```
.
├── notebooks/
│   └── customer_segmentation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

**Pierre Ashraf** — B.Sc. Business Informatics, German International University
[LinkedIn](https://linkedin.com/in/pierreaziz21)
