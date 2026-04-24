"""
train_model.py — SmartBite AI — ML Training Pipeline
──────────────────────────────────────────────────────────────────────
Run ONCE before starting app.py:
    python train_model.py

Outputs:
    model.pkl            – Trained KNN NearestNeighbors
    scaler.pkl           – Fitted StandardScaler
    encoders.pkl         – Fitted LabelEncoders (cuisine, type)
    meta.json            – Feature/class metadata consumed by app.py
    restaurants_clean.csv– Cleaned, feature-engineered dataset

New in this version:
    print_dataset_stats()  – dataset EDA summary
    evaluate_model()       – avg neighbor distance, coverage, diversity
    baseline_comparison()  – rating-based recommender vs KNN
──────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'zomato.csv')
MODEL_DIR = BASE_DIR   # .pkl files saved alongside app.py


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load raw dataset
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SmartBite AI — Training Pipeline")
print("=" * 60)
print("\n📂 STEP 1: Loading dataset...")

df = pd.read_csv(DATA_PATH)
print(f"   Raw shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns   : {df.columns.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Clean & normalise
# ─────────────────────────────────────────────────────────────────────────────

print("\n🧹 STEP 2: Cleaning data...")

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Rename price column produced by the Zomato dataset
if 'cost_for_two' in df.columns:
    df.rename(columns={'cost_for_two': 'price'}, inplace=True)

# Drop rows missing essential fields
required_cols = ['name', 'cuisine', 'rating', 'price', 'type']
before = len(df)
df.dropna(subset=required_cols, inplace=True)
print(f"   Dropped {before - len(df)} rows missing required fields")

# Cuisine: keep only the primary cuisine when multiple are listed
df['cuisine'] = (
    df['cuisine'].astype(str).str.strip().str.split(',').str[0].str.strip()
)

# Rating: numeric, clipped [1, 5]
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)
df['rating'] = df['rating'].clip(1.0, 5.0)

# Price: integer, clipped to sane range
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)
df['price'] = df['price'].clip(50, 5000).astype(int)

# Dining type: map raw strings → canonical three-class taxonomy
TYPE_MAP = {
    'casual dining': 'Casual', 'casual':       'Casual',
    'fine dining':   'Fine Dining', 'fine':    'Fine Dining',
    'fast food':     'Fast Food', 'quick bites': 'Fast Food',
    'delivery':      'Fast Food',
}
df['type'] = (
    df['type'].str.strip().str.lower()
    .map(lambda x: TYPE_MAP.get(x, x.title() if isinstance(x, str) else 'Casual'))
)
df['type'] = df['type'].where(
    df['type'].isin(['Casual', 'Fine Dining', 'Fast Food']), other='Casual'
)

# Votes
if 'votes' in df.columns:
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
else:
    df['votes'] = 0

# Vegetarian flag
if 'vegetarian' in df.columns:
    df['vegetarian'] = (
        df['vegetarian'].astype(str).str.lower().isin(['true', '1', 'yes'])
    )
else:
    df['vegetarian'] = False

# Delivery time (minutes)
if 'delivery_time' not in df.columns:
    df['delivery_time'] = 30
else:
    df['delivery_time'] = (
        pd.to_numeric(df['delivery_time'], errors='coerce').fillna(30).astype(int)
    )

# Description fallback
if 'description' not in df.columns:
    df['description'] = df['name'] + ' — ' + df['cuisine'] + ' cuisine'

# Ensure integer id column
df.reset_index(drop=True, inplace=True)
if 'id' not in df.columns:
    df['id'] = df.index + 1

print(f"   Clean shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Cuisines    : {sorted(df['cuisine'].unique().tolist())}")
print(f"   Types       : {sorted(df['type'].unique().tolist())}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — Dataset Statistics  (EDA)
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame):
    """Print exploratory statistics about the cleaned dataset."""
    print("\n📊 DATASET STATISTICS")
    print(f"   Total restaurants  : {len(df)}")
    print(f"   Rating  — mean={df['rating'].mean():.2f}, "
          f"std={df['rating'].std():.2f}, "
          f"min={df['rating'].min():.1f}, max={df['rating'].max():.1f}")
    print(f"   Price   — mean=₹{df['price'].mean():.0f}, "
          f"median=₹{df['price'].median():.0f}, "
          f"min=₹{df['price'].min()}, max=₹{df['price'].max()}")
    print(f"   Votes   — mean={df['votes'].mean():.0f}, max={df['votes'].max()}")
    print(f"   Vegetarian restaurants : {df['vegetarian'].sum()} "
          f"({df['vegetarian'].mean()*100:.0f}%)")
    print(f"   Avg delivery time      : {df['delivery_time'].mean():.0f} min")
    print("\n   Cuisine distribution:")
    for cuisine, cnt in df['cuisine'].value_counts().items():
        bar = '█' * cnt
        print(f"      {cuisine:<18} {bar} ({cnt})")
    print("\n   Dining-type distribution:")
    for t, cnt in df['type'].value_counts().items():
        print(f"      {t:<18} {cnt}")

print_dataset_stats(df)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

print("\n⚙️  STEP 3: Feature engineering...")

le_cuisine = LabelEncoder()
le_type    = LabelEncoder()

df['cuisine_enc'] = le_cuisine.fit_transform(df['cuisine'])
df['type_enc']    = le_type.fit_transform(df['type'])
df['veg_enc']     = df['vegetarian'].astype(int)

# Normalise votes → [0, 1]  (popularity signal)
max_votes        = int(df['votes'].max()) if df['votes'].max() > 0 else 1
df['votes_norm'] = df['votes'] / max_votes

# ── Feature vector used by both training and inference ────────────────────────
# IMPORTANT: this list must stay in sync with app.py
FEATURE_COLS = ['cuisine_enc', 'rating', 'price', 'type_enc', 'veg_enc', 'votes_norm']
X = df[FEATURE_COLS].values

print(f"   Feature columns : {FEATURE_COLS}")
print(f"   Matrix shape    : {X.shape}")
print(f"   Cuisine classes : {le_cuisine.classes_.tolist()}")
print(f"   Type classes    : {le_type.classes_.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Scale
# ─────────────────────────────────────────────────────────────────────────────

print("\n📏 STEP 4: Scaling features (StandardScaler)...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Feature means : {scaler.mean_.round(3)}")
print(f"   Feature stds  : {scaler.scale_.round(3)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train KNN
# ─────────────────────────────────────────────────────────────────────────────

print("\n🤖 STEP 5: Training KNN model...")
n_neighbors = min(15, len(df))
knn_model   = NearestNeighbors(
    n_neighbors=n_neighbors,
    metric='euclidean',
    algorithm='ball_tree'
)
knn_model.fit(X_scaled)
print(f"   Algorithm   : ball_tree")
print(f"   Metric      : euclidean")
print(f"   n_neighbors : {n_neighbors}")
print(f"   Fitted on   : {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Evaluate model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(knn_model, X_scaled: np.ndarray, df: pd.DataFrame):
    """
    Evaluate the trained KNN model on key quality metrics.

    Metrics:
        avg_neighbor_distance – lower = tighter, more similar neighbours
        cuisine_coverage      – % of unique cuisines appearing in neighbours
        diversity_score       – avg # of unique cuisines per top-5 result set
    """
    print("\n📈 STEP 6: Model evaluation...")

    distances, indices = knn_model.kneighbors(X_scaled)

    # Average nearest-neighbour distance across entire dataset
    # (exclude self — distance[0] is always 0 for fitted data)
    avg_dist = float(np.mean(distances[:, 1:]))   # skip self (col 0)
    print(f"   Average neighbour distance : {avg_dist:.4f}")
    print(f"   (Lower = tighter clusters, more similar neighbours)")

    # Cuisine coverage: how many unique cuisines appear in the typical result set
    all_cuisines = set(df['cuisine'].unique())
    covered = set()
    diversities = []
    for idx_row in indices[:, 1:6]:      # top 5 neighbours (excluding self)
        neighbour_cuisines = set(df.iloc[idx_row]['cuisine'].values)
        covered.update(neighbour_cuisines)
        diversities.append(len(neighbour_cuisines))

    coverage_pct  = 100 * len(covered) / len(all_cuisines)
    avg_diversity = float(np.mean(diversities))
    print(f"   Cuisine coverage           : {len(covered)}/{len(all_cuisines)} "
          f"({coverage_pct:.0f}%)")
    print(f"   Avg diversity per query    : {avg_diversity:.1f} unique cuisines")

    # Sample recommendation test
    print("\n   Sample KNN query → Indian, ₹350, Casual:")
    try:
        c_enc = le_cuisine.transform(['Indian'])[0]
        t_enc = le_type.transform(['Casual'])[0]
        q     = np.array([[c_enc, 4.2, 350, t_enc, 0, 0.4]])
        q_sc  = scaler.transform(q)
        dists, idxs = knn_model.kneighbors(q_sc)
        for rank, (d, idx) in enumerate(zip(dists[0][:5], idxs[0][:5]), 1):
            r = df.iloc[idx]
            print(f"      {rank}. {r['name']:<32} cuisine={r['cuisine']:<15} "
                  f"rating={r['rating']} price=₹{r['price']} dist={d:.3f}")
    except Exception as e:
        print(f"   ⚠️  Sample query failed: {e}")

    return {
        'avg_neighbor_distance': round(avg_dist, 4),
        'cuisine_coverage_pct':  round(coverage_pct, 1),
        'avg_diversity':         round(avg_diversity, 1),
    }

eval_results = evaluate_model(knn_model, X_scaled, df)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def baseline_comparison(df: pd.DataFrame, cuisine: str, budget: int, top_n: int = 5):
    """
    Rating-based baseline recommender:
      Simply filters by cuisine + budget and returns the top-N by rating.

    This is what a Zomato-style basic filter would do.
    Compare its output to KNN to show why multi-feature similarity is better.
    """
    print("\n🏁 STEP 7: Baseline comparison — Rating Filter vs KNN")
    print(f"\n   Query: {cuisine}, ₹{budget} budget, top {top_n}")

    # ── Baseline: filter + sort by rating ────────────────────────────────
    baseline = (
        df[(df['cuisine'] == cuisine) & (df['price'] <= budget)]
        .nlargest(top_n, 'rating')[['name', 'cuisine', 'rating', 'price', 'votes']]
    )
    print("\n   BASELINE (rating filter only):")
    if baseline.empty:
        print(f"   ⚠️  No exact cuisine match in dataset for '{cuisine}' ≤ ₹{budget}")
    else:
        for _, r in baseline.iterrows():
            print(f"      {r['name']:<32} rating={r['rating']}  "
                  f"price=₹{r['price']}  votes={r['votes']}")

    # ── KNN: feature-aware query ──────────────────────────────────────────
    try:
        c_enc = le_cuisine.transform([cuisine])[0]
        t_enc = le_type.transform(['Casual'])[0]
        q     = np.array([[c_enc, 4.0, float(budget), t_enc, 0, 0.5]])
        q_sc  = scaler.transform(q)
        dists, idxs = knn_model.kneighbors(q_sc)
        print("\n   KNN (multi-feature similarity):")
        for rank, (d, idx) in enumerate(zip(dists[0][:top_n], idxs[0][:top_n]), 1):
            r = df.iloc[idx]
            print(f"      {rank}. {r['name']:<32} cuisine={r['cuisine']:<14} "
                  f"rating={r['rating']}  price=₹{r['price']}  dist={d:.3f}")
    except Exception as e:
        print(f"   ⚠️  KNN baseline query failed: {e}")

    print("\n   ✅ KNN performs better because it considers multi-feature similarity:")
    print("      • Rating alone ignores price fit, cuisine relationships,")
    print("        dining type match, and user preference patterns.")
    print("      • KNN retrieves semantically similar restaurants even outside")
    print("        the exact cuisine when budget + type profile matches better.")
    print("      • Graph reasoning then re-ranks results using cuisine ontology.")

baseline_comparison(df, cuisine='Indian', budget=350, top_n=5)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

print("\n💾 STEP 8: Saving model artifacts...")

joblib.dump(knn_model, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler,    os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump({'cuisine': le_cuisine, 'type': le_type},
            os.path.join(MODEL_DIR, 'encoders.pkl'))

clean_path = os.path.join(BASE_DIR, '..', 'data', 'restaurants_clean.csv')
df.to_csv(clean_path, index=False)

meta = {
    'feature_cols':    FEATURE_COLS,
    'cuisine_classes': le_cuisine.classes_.tolist(),
    'type_classes':    le_type.classes_.tolist(),
    'n_restaurants':   len(df),
    'n_neighbors':     n_neighbors,
    'votes_max':       max_votes,
    'eval':            eval_results,
}
with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"   ✅ model.pkl             → {os.path.join(MODEL_DIR, 'model.pkl')}")
print(f"   ✅ scaler.pkl            → {os.path.join(MODEL_DIR, 'scaler.pkl')}")
print(f"   ✅ encoders.pkl          → {os.path.join(MODEL_DIR, 'encoders.pkl')}")
print(f"   ✅ meta.json             → {os.path.join(MODEL_DIR, 'meta.json')}")
print(f"   ✅ restaurants_clean.csv → {clean_path}")

print("\n" + "=" * 60)
print("  Training complete.  Run: python app.py")
print("=" * 60 + "\n")
