"""
Rule Extraction Script for Part 2 - target02
This script automatically discovers the rules for predicting target02.

Author: Rutvikkumar Babubhai Balar
Course: Machine Learning - Winter 2025
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)


# ============================================================================
# STEP 1: Load the data
# ============================================================================
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

X = pd.read_csv('dataset_50.csv')
y = pd.read_csv('target_50.csv')['target02'].values
feature_names = X.columns.tolist()

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")


# ============================================================================
# STEP 2: Feature Importance Analysis (8 Methods)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Feature Importance Analysis")
print("=" * 60)

# Store results from each method
importance_results = {}
X_arr = X.values

# Method 1: Decision Tree
print("\n[1] Decision Tree...")
dt = DecisionTreeRegressor(max_depth=5, random_state=SEED)
dt.fit(X_arr, y)
importance_results['DecisionTree'] = dt.feature_importances_

# Method 2: Random Forest
print("[2] Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
rf.fit(X_arr, y)
importance_results['RandomForest'] = rf.feature_importances_

# Method 3: Extra Trees
print("[3] Extra Trees...")
et = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
et.fit(X_arr, y)
importance_results['ExtraTrees'] = et.feature_importances_

# Method 4: Gradient Boosting
print("[4] Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=SEED)
gb.fit(X_arr, y)
importance_results['GradientBoosting'] = gb.feature_importances_

# Method 5: AdaBoost
print("[5] AdaBoost...")
ab = AdaBoostRegressor(n_estimators=50, random_state=SEED)
ab.fit(X_arr, y)
importance_results['AdaBoost'] = ab.feature_importances_

# Method 6: Mutual Information
print("[6] Mutual Information...")
mi_scores = mutual_info_regression(X_arr, y, random_state=SEED)
importance_results['MutualInfo'] = mi_scores

# Method 7: F-Regression
print("[7] F-Regression...")
f_scores, _ = f_regression(X_arr, y)
importance_results['F_Regression'] = f_scores

# Method 8: Correlation
print("[8] Correlation...")
correlations = np.array([abs(np.corrcoef(X_arr[:, i], y)[0, 1]) for i in range(X.shape[1])])
importance_results['Correlation'] = correlations


# ============================================================================
# STEP 3: Find Consensus Features
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Finding Consensus Features")
print("=" * 60)

# Count how many times each feature appears in top 5 across all methods
feature_votes = {f: 0 for f in feature_names}

print("\nTop 5 features per method:")
for method_name, scores in importance_results.items():
    top5_idx = np.argsort(scores)[-5:][::-1]
    top5_names = [feature_names[i] for i in top5_idx]
    print(f"  {method_name:18s}: {top5_names}")
    
    for idx in top5_idx:
        feature_votes[feature_names[idx]] += 1

# Sort by votes
sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

print("\nFeature consensus ranking:")
for feat, votes in sorted_features[:6]:
    print(f"  {feat}: {votes}/8 methods")

# Get consensus features (appear in 6+ methods)
consensus_features = [f for f, v in sorted_features if v >= 6]
print(f"\nCONSENSUS FEATURES: {consensus_features}")


# ============================================================================
# STEP 4: Identify Switcher Variable
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Identifying Switcher Variable")
print("=" * 60)

# The feature that appears as root node in decision trees is the switcher
# Check which feature has highest overall importance
switcher_feature = consensus_features[0]
switcher_idx = feature_names.index(switcher_feature)

# Verify by checking decision tree root node
root_feature_idx = dt.tree_.feature[0]
root_feature = feature_names[root_feature_idx]

print(f"Feature with highest consensus: {switcher_feature}")
print(f"Decision tree root node feature: {root_feature}")

if root_feature == switcher_feature:
    print("CONFIRMED: Switcher variable identified correctly!")
else:
    print("WARNING: Root feature differs from highest consensus feature")
    switcher_feature = root_feature
    switcher_idx = root_feature_idx

# The other features are used for linear calculations
calc_features = [f for f in consensus_features if f != switcher_feature]
calc_indices = [feature_names.index(f) for f in calc_features]

print(f"\nSwitcher variable: {switcher_feature} (index {switcher_idx})")
print(f"Calculation features: {calc_features}")
print(f"Calculation indices: {calc_indices}")


# ============================================================================
# STEP 5: Threshold Discovery
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Threshold Discovery")
print("=" * 60)

switcher_values = X_arr[:, switcher_idx]

# Extract thresholds from decision tree
print("\nExtracting thresholds from decision tree splits...")
tree_thresholds = []
for node_idx in range(len(dt.tree_.feature)):
    if dt.tree_.feature[node_idx] == switcher_idx:
        tree_thresholds.append(dt.tree_.threshold[node_idx])

tree_thresholds = sorted(set([round(t, 2) for t in tree_thresholds]))
print(f"Thresholds from tree: {tree_thresholds}")

# Grid search to find optimal thresholds
print("\nGrid search for optimal thresholds...")
candidates = np.arange(0.1, 0.9, 0.05)
best_score = 0
best_thresholds = None

for t1 in candidates:
    for t2 in candidates:
        for t3 in candidates:
            if not (t1 < t2 < t3):
                continue
            
            # Create segment masks
            masks = [
                switcher_values <= t1,
                (switcher_values > t1) & (switcher_values <= t2),
                (switcher_values > t2) & (switcher_values <= t3),
                switcher_values > t3
            ]
            
            # Skip if any segment is too small
            if any(m.sum() < 50 for m in masks):
                continue
            
            # Calculate segmentation quality
            ss_res = sum(y[m].var() * m.sum() for m in masks if m.sum() > 0)
            ss_tot = y.var() * len(y)
            r2 = 1 - ss_res / ss_tot
            
            if r2 > best_score:
                best_score = r2
                best_thresholds = (round(t1, 2), round(t2, 2), round(t3, 2))

t1, t2, t3 = best_thresholds
print(f"Best thresholds found: {best_thresholds}")
print(f"Segmentation quality (R²): {best_score:.4f}")


# ============================================================================
# STEP 6: Coefficient Extraction via Linear Regression
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Coefficient Extraction")
print("=" * 60)

# Define segments
segments = [
    (f"{switcher_feature} <= {t1}", switcher_values <= t1),
    (f"{t1} < {switcher_feature} <= {t2}", (switcher_values > t1) & (switcher_values <= t2)),
    (f"{t2} < {switcher_feature} <= {t3}", (switcher_values > t2) & (switcher_values <= t3)),
    (f"{switcher_feature} > {t3}", switcher_values > t3)
]

discovered_rules = []

for seg_name, seg_mask in segments:
    X_seg = X_arr[seg_mask][:, calc_indices]
    y_seg = y[seg_mask]
    
    print(f"\n{seg_name}")
    print(f"  Samples: {len(y_seg)}")
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_seg, y_seg)
    
    # Calculate R² on this segment
    y_pred = lr.predict(X_seg)
    seg_r2 = r2_score(y_seg, y_pred)
    
    print(f"  R² on segment: {seg_r2:.6f}")
    print(f"  Intercept: {lr.intercept_:.6f}")
    print(f"  Coefficients:")
    
    coeffs = {}
    for i, feat in enumerate(calc_features):
        coef = lr.coef_[i]
        coeffs[feat] = round(coef, 2)
        print(f"    {feat}: {coef:.4f} -> rounded: {round(coef, 2)}")
    
    discovered_rules.append({
        'condition': seg_name,
        'coefficients': coeffs,
        'intercept': round(lr.intercept_, 2),
        'r2': seg_r2
    })


# ============================================================================
# STEP 7: Validation
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: Validation")
print("=" * 60)

# Apply discovered rules to predict
def apply_rules(X, switcher_idx, calc_indices, t1, t2, t3, rules):
    predictions = []
    for i in range(len(X)):
        row = X[i]
        switcher_val = row[switcher_idx]
        
        # Select rule based on switcher value
        if switcher_val <= t1:
            rule = rules[0]
        elif switcher_val <= t2:
            rule = rules[1]
        elif switcher_val <= t3:
            rule = rules[2]
        else:
            rule = rules[3]
        
        # Apply linear formula
        pred = rule['intercept']
        for j, feat in enumerate(calc_features):
            pred += rule['coefficients'][feat] * row[calc_indices[j]]
        
        predictions.append(pred)
    
    return np.array(predictions)

# Predict and validate
y_pred = apply_rules(X_arr, switcher_idx, calc_indices, t1, t2, t3, discovered_rules)
overall_r2 = r2_score(y, y_pred)
max_error = np.max(np.abs(y - y_pred))

print(f"\nValidation on full dataset:")
print(f"  R² Score: {overall_r2:.10f}")
print(f"  Max Absolute Error: {max_error:.2e}")

if overall_r2 > 0.9999:
    print("\n  Result: EXACT MATCH (rules are correct)")
else:
    print("\n  Result: Approximate match (rules may need refinement)")


# ============================================================================
# STEP 8: Print Final Rules
# ============================================================================
print("\n" + "=" * 60)
print("FINAL DISCOVERED RULES")
print("=" * 60)

print(f"\nSwitcher variable: {switcher_feature} (index {switcher_idx})")
print(f"Thresholds: {t1}, {t2}, {t3}")
print(f"Calculation features: {calc_features}")
print(f"Feature indices: {calc_indices}")

for i, rule in enumerate(discovered_rules, 1):
    print(f"\nRule {i}: If {rule['condition']}")
    
    formula_parts = []
    for feat in calc_features:
        coef = rule['coefficients'][feat]
        if coef >= 0:
            formula_parts.append(f"{coef} * {feat}")
        else:
            formula_parts.append(f"({coef}) * {feat}")
    
    formula = " + ".join(formula_parts)
    print(f"  target02 = {formula}")

print("\n" + "=" * 60)
print("Script completed successfully!")
print("=" * 60)
