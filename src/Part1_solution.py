import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error

# Load data
X_train = pd.read_csv('dataset_50.csv')
y_data = pd.read_csv('target_50.csv')
X_eval = pd.read_csv('EVAL_50.csv')
y_train = y_data['target01']

print("="*60)
print("Part 1: Predicting target01")
print("="*60)
print(f"Training data shape: {X_train.shape}")
print(f"Evaluation data shape: {X_eval.shape}")

# Step 1: 5-Fold Cross-Validation
print("\nStep 1: Running 5-Fold Cross-Validation...")
print("-"*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_predictions = np.zeros(len(y_train))

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # Split data
    X_tr_fold = X_train.iloc[train_idx]
    y_tr_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Feature selection on training fold
    cat_select = CatBoostRegressor(n_estimators=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)
    cat_select.fit(X_tr_fold, y_tr_fold)
    importances = cat_select.feature_importances_
    top_indices = np.argsort(importances)[-10:]
    top_features = X_train.columns[top_indices].tolist()
    
    # Polynomial features (degree=3)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr_fold[top_features])
    X_val_poly = poly.transform(X_val_fold[top_features])
    
    other_features = [f for f in X_train.columns if f not in top_features]
    X_tr_combined = np.hstack([X_tr_fold[other_features].values, X_tr_poly])
    X_val_combined = np.hstack([X_val_fold[other_features].values, X_val_poly])
    
    # Train model
    model = CatBoostRegressor(n_estimators=800, learning_rate=0.05, depth=10, random_state=42, verbose=0)
    model.fit(X_tr_combined, y_tr_fold)
    
    # Predictions
    train_pred = model.predict(X_tr_combined)
    val_pred = model.predict(X_val_combined)
    cv_predictions[val_idx] = val_pred
    
    # Calculate metrics for this fold
    train_rmse = np.sqrt(mean_squared_error(y_tr_fold, train_pred))
    train_mae = mean_absolute_error(y_tr_fold, train_pred)
    train_r2 = r2_score(y_tr_fold, train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    val_mae = mean_absolute_error(y_val_fold, val_pred)
    val_r2 = r2_score(y_val_fold, val_pred)
    
    fold_metrics.append({
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2
    })
    
    print(f"\nFold {fold+1}/5:")
    print(f"  TRAIN: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, R²={train_r2:.4f}")
    print(f"  VAL:   RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}")

# Overall CV metrics
cv_rmse = np.sqrt(mean_squared_error(y_train, cv_predictions))
cv_mae = mean_absolute_error(y_train, cv_predictions)
cv_r2 = r2_score(y_train, cv_predictions)
cv_ev = explained_variance_score(y_train, cv_predictions)

print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)
print(f"\nPer-Fold Validation R²: {[round(f['val_r2'], 4) for f in fold_metrics]}")
print(f"Mean Val R²: {np.mean([f['val_r2'] for f in fold_metrics]):.4f}")
print(f"Std Val R²:  {np.std([f['val_r2'] for f in fold_metrics]):.4f}")
print(f"\nOverall CV Metrics:")
print(f"  RMSE:               {cv_rmse:.4f}")
print(f"  MAE:                {cv_mae:.4f}")
print(f"  R² Score:           {cv_r2:.4f}")
print(f"  Explained Variance: {cv_ev:.4f}")
print("="*60)

# Step 2: Train final model on all data
print("\nStep 2: Training final model on full training data...")

cat_final_select = CatBoostRegressor(n_estimators=300, learning_rate=0.05, depth=6, random_state=42, verbose=0)
cat_final_select.fit(X_train, y_train)
importances = cat_final_select.feature_importances_
top_indices = np.argsort(importances)[-10:]
top_features = X_train.columns[top_indices].tolist()
print(f"Top 10 features: {top_features}")

poly_final = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly_final.fit_transform(X_train[top_features])
X_poly_eval = poly_final.transform(X_eval[top_features])

other_features = [f for f in X_train.columns if f not in top_features]
X_combined_train = np.hstack([X_train[other_features].values, X_poly_train])
X_combined_eval = np.hstack([X_eval[other_features].values, X_poly_eval])
print(f"Combined features: {X_combined_train.shape[1]}")

final_model = CatBoostRegressor(n_estimators=800, learning_rate=0.05, depth=10, random_state=42, verbose=100)
final_model.fit(X_combined_train, y_train)

# Step 3: Generate predictions
print("\nStep 3: Generating predictions...")
final_predictions = final_model.predict(X_combined_eval)
submission = pd.DataFrame({'target01': final_predictions})
submission.to_csv('EVAL_target01_50.csv', index=False)

print("\n" + "="*60)
print("OUTPUT")
print("="*60)
print(f"Predictions saved to: EVAL_target01_50.csv")
print(f"Prediction count: {len(final_predictions)}")
print(f"Prediction range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"Prediction mean:  {final_predictions.mean():.4f}")

# Step 4: KDE Distribution Plots
print("\nStep 4: Generating distribution plots...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Data - Actual vs CV Predicted
axes[0].hist(y_train, bins=50, alpha=0.5, density=True, label='Actual', color='gray', edgecolor='black')
axes[0].hist(cv_predictions, bins=50, alpha=0.5, density=True, label='CV Predicted', color='blue')
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'Training Data (dataset_50.csv)\nActual vs CV Predicted | R²={cv_r2:.4f}', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Eval Data - Training Actual vs Eval Predicted
axes[1].hist(y_train, bins=50, alpha=0.5, density=True, label='Train Actual', color='gray', edgecolor='black')
axes[1].hist(final_predictions, bins=50, alpha=0.5, density=True, label='Eval Predicted', color='green')
axes[1].set_xlabel('Value', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'EVAL Data (EVAL_50.csv)\nTrain Actual vs Eval Predicted', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
