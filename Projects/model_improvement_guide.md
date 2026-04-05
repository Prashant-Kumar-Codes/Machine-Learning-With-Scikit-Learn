# <span style="font-size: 18px;">🚀 Comprehensive Guide: Breaking the 86% R² Plateau in Your Flood Prediction Model</span>

## <span style="font-size: 18px;">Current Situation Summary</span>

| <span style="font-size: 18px;">Metric</span>        | <span style="font-size: 18px;">Value</span>                               |
| --------------------------------------------------- | ------------------------------------------------------------------------- |
| <span style="font-size: 18px;">Best R²</span>       | <span style="font-size: 18px;">\~0.866</span>                             |
| <span style="font-size: 18px;">Best RMSE</span>     | <span style="font-size: 18px;">\~0.0187</span>                            |
| <span style="font-size: 18px;">Model</span>         | <span style="font-size: 18px;">XGBoost Regressor</span>                   |
| <span style="font-size: 18px;">Training Rows</span> | <span style="font-size: 18px;">\~1.1M</span>                              |
| <span style="font-size: 18px;">Features</span>      | <span style="font-size: 18px;">20 original integer features</span>        |
| <span style="font-size: 18px;">Target</span>        | <span style="font-size: 18px;">FloodProbability (continuous float)</span> |

> <span style="font-size: 18px;">\[!IMPORTANT]
> This Kaggle competition dataset was generated from a deep learning model trained on the original Flood Prediction Factors dataset. The target FloodProbability appears to be approximately the normalized mean of all 20 features (each feature ranges 0-10). This is a key insight — understanding the data generation process is critical for breaking the plateau.</span>

***

## <span style="font-size: 18px;">1. 🔬 Advanced Feature Engineering Ideas</span>

### <span style="font-size: 18px;">1.1 Exploit the Row-Level Mean/Sum Signal</span>

<span style="font-size: 18px;">Since FloodProbability ≈ sum\_of\_features / (20 \* max\_value), engineer features that capture this:</span>

```python
# Row-wise statistics (you may already have some of these)
features = [col for col in train_df.columns if col not in ['id', 'FloodProbability']]

for df in [train_df, test_df]:
    df['RowMean'] = df[features].mean(axis=1)
    df['RowStd'] = df[features].std(axis=1)
    df['RowMedian'] = df[features].median(axis=1)
    df['RowMax'] = df[features].max(axis=1)
    df['RowMin'] = df[features].min(axis=1)
    df['RowRange'] = df['RowMax'] - df['RowMin']
    df['RowSum'] = df[features].sum(axis=1)
    df['RowSkew'] = df[features].skew(axis=1)
    df['RowKurtosis'] = df[features].kurtosis(axis=1)
    
    # Coefficient of variation (captures relative spread)
    df['RowCV'] = df['RowStd'] / (df['RowMean'] + 1e-8)
    
    # Percentile-based features
    df['Row25'] = df[features].quantile(0.25, axis=1)
    df['Row75'] = df[features].quantile(0.75, axis=1)
    df['RowIQR'] = df['Row75'] - df['Row25']
```

### <span style="font-size: 18px;">1.2 Domain-Specific Grouping & Interaction Features</span>

<span style="font-size: 18px;">Group features by domain meaning and create composite scores:</span>

```python
# Domain groupings for flood prediction
WEATHER_FACTORS = ['MonsoonIntensity', 'ClimateChange']
GEOGRAPHY_FACTORS = ['TopographyDrainage', 'CoastalVulnerability', 'Landslides', 'Watersheds']
HUMAN_FACTORS = ['Urbanization', 'Deforestation', 'AgriculturalPractices', 
                 'Encroachments', 'PopulationScore']
INFRASTRUCTURE = ['RiverManagement', 'DamsQuality', 'DrainageSystems', 
                  'DeterioratingInfrastructure', 'Siltation']
GOVERNANCE = ['IneffectiveDisasterPreparedness', 'InadequatePlanning', 
              'PoliticalFactors', 'WetlandLoss']

for df in [train_df, test_df]:
    # Group means
    df['WeatherScore'] = df[WEATHER_FACTORS].mean(axis=1)
    df['GeographyScore'] = df[GEOGRAPHY_FACTORS].mean(axis=1)
    df['HumanScore'] = df[HUMAN_FACTORS].mean(axis=1)
    df['InfraScore'] = df[INFRASTRUCTURE].mean(axis=1)
    df['GovernanceScore'] = df[GOVERNANCE].mean(axis=1)
    
    # Cross-group interactions (products capture non-linear relationships)
    df['Weather_x_Geography'] = df['WeatherScore'] * df['GeographyScore']
    df['Human_x_Infra'] = df['HumanScore'] * df['InfraScore']
    df['Weather_x_Governance'] = df['WeatherScore'] * df['GovernanceScore']
    
    # High-risk indicator: how many features are above median (5)
    df['HighRiskCount'] = (df[features] > 5).sum(axis=1)
    df['LowRiskCount'] = (df[features] < 3).sum(axis=1)
    df['ExtremeCount'] = ((df[features] >= 8) | (df[features] <= 2)).sum(axis=1)
```

### <span style="font-size: 18px;">1.3 Pairwise Interaction Features (Selective)</span>

<span style="font-size: 18px;">Don't create ALL pairwise interactions (too many). Use feature importance to pick top features:</span>

```python
from itertools import combinations

# After training an initial model, get top N most important features
important_features = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 
                      'ClimateChange', 'DamsQuality']  # adjust based on importance

for f1, f2 in combinations(important_features, 2):
    for df in [train_df, test_df]:
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1)
        df[f'{f1}_minus_{f2}'] = df[f1] - df[f2]
```

### <span style="font-size: 18px;">1.4 Residual-Based Features</span>

<span style="font-size: 18px;">The difference between RowMean/20 and FloodProbability is the "noise" the model needs to capture:</span>

```python
# Train a simple model first, then engineer on residuals
# This is useful for stacking approaches
```

***

## <span style="font-size: 18px;">2. 🏗️ Model Architecture Improvements</span>

### <span style="font-size: 18px;">2.1 Ensemble / Stacking (Most Impactful)</span>

<span style="font-size: 18px;">This is typically the #1 method to break plateaus on Kaggle:</span>

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
import catboost as cb

# Level 1: Diverse base models
models = {
    'xgb': XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.01, ...),
    'lgbm': lgb.LGBMRegressor(n_estimators=1000, max_depth=7, learning_rate=0.01, ...),
    'catboost': cb.CatBoostRegressor(iterations=1000, depth=7, learning_rate=0.01, ...),
    'rf': RandomForestRegressor(n_estimators=500, max_depth=15, ...),
    'et': ExtraTreesRegressor(n_estimators=500, max_depth=15, ...),
}

# Use K-Fold to create out-of-fold predictions for stacking
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {}
test_preds = {}

for name, model in models.items():
    oof_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        oof_pred[val_idx] = model.predict(X_val)
        test_pred += model.predict(X_test) / kf.n_splits
    
    oof_preds[name] = oof_pred
    test_preds[name] = test_pred

# Level 2: Meta-learner (Ridge regression works well)
stack_train = pd.DataFrame(oof_preds)
stack_test = pd.DataFrame(test_preds)

meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, Y_train)
final_pred = meta_model.predict(stack_test)
```

### <span style="font-size: 18px;">2.2 Try LightGBM and CatBoost</span>

<span style="font-size: 18px;">Different gradient boosting implementations often capture different patterns:</span>

```python
# LightGBM — often faster and handles large datasets better
import lightgbm as lgb

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'max_depth': 8,
    'learning_rate': 0.01,
    'num_leaves': 63,  # 2^max_depth - 1
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'n_jobs': -1,
    'random_state': 42,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

# CatBoost — often best out-of-the-box
import catboost as cb

cb_params = {
    'iterations': 2000,
    'depth': 8,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
}

cb_model = cb.CatBoostRegressor(**cb_params)
```

### <span style="font-size: 18px;">2.3 Simple Weighted Average Ensemble</span>

<span style="font-size: 18px;">Even without full stacking, a weighted average of diverse models helps:</span>

```python
# After training XGBoost, LightGBM, CatBoost separately
final_pred = 0.4 * xgb_pred + 0.35 * lgb_pred + 0.25 * cb_pred

# Or use scipy.optimize to find optimal weights
from scipy.optimize import minimize

def rmse_objective(weights, preds, true):
    blended = sum(w * p for w, p in zip(weights, preds))
    return np.sqrt(np.mean((true - blended) ** 2))

result = minimize(rmse_objective, x0=[1/3, 1/3, 1/3], 
                  args=([xgb_oof, lgb_oof, cb_oof], Y_train),
                  method='Nelder-Mead')
optimal_weights = result.x / result.x.sum()
```

***

## <span style="font-size: 18px;">3. ⚙️ Hyperparameter Tuning</span>

### <span style="font-size: 18px;">3.1 Bayesian Optimization with Optuna</span>

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42,
        'early_stopping_rounds': 50,
    }
    
    model = XGBRegressor(**params)
    
    # Use cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train, Y_train, 
                             cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)
```

### <span style="font-size: 18px;">3.2 Key XGBoost Parameters to Explore Further</span>

| <span style="font-size: 18px;">Parameter</span>          | <span style="font-size: 18px;">Current</span> | <span style="font-size: 18px;">Try Range</span>  | <span style="font-size: 18px;">Why</span>                                  |
| -------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------- |
| <span style="font-size: 18px;">max\_depth</span>         | <span style="font-size: 18px;">7</span>       | <span style="font-size: 18px;">6-10</span>       | <span style="font-size: 18px;">Deeper trees capture more complexity</span> |
| <span style="font-size: 18px;">n\_estimators</span>      | <span style="font-size: 18px;">1000</span>    | <span style="font-size: 18px;">2000-5000</span>  | <span style="font-size: 18px;">More trees with lower learning rate</span>  |
| <span style="font-size: 18px;">learning\_rate</span>     | <span style="font-size: 18px;">0.01</span>    | <span style="font-size: 18px;">0.005-0.03</span> | <span style="font-size: 18px;">Lower LR + more trees = better</span>       |
| <span style="font-size: 18px;">min\_child\_weight</span> | <span style="font-size: 18px;">12</span>      | <span style="font-size: 18px;">5-30</span>       | <span style="font-size: 18px;">Controls overfitting</span>                 |
| <span style="font-size: 18px;">colsample\_bylevel</span> | <span style="font-size: 18px;">default</span> | <span style="font-size: 18px;">0.5-0.9</span>    | <span style="font-size: 18px;">Per-level feature sampling</span>           |
| <span style="font-size: 18px;">max\_bin</span>           | <span style="font-size: 18px;">default</span> | <span style="font-size: 18px;">128-512</span>    | <span style="font-size: 18px;">Histogram resolution</span>                 |
| <span style="font-size: 18px;">grow\_policy</span>       | <span style="font-size: 18px;">default</span> | <span style="font-size: 18px;">lossguide</span>  | <span style="font-size: 18px;">Different tree growth strategy</span>       |

***

## <span style="font-size: 18px;">4. 🔄 Cross-Validation Strategy</span>

### <span style="font-size: 18px;">4.1 Use Proper K-Fold CV</span>

<span style="font-size: 18px;">With 1.1M rows, you have plenty of data. Use robust CV:</span>

```python
from sklearn.model_selection import KFold, RepeatedKFold

# Standard 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# More robust: Repeated KFold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Track both train and validation scores per fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # Train and evaluate
    # Average final predictions across folds
    pass
```

### <span style="font-size: 18px;">4.2 Seed Averaging</span>

<span style="font-size: 18px;">Train the same model with different random seeds and average:</span>

```python
seeds = [42, 57, 123, 456, 789]
test_preds = []

for seed in seeds:
    model = XGBRegressor(random_state=seed, ...)
    model.fit(X_train, Y_train, ...)
    test_preds.append(model.predict(X_test))

final_pred = np.mean(test_preds, axis=0)
```

***

## <span style="font-size: 18px;">5. 🎯 Target Engineering</span>

### <span style="font-size: 18px;">5.1 Target Transformation</span>

<span style="font-size: 18px;">If the target distribution is skewed, transforming it can help:</span>

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox or Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson')
Y_train_transformed = pt.fit_transform(Y_train.values.reshape(-1, 1)).ravel()

# Train model on transformed target
model.fit(X_train, Y_train_transformed)

# Inverse transform predictions
pred_transformed = model.predict(X_test)
final_pred = pt.inverse_transform(pred_transformed.reshape(-1, 1)).ravel()
```

### <span style="font-size: 18px;">5.2 Log Transform (if target is right-skewed)</span>

```python
import numpy as np

Y_train_log = np.log1p(Y_train)
# Train model on log target
# Inverse: np.expm1(predictions)
```

***

## <span style="font-size: 18px;">6. 📊 Feature Selection</span>

### <span style="font-size: 18px;">6.1 Remove Noisy Features</span>

<span style="font-size: 18px;">Too many weak features can hurt:</span>

```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_train, Y_train, random_state=42)
mi_df = pd.DataFrame({'feature': X_train.columns, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)

# Keep only features with MI score above threshold
good_features = mi_df[mi_df['mi_score'] > 0.01]['feature'].tolist()
```

### <span style="font-size: 18px;">6.2 Recursive Feature Elimination</span>

```python
from sklearn.feature_selection import RFECV

selector = RFECV(XGBRegressor(n_estimators=100, max_depth=5), 
                 step=1, cv=3, scoring='neg_root_mean_squared_error')
selector.fit(X_train, Y_train)
selected_features = X_train.columns[selector.support_].tolist()
```

***

## <span style="font-size: 18px;">7. 🔧 Data-Level Improvements</span>

### <span style="font-size: 18px;">7.1 Use the Original Dataset</span>

<span style="font-size: 18px;">The competition says: "Feel free to use the original dataset to see whether incorporating the original in training improves model performance."</span>

```python
# Load original Flood Prediction Factors dataset from Kaggle
original_df = pd.read_csv('original_flood_prediction.csv')

# Combine with competition training data
combined_train = pd.concat([train_df, original_df], ignore_index=True)
```

### <span style="font-size: 18px;">7.2 Pseudo-Labeling</span>

<span style="font-size: 18px;">Use confident predictions on test data as additional training data:</span>

```python
# Train initial model
model.fit(X_train, Y_train)
test_pred = model.predict(X_test)

# Select high-confidence predictions (close to 0 or 1, or low uncertainty)
confident_mask = (test_pred < 0.2) | (test_pred > 0.8)
pseudo_X = X_test[confident_mask]
pseudo_Y = pd.Series(test_pred[confident_mask])

# Add pseudo-labels to training
X_train_extended = pd.concat([X_train, pseudo_X])
Y_train_extended = pd.concat([Y_train, pseudo_Y])

# Retrain
model.fit(X_train_extended, Y_train_extended)
```

***

## <span style="font-size: 18px;">8. 📋 Recommended Action Plan (Priority Order)</span>

> <span style="font-size: 18px;">\[!TIP]
> For maximum impact with minimum effort, follow this order:</span>

1. <span style="font-size: 18px;">🥇 Ensemble/Stacking — Train XGBoost + LightGBM + CatBoost and blend predictions. This alone can gain 1-3% R².</span>
2. <span style="font-size: 18px;">🥈 Row-Level Statistics — Add RowMean, RowSum, RowStd, etc. These directly capture the data generation signal.</span>
3. <span style="font-size: 18px;">🥉 Domain Grouping — Create composite scores for Weather, Geography, Human, Infrastructure, Governance factors.</span>
4. <span style="font-size: 18px;">4th: Hyperparameter Tuning — Use Optuna for Bayesian search on your best model.</span>
5. <span style="font-size: 18px;">5th: K-Fold Cross-Validation — Use proper 5-fold CV instead of single train/val split.</span>
6. <span style="font-size: 18px;">6th: Original Dataset — Incorporate the original Flood Prediction dataset.</span>
7. <span style="font-size: 18px;">7th: Seed Averaging — Average predictions from 5+ different random seeds.</span>
8. <span style="font-size: 18px;">8th: Feature Selection — Remove noisy engineered features that don't help.</span>

> <span style="font-size: 18px;">\[!WARNING]
> Be careful not to overfit on the validation set. Always use cross-validation to estimate your score, and make sure the public leaderboard score aligns with your local CV score.</span>

***

## <span style="font-size: 18px;">9. Quick Win: Simple Blending Template</span>

```python
# Install if needed: pip install lightgbm catboost optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_blend(X_train, Y_train, X_test, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    xgb_oof = np.zeros(len(X_train))
    xgb_test = np.zeros(len(X_test))
    lgb_oof = np.zeros(len(X_train))
    lgb_test = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = Y_train.iloc[tr_idx], Y_train.iloc[val_idx]
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=2000, max_depth=7, learning_rate=0.01,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=12,
            gamma=0.2, reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', n_jobs=-1, random_state=42,
            early_stopping_rounds=50
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        xgb_oof[val_idx] = xgb_model.predict(X_val)
        xgb_test += xgb_model.predict(X_test) / n_folds
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000, max_depth=7, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42,
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_oof[val_idx] = lgb_model.predict(X_val)
        lgb_test += lgb_model.predict(X_test) / n_folds
        
        print(f"Fold {fold}: XGB RMSE={np.sqrt(mean_squared_error(y_val, xgb_oof[val_idx])):.6f}, "
              f"LGB RMSE={np.sqrt(mean_squared_error(y_val, lgb_oof[val_idx])):.6f}")
    
    # Simple weighted average
    best_blend = 0.5 * xgb_test + 0.5 * lgb_test
    
    xgb_rmse = np.sqrt(mean_squared_error(Y_train, xgb_oof))
    lgb_rmse = np.sqrt(mean_squared_error(Y_train, lgb_oof))
    blend_rmse = np.sqrt(mean_squared_error(Y_train, 0.5*xgb_oof + 0.5*lgb_oof))
    
    print(f"\nOverall XGB RMSE: {xgb_rmse:.6f}")
    print(f"Overall LGB RMSE: {lgb_rmse:.6f}")
    print(f"Blend RMSE: {blend_rmse:.6f}")
    
    return best_blend
```

​
