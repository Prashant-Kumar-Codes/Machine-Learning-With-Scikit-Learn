## Project: Dual-Engine Credit Card Fraud Detection Pipeline

### 1. What This Is All About
This guide covers how to build a fraud detection system that uses two different approaches, working together to catch suspicious transactions:
1. **Supervised Approach:** A model trained on known fraud cases to spot similar patterns.
2. **Unsupervised Approach:** A model that finds anything unusual, even if we've never seen it before.

The goal is to create a flexible system that can handle imbalanced data (way more legit transactions than fraud), avoid sneaky data leakage issues, and give us metrics that actually matter in the real world.

---

### 2. The Data We're Working With
* **What we're trying to predict:** `Class` (0 = Legit transaction, 1 = Fraud).
* **The imbalance problem:** Fraud is rare—only about 0.172% of transactions are actually fraudulent.
* **The features:** We've got 28 privacy-protected features (`V1` through `V28`) from PCA, plus two regular features: `Time` (how many seconds since the start) and `Amount` (how much money).

---

### 3. Feature Engineering & Validation Strategy

#### 3.1 Avoiding Data Leakage (The Golden Rule)
We need to be super careful about how we split the data:
* **How we split:** 80% for training/validation, 20% as a completely separate test set.
* **Scaling matters:** Any adjustments we make (like scaling `Amount`) need to be calculated only on the training data. Then we apply those same adjustments to validation and testing.

#### 3.2 Feature Tweaks & Transformations
* **Amount feature:** It's really skewed with big outliers. We need scaling techniques that handle extreme values well (using the median and quartile ranges instead of the mean).
* **Time feature:** We can convert the time into sine/cosine waves to capture patterns that repeat throughout the day or week.

---

### 4. The Supervised Approach (Learning from Known Fraud)

#### 4.1 Which Models & How to Train Them
We're using boosting models because they're great at finding patterns in complex data:
* **Models to try:** XGBoost, LightGBM, and CatBoost.
* **What to optimize:** Binary cross-entropy loss, weighted so that missing a fraud case is way more costly than a false alarm.

#### 4.2 Dealing with Imbalanced Data
Since fraud is rare, we need strategies:
* **Cost weighting:** Tell the model that frauds are more important to catch than regular transactions.
* **Oversampling:** Create synthetic fraud examples (using SMOTE or ADASYN) to balance the training data. **Important:** Only do this on the training data, never touch the validation/test data with synthetic examples.

#### 4.3 Finding the Best Settings
* Use `Optuna` to automatically search for the best hyperparameters.
* **What metric matters most:** Precision-Recall Area Under the Curve (PR-AUC). Regular ROC-AUC doesn't work well when data is this imbalanced.

---

### 5. The Unsupervised Approach (Finding What's Weird)

#### 5.1 Which Models Work From Scratch
No need to teach these models what fraud looks like—they'll just find anything that doesn't fit the normal pattern:

* **Isolation Forest:** Finds things that are hard to isolate. We set the contamination factor based on what we expect (about 0.2% fraud).
* **Local Outlier Factor (LOF):** Looks at how dense the neighborhood around each point is. If a point is way less dense than its neighbors, it's probably weird.
* **Autoencoders:** Train on normal transactions only. If the model can't reconstruct a transaction well, it's probably fraud.

#### 5.2 Using Compression to Get Better Features
* The middle layers of an Autoencoder create compressed, cleaned-up versions of the data. We can use these as extra features for our supervised model.

---

### 6. Putting It All Together (Two Models as One)

#### 6.1 Blending Both Approaches
Here's where the magic happens—we combine both models:
1. **Use unsupervised scores as features:** Run the unsupervised models first, get their anomaly scores, then feed those scores into the supervised model.
2. **Blend the results:** Take the probability outputs from both models and combine them (using a technique like logistic calibration).

```plaintext
       [ Input Vector: Raw & PCA Features ]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 [ Unsupervised Engine ]      [ Supervised Engine ]
 (IForest/Autoencoder)         (XGBoost/CatBoost)
        │                             │
  Anomaly Score                 Raw Class Prob
        │                             │
        └──────────────┬──────────────┘
                       ▼
          [ Probability Calibrator ]
                       │
                       ▼
        [ Production Decision Matrix ]
```

---

### 7. Checking How Well It Works

#### 7.1 Metrics That Matter
We're tracking performance in a few different ways:
* **Precision-Recall curves:** How many frauds do we catch at different "strictness" levels? We want to catch at least 85% of frauds.
* **Confusion matrix:** How many false positives (innocent people flagged) vs. false negatives (fraudsters who slipped through)?
* **Brier score:** Is our model giving honest confidence scores, or is it just guessing?

#### 7.2 Picking the Right Decision Threshold
The classic 0.5 threshold doesn't work here. Instead:
* Calculate the cost of missing fraud vs. the cost of flagging an innocent customer.
* Find the threshold that minimizes total cost to the business.

---

### 8. Running This in Production
* **Track everything:** Use MLflow to log every training run, hyperparameters, data versions, and results.
* **Version control:** Use DVC to save and version your trained models and scalers.
* **API layer:** Wrap the final model in a FastAPI service with Pydantic validation to ensure data contracts are met.
* **Deployment:** Package everything in Docker so it can run reliably anywhere.
