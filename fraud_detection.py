# ============================================================
#   CREDIT CARD FRAUD DETECTION - COMPLETE PROJECT
#   File: fraud_detection.py
#   Run:  python fraud_detection.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PHASE 1 - LOAD DATA
# ============================================================
print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION - PORTFOLIO PROJECT")
print("=" * 60)
print("\n[Phase 1] Loading dataset...")

df = pd.read_csv('creditcard.csv')

print(f"  Dataset shape   : {df.shape}")
print(f"  Total columns   : {list(df.columns)}")
print(f"  Total fraud     : {df['Class'].sum()}")
print(f"  Fraud percentage: {df['Class'].mean()*100:.4f}%")


# ============================================================
# PHASE 2 - EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[Phase 2] Exploratory Data Analysis...")

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('EDA Overview', fontsize=16, fontweight='bold')

# Plot 1 - Class imbalance bar chart
class_counts = df['Class'].value_counts()
axes[0].bar(['Legit (0)', 'Fraud (1)'], class_counts.values,
            color=['steelblue', 'crimson'], edgecolor='black', linewidth=0.5)
axes[0].set_title('Class Imbalance', fontsize=13)
axes[0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Plot 2 - Transaction amount distribution
df[df['Class'] == 0]['Amount'].clip(upper=500).plot(
    kind='hist', bins=50, ax=axes[1], alpha=0.6, color='steelblue', label='Legit')
df[df['Class'] == 1]['Amount'].clip(upper=500).plot(
    kind='hist', bins=50, ax=axes[1], alpha=0.7, color='crimson', label='Fraud')
axes[1].set_title('Transaction Amount Distribution', fontsize=13)
axes[1].set_xlabel('Amount (clipped at $500)')
axes[1].legend()

# Plot 3 - Scatter over time
axes[2].scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'],
                alpha=0.1, s=1, color='steelblue', label='Legit')
axes[2].scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'],
                alpha=0.5, s=10, color='crimson', label='Fraud')
axes[2].set_title('Fraud vs Legit Over Time', fontsize=13)
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Amount')
axes[2].legend()

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: eda_overview.png")


# ============================================================
# PHASE 3 - STATISTICAL ANALYSIS
# ============================================================
print("\n[Phase 3] Statistical Analysis...")

# --- Z-Score ---
df['Amount_zscore'] = np.abs(stats.zscore(df['Amount']))
high_zscore = df[df['Amount_zscore'] > 3]
print(f"\n  Z-Score (|Z| > 3 threshold):")
print(f"    Anomalies detected : {len(high_zscore)}")
print(f"    Fraud among them   : {high_zscore['Class'].sum()} ({high_zscore['Class'].mean()*100:.1f}%)")

# --- IQR ---
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers = df[df['Amount'] > upper_fence]
print(f"\n  IQR Outlier Detection:")
print(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"    Upper fence        : ${upper_fence:.2f}")
print(f"    Outliers detected  : {len(outliers)}")
print(f"    Fraud among them   : {outliers['Class'].sum()} ({outliers['Class'].mean()*100:.2f}%)")

# --- Hypothesis Testing (t-test) ---
t_stat, p_value = stats.ttest_ind(fraud['Amount'], legit['Amount'])
print(f"\n  Hypothesis Test (Independent t-test on Amount):")
print(f"    H0: Fraud and legit transaction amounts are equal")
print(f"    t-statistic = {t_stat:.4f}")
print(f"    p-value     = {p_value:.8f}")
if p_value < 0.05:
    print(f"    Result: REJECT H0 — amounts are significantly different (p < 0.05)")
else:
    print(f"    Result: Fail to reject H0")

# --- Descriptive stats comparison ---
print(f"\n  Descriptive Stats Comparison (Amount):")
comparison = pd.DataFrame({
    'Legit': legit['Amount'].describe(),
    'Fraud': fraud['Amount'].describe()
}).round(2)
print(comparison.to_string())

# --- Bayes Theorem ---
P_fraud = len(fraud) / len(df)
threshold = 200
P_high_given_fraud = len(fraud[fraud['Amount'] > threshold]) / len(fraud)
P_high_given_legit = len(legit[legit['Amount'] > threshold]) / len(legit)
P_high = (P_high_given_fraud * P_fraud) + (P_high_given_legit * (1 - P_fraud))
P_fraud_given_high = (P_high_given_fraud * P_fraud) / P_high
print(f"\n  Bayes' Theorem — P(Fraud | Amount > ${threshold}):")
print(f"    P(Fraud)                   = {P_fraud:.6f}")
print(f"    P(Amount>200 | Fraud)      = {P_high_given_fraud:.4f}")
print(f"    P(Amount>200 | Legit)      = {P_high_given_legit:.4f}")
print(f"    P(Fraud | Amount>200)      = {P_fraud_given_high:.6f}  <-- posterior probability")

# --- Stats plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')

axes[0].hist(df[df['Class'] == 0]['Amount_zscore'].clip(upper=10),
             bins=60, color='steelblue', alpha=0.6, label='Legit', density=True)
axes[0].hist(df[df['Class'] == 1]['Amount_zscore'].clip(upper=10),
             bins=60, color='crimson', alpha=0.7, label='Fraud', density=True)
axes[0].axvline(x=3, color='orange', linestyle='--', linewidth=2, label='Z=3 threshold')
axes[0].set_title('Z-Score Distribution by Class', fontsize=13)
axes[0].set_xlabel('|Z-score| of Amount')
axes[0].legend()

top_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'Amount', 'Class']
corr = df[top_features].corr()
sns.heatmap(corr, ax=axes[1], annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, annot_kws={'size': 8})
axes[1].set_title('Feature Correlation Heatmap', fontsize=13)

plt.tight_layout()
plt.savefig('stats_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: stats_analysis.png")


# ============================================================
# PHASE 4 - PREPROCESSING & SMOTE
# ============================================================
print("\n[Phase 4] Preprocessing & Handling Class Imbalance...")

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = scaler.fit_transform(df[['Time']])

feature_cols = [c for c in df.columns if c.startswith('V')] + ['Amount_scaled', 'Time_scaled']
X = df[feature_cols]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train size : {len(X_train)} samples")
print(f"  Test size  : {len(X_test)} samples")
print(f"  Fraud in train (before SMOTE): {y_train.sum()}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  Train size after SMOTE : {len(X_train_sm)} samples")
print(f"  Fraud after SMOTE      : {pd.Series(y_train_sm).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('SMOTE: Class Balance Before vs After', fontsize=14, fontweight='bold')
axes[0].bar(['Legit', 'Fraud'],
            [y_train.value_counts()[0], y_train.value_counts()[1]],
            color=['steelblue', 'crimson'])
axes[0].set_title('Before SMOTE')
axes[0].set_ylabel('Count')
axes[1].bar(['Legit', 'Fraud'],
            [pd.Series(y_train_sm).value_counts()[0], pd.Series(y_train_sm).value_counts()[1]],
            color=['steelblue', 'crimson'])
axes[1].set_title('After SMOTE')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('smote_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: smote_comparison.png")


# ============================================================
# PHASE 5 - TRAIN MODELS
# ============================================================
print("\n[Phase 5] Training 3 Models (takes 1-2 minutes)...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model':  model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc':    auc
    }
    print(f"  AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

print("\n  === FINAL MODEL COMPARISON ===")
for name, r in results.items():
    print(f"  {name:25s} --> AUC: {r['auc']:.4f}")


# ============================================================
# PHASE 6 - EVALUATION CHARTS
# ============================================================
print("\n[Phase 6] Generating Evaluation Charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Evaluation — Confusion Matrices & ROC Curves',
             fontsize=16, fontweight='bold')

for idx, (name, r) in enumerate(results.items()):

    # Confusion matrix
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0][idx],
                cmap='Blues', linewidths=0.5,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    axes[0][idx].set_title(f'{name}\nAUC: {r["auc"]:.4f}', fontsize=11)
    axes[0][idx].set_ylabel('Actual')
    axes[0][idx].set_xlabel('Predicted')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    axes[1][idx].plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {r["auc"]:.4f}')
    axes[1][idx].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    axes[1][idx].fill_between(fpr, tpr, alpha=0.1, color='crimson')
    axes[1][idx].set_title(f'ROC Curve — {name}', fontsize=11)
    axes[1][idx].set_xlabel('False Positive Rate')
    axes[1][idx].set_ylabel('True Positive Rate')
    axes[1][idx].legend(loc='lower right')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: model_evaluation.png")


# ============================================================
# SAVE BEST MODEL
# ============================================================
best_name  = max(results, key=lambda x: results[x]['auc'])
best_model = results[best_name]['model']

print(f"\n  Best model: {best_name} (AUC = {results[best_name]['auc']:.4f})")

with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model':    best_model,
        'scaler':   scaler,
        'features': feature_cols
    }, f)

print("  Saved: best_model.pkl")

print("\n" + "=" * 60)
print("  ALL DONE!")
print("  Charts saved: eda_overview.png, stats_analysis.png,")
print("                smote_comparison.png, model_evaluation.png")
print("  Model saved : best_model.pkl")
print()
print("  Next step -> run the dashboard:")
print("  streamlit run app.py")
print("=" * 60)
