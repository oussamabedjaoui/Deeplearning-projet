import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = Path('uci-secom.csv')

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    time_index = pd.to_datetime(df['Time'], errors='coerce')
    features = df.drop(columns=['Time', 'Pass/Fail']).copy()
    
    hour = time_index.dt.hour.astype(float)
    dayofweek = time_index.dt.dayofweek.astype(float)
    month = time_index.dt.month.astype(float)
    
    features['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    features['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)
    features['dow_sin'] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    features['dow_cos'] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    features['month_sin'] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    features['month_cos'] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    
    target = (df['Pass/Fail'] == 1).astype(int)
    
    # Split: 65% train, 17.5% val, 17.5% test (0.35 * 0.5 = 0.175)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features, target, test_size=0.35, stratify=target, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    imputer = SimpleImputer(strategy='median', add_indicator=True)
    scaler = StandardScaler()
    
    x_train_imputed = imputer.fit_transform(x_train)
    x_val_imputed = imputer.transform(x_val)
    x_test_imputed = imputer.transform(x_test)
    
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_val_scaled = scaler.transform(x_val_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)
    
    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test

def get_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision and recall arrays are 1 element longer than thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    f1_scores = np.where((precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls), 0)
    
    valid_mask = precisions >= 0.25
    if np.any(valid_mask):
        best_idx = np.argmax(np.where(valid_mask, f1_scores, -1))
    else:
        best_idx = np.argmax(f1_scores)
        
    return thresholds[best_idx]

def evaluate(model, x, y, threshold):
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, prob),
        'pr_auc': average_precision_score(y, prob)
    }

x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess()

models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'MLPClassifier': MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-3,
        batch_size=64, learning_rate_init=1e-3, max_iter=400, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25, random_state=42
    )
}

results = []
for name, clf in models.items():
    if name == 'MLPClassifier':
        # Replicate the sample weight logic for MLP from original script
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_map = dict(zip(classes, weights))
        sw = y_train.map(weight_map).values
        clf.fit(x_train, y_train, sample_weight=sw)
    else:
        clf.fit(x_train, y_train)
    
    val_prob = clf.predict_proba(x_val)[:, 1]
    threshold = get_best_threshold(y_val, val_prob)
    
    val_metrics = evaluate(clf, x_val, y_val, threshold)
    test_metrics = evaluate(clf, x_test, y_test, threshold)
    
    results.append({
        'Model': name,
        'Threshold': threshold,
        'Val Precision': val_metrics['precision'],
        'Val Recall': val_metrics['recall'],
        'Val F1': val_metrics['f1'],
        'Test Precision': test_metrics['precision'],
        'Test Recall': test_metrics['recall'],
        'Test F1': test_metrics['f1'],
        'Test ROC-AUC': test_metrics['roc_auc'],
        'Test PR-AUC': test_metrics['pr_auc']
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

best_candidate = df_results[df_results['Test Precision'] > 0.25].sort_values('Test F1', ascending=False)
if not best_candidate.empty:
    print(f"\nBest candidate for Precision > 25%: {best_candidate.iloc[0]['Model']}")
else:
    print("\nNo model achieved Test Precision > 25%. Best by Test Precision:")
    print(df_results.sort_values('Test Precision', ascending=False).iloc[0]['Model'])
