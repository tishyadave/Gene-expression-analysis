import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse metadata for labels
    sample_ids = []
    sample_titles = []
    sample_sources = []

    for line in lines:
        clean_line = lambda x: x.strip().strip('"').strip("'")
        parts = [clean_line(s) for s in line.split('\t')[1:]]
        
        if line.startswith('!Sample_geo_accession'):
            sample_ids = parts
        elif line.startswith('!Sample_title'):
            sample_titles = parts
        elif line.startswith('!Sample_source_name_ch1'):
            sample_sources = parts

    # Parse expression matrix
    matrix_start = next(i + 1 for i, line in enumerate(lines) if line.startswith('!series_matrix_table_begin'))
    matrix_end = next(i for i, line in enumerate(lines) if line.startswith('!series_matrix_table_end'))
    matrix_text = ''.join(lines[matrix_start:matrix_end])

    df = pd.read_csv(StringIO(matrix_text), sep='\t', index_col=0)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Generate labels based on metadata
    labels = pd.Series(index=sample_ids, dtype=int, name='disease_status')
    for sid, title, source in zip(sample_ids, sample_titles, sample_sources):
        desc = (title + ' ' + source).lower()
        if any(x in desc for x in ['tumor', 'cancer', 'adenocarcinoma']):
            labels[sid] = 1
        elif any(x in desc for x in ['normal', 'adjacent']):
            labels[sid] = 0
        else:
            labels[sid] = -1

    # Filter undefined samples and align data
    labels = labels[labels != -1]
    common = labels.index.intersection(df.columns)
    
    return df[common].T, labels[common]

def preprocess(X):
    # Variance filtering
    X = X.loc[:, X.var() > 0]
    
    # Log transformation
    if X.max().max() > 100:
        X = np.log2(X + 1)
    
    # Filter low variance genes (bottom 10%)
    threshold = X.var().quantile(0.10)
    return X.loc[:, X.var() > threshold]

def train_evaluate(X, y, n_features=100):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=500, random_state=RANDOM_STATE)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=n_features)),
            ('clf', model)
        ])

        y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        results[name] = {
            'accuracy': accuracy_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'y_true': y,
            'y_prob': y_prob
        }
        
    return results

def get_top_features(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    top_idx = np.argsort(selector.scores_)[-k:][::-1]
    return X.columns[top_idx]

def plot_analysis(X, y, results, top_genes):
    os.makedirs('results/figures', exist_ok=True)
    
    # Figure 1: PCA and Heatmap
    fig1 = plt.figure(figsize=(18, 8))
    gs1 = fig1.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.25)

    # PCA
    ax_pca = fig1.add_subplot(gs1[0])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=['#3498DB', '#E74C3C'], 
                    s=150, alpha=0.8, edgecolor='white', ax=ax_pca)
    ax_pca.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax_pca.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax_pca.set_title('PCA Projection')
    ax_pca.legend(['Normal', 'Cancer'])

    # Heatmap
    ax_heat = fig1.add_subplot(gs1[1])
    idx = np.argsort(y)
    X_top = X[top_genes[:15]].iloc[idx]
    X_z = (X_top - X_top.mean()) / X_top.std()

    sns.heatmap(X_z.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Z-Score'},
                xticklabels=False, yticklabels=True, ax=ax_heat)
    ax_heat.set_title('Top Discriminative Genes')
    ax_heat.set_xlabel('Samples')
    
    plt.savefig('results/figures/biological_signal.png', dpi=300, bbox_inches='tight')

    # Figure 2: Model Performance
    fig2 = plt.figure(figsize=(18, 8))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # ROC Curves
    ax_roc = fig2.add_subplot(gs2[0])
    colors = ['#2C3E50', '#E67E22', '#27AE60']
    
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        ax_roc.plot(fpr, tpr, lw=3, color=colors[i], label=f"{name} (AUC={res['auc']:.3f})")
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend(loc="lower right")

    # Metrics Bar Plot
    ax_bar = fig2.add_subplot(gs2[1])
    df_metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Sensitivity': [r['sensitivity'] for r in results.values()],
        'Specificity': [r['specificity'] for r in results.values()]
    }).melt(id_vars='Model', var_name='Metric', value_name='Score')

    sns.barplot(data=df_metrics, x='Model', y='Score', hue='Metric', palette="viridis", ax=ax_bar)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title('Performance Metrics')
    
    for container in ax_bar.containers:
        ax_bar.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.savefig('results/figures/model_performance.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    filepath = r"C:\Users\Tishya H Dave\Desktop\Tishya\gene_expression_project\data\raw\GSE10072_series_matrix.txt"
    
    if os.path.exists(filepath):
        X, y = load_data(filepath)
        X = preprocess(X)
        
        results = train_evaluate(X, y)
        top_genes = get_top_features(X, y)
        
        plot_analysis(X, y, results, top_genes)
        
        # Save summary
        summary = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC': [r['auc'] for r in results.values()],
            'Accuracy': [r['accuracy'] for r in results.values()]
        }).sort_values('AUC', ascending=False)
        
        summary.to_csv('results/model_performance.csv', index=False)
        print(summary)
    else:
        print(f"File not found: {filepath}")