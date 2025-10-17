import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             confusion_matrix, classification_report, precision_recall_curve,
                             auc, f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
from scipy import stats

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
warnings.filterwarnings('ignore')

print("= " * 60)
print("MODELO DE PREVISÃO NBA - COM FEATURES DE JOGADORES (VERSÃO COMPLETA)")
print("=" * 60)

# Fixar seeds para reprodutibilidade
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def generate_requirements():
    """Gera arquivo requirements.txt para reprodutibilidade"""
    requirements = """catboost>=1.0.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
shap>=0.40.0
seaborn>=0.11.0
scipy>=1.7.0
openpyxl>=3.0.0
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Arquivo requirements.txt gerado")

def purged_kfold_validation(df, n_splits=5, embargo_days=3):
    """Implementa Purged K-Fold com embargo para dados temporais"""
    # Garantir que o DataFrame está ordenado por data e com índice resetado
    df_sorted = df.sort_values('date').reset_index(drop=True)

    if len(df_sorted) == 0:
        return []

    # Usar as datas normalizadas para evitar problemas de timezone
    df_sorted['date_norm'] = df_sorted['date'].dt.normalize()
    unique_dates = df_sorted['date_norm'].unique()

    if len(unique_dates) < n_splits + 1:
        print(f"  ⚠️ Poucas datas únicas ({len(unique_dates)}) para {n_splits} splits")
        return []

    folds = []
    date_ranges = []

    # Determinar os intervalos de datas para cada fold
    for i in range(n_splits):
        test_size = len(unique_dates) // n_splits
        test_start_idx = i * test_size
        test_end_idx = (i + 1) * test_size if i < n_splits - 1 else len(unique_dates)

        test_dates = unique_dates[test_start_idx:test_end_idx]

        if len(test_dates) == 0:
            continue

        test_start_date = pd.to_datetime(test_dates[0])
        test_end_date = pd.to_datetime(test_dates[-1])

        # Aplicar embargo
        embargo_start = test_start_date - pd.Timedelta(days=embargo_days)
        embargo_end = test_end_date + pd.Timedelta(days=embargo_days)

        date_ranges.append({
            'fold': i,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'embargo_start': embargo_start,
            'embargo_end': embargo_end
        })

    # Para cada fold, criar máscaras baseadas nas datas
    for dr in date_ranges:
        # Train: datas antes do embargo_start OU depois do embargo_end
        train_mask = (
                (df_sorted['date_norm'] < dr['embargo_start']) |
                (df_sorted['date_norm'] > dr['embargo_end'])
        )

        # Test: datas dentro do período de teste
        test_mask = (
                (df_sorted['date_norm'] >= dr['test_start']) &
                (df_sorted['date_norm'] <= dr['test_end'])
        )

        train_idx = df_sorted[train_mask].index.tolist()
        test_idx = df_sorted[test_mask].index.tolist()

        if len(train_idx) >= 50 and len(test_idx) >= 10:
            folds.append((train_idx, test_idx))
        else:
            print(f"  ⚠️ Fold {dr['fold']} ignorado: treino={len(train_idx)}, teste={len(test_idx)}")

    print(f"  ✅ {len(folds)} folds válidos criados")
    return folds

def plot_calibration_analysis(y_true, y_pred_proba_raw, y_pred_proba_cal=None, model_name="Modelo"):
    """Plota análise completa de calibração"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Curva de calibração
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_pred_proba_raw, n_bins=10, strategy='quantile')

    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfeitamente calibrado')
    axes[0, 0].plot(prob_pred_raw, prob_true_raw, 's-', label=f'{model_name} (Bruto)')

    if y_pred_proba_cal is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_pred_proba_cal, n_bins=10, strategy='quantile')
        axes[0, 0].plot(prob_pred_cal, prob_true_cal, 's-', label=f'{model_name} (Calibrado)')

    axes[0, 0].set_xlabel('Probabilidade Média Prevista')
    axes[0, 0].set_ylabel('Fração de Positivos')
    axes[0, 0].set_title('Curva de Calibração')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Histograma das probabilidades
    axes[0, 1].hist(y_pred_proba_raw, bins=20, alpha=0.7, label='Bruto', color='blue')
    if y_pred_proba_cal is not None:
        axes[0, 1].hist(y_pred_proba_cal, bins=20, alpha=0.7, label='Calibrado', color='red')
    axes[0, 1].set_xlabel('Probabilidade Prevista')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição das Probabilidades')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba_raw)
    pr_auc = auc(recall, precision)
    axes[1, 0].plot(recall, precision, label=f'Bruto (AUC={pr_auc:.3f})')

    if y_pred_proba_cal is not None:
        precision_cal, recall_cal, _ = precision_recall_curve(y_true, y_pred_proba_cal)
        pr_auc_cal = auc(recall_cal, precision_cal)
        axes[1, 0].plot(recall_cal, precision_cal, label=f'Calibrado (AUC={pr_auc_cal:.3f})')

    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Curva Precision-Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Lift Chart
    percentiles = np.linspace(0, 100, 21)
    thresholds = np.percentile(y_pred_proba_raw, percentiles)
    lift_values = []
    for threshold in thresholds[:-1]:
        pred_pos = y_pred_proba_raw >= threshold
        if pred_pos.sum() > 0:
            lift = y_true[pred_pos].mean() / y_true.mean()
        else:
            lift = 0
        lift_values.append(lift)

    axes[1, 1].plot(percentiles[:-1], lift_values, 'o-')
    axes[1, 1].axhline(1, color='red', linestyle='--', label='Baseline')
    axes[1, 1].set_xlabel('Percentil')
    axes[1, 1].set_ylabel('Lift')
    axes[1, 1].set_title('Gráfico de Lift')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return fig

def plot_confusion_matrix_analysis(y_true, y_pred, y_pred_cal=None, model_name="Modelo"):
    """Plota análise detalhada da matriz de confusão"""
    if y_pred_cal is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Matriz de confusão bruta
        cm_raw = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Matriz de Confusão - {model_name} (Bruto)')
        axes[0].set_xlabel('Predito')
        axes[0].set_ylabel('Real')

        # Matriz de confusão calibrada
        cm_cal = confusion_matrix(y_true, y_pred_cal)
        sns.heatmap(cm_cal, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title(f'Matriz de Confusão - {model_name} (Calibrado)')
        axes[1].set_xlabel('Predito')
        axes[1].set_ylabel('Real')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_raw = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matriz de Confusão - {model_name}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    return fig

def plot_feature_ablation_analysis(model, X, y, feature_groups, model_name="CatBoost"):
    """Análise de ablação de features por grupos"""
    baseline_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    ablation_results = []
    for group_name, features in feature_groups.items():
        features_to_remove = [f for f in features if f in X.columns]
        if not features_to_remove:
            continue

        X_ablated = X.drop(columns=features_to_remove)

        # Re-treinar modelo simplificado
        if model_name == "CatBoost":
            ablated_model = CatBoostClassifier(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                random_seed=SEED,
                verbose=False
            )
            cat_features = [f for f in X_ablated.columns if
                            f in ['home_team', 'visitor_team', 'month', 'day_of_week', 'season_phase']]
            ablated_model.fit(X_ablated, y, cat_features=cat_features, verbose=False)
        else:
            ablated_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            ablated_model.fit(X_ablated, y)

        ablated_auc = roc_auc_score(y, ablated_model.predict_proba(X_ablated)[:, 1])
        performance_drop = baseline_auc - ablated_auc

        ablation_results.append({
            'feature_group': group_name,
            'features_removed': len(features_to_remove),
            'baseline_auc': baseline_auc,
            'ablated_auc': ablated_auc,
            'performance_drop': performance_drop
        })

    ablation_df = pd.DataFrame(ablation_results).sort_values('performance_drop', ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(ablation_df))

    ax.barh(y_pos, ablation_df['performance_drop'], color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ablation_df['feature_group'])
    ax.set_xlabel('Queda no AUC')
    ax.set_title(f'Análise de Ablação de Features - {model_name}\n(AUC Baseline: {baseline_auc:.4f})')

    for i, v in enumerate(ablation_df['performance_drop']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    return ablation_df

def compare_with_baseline_models(X_train, y_train, X_test, y_test, categorical_features):
    """Compara CatBoost com modelos baseline"""
    print("\n" + "=" * 60)
    print("COMPARAÇÃO COM MODELOS BASELINE")
    print("=" * 60)

    results = {}

    # 1. Regressão Logística
    print("📊 Treinando Regressão Logística...")
    lr = LogisticRegression(random_state=SEED, max_iter=1000)

    # Preparar dados para LR (one-hot encoding para categóricas)
    X_train_lr = X_train.copy()
    X_test_lr = X_test.copy()

    for col in categorical_features:
        if col in X_train_lr.columns:
            X_train_lr = pd.get_dummies(X_train_lr, columns=[col], prefix=col, drop_first=True)
            X_test_lr = pd.get_dummies(X_test_lr, columns=[col], prefix=col, drop_first=True)

    # Garantir que train e test tenham as mesmas colunas
    common_cols = X_train_lr.columns.intersection(X_test_lr.columns)
    X_train_lr = X_train_lr[common_cols]
    X_test_lr = X_test_lr[common_cols]

    # Scale features
    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr)
    X_test_lr_scaled = scaler.transform(X_test_lr)

    lr.fit(X_train_lr_scaled, y_train)
    y_pred_lr = lr.predict(X_test_lr_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_lr_scaled)[:, 1]

    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'auc_roc': roc_auc_score(y_test, y_pred_proba_lr),
        'log_loss': log_loss(y_test, y_pred_proba_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr)
    }

    # 2. Random Forest
    print("📊 Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)

    # Preparar dados para RF (label encoding para categóricas)
    X_train_rf = X_train.copy()
    X_test_rf = X_test.copy()

    for col in categorical_features:
        if col in X_train_rf.columns:
            X_train_rf[col] = X_train_rf[col].astype('category').cat.codes
            X_test_rf[col] = X_test_rf[col].astype('category').cat.codes

    rf.fit(X_train_rf, y_train)
    y_pred_rf = rf.predict(X_test_rf)
    y_pred_proba_rf = rf.predict_proba(X_test_rf)[:, 1]

    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'auc_roc': roc_auc_score(y_test, y_pred_proba_rf),
        'log_loss': log_loss(y_test, y_pred_proba_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf)
    }

    # 3. CatBoost (já treinado, vamos apenas calcular métricas)
    print("📊 Avaliando CatBoost...")
    # Isso será preenchido depois com o modelo principal

    # Comparação
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)

    print("\n📈 COMPARAÇÃO DE MODELOS:")
    print(comparison_df)

    # Plot comparativo
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['accuracy', 'auc_roc', 'log_loss', 'f1', 'precision', 'recall']
    metric_names = ['Acurácia', 'AUC-ROC', 'Log Loss', 'F1-Score', 'Precision', 'Recall']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        values = [results[model][metric] for model in results.keys()]

        if metric == 'log_loss':
            # Para log loss, menor é melhor
            bars = ax.bar(results.keys(), values, color=['red' if x == max(values) else 'blue' for x in values])
            ax.set_ylabel(name)
            ax.set_title(f'{name} (Menor é Melhor)')
        else:
            # Para outras métricas, maior é melhor
            bars = ax.bar(results.keys(), values, color=['green' if x == max(values) else 'lightblue' for x in values])
            ax.set_ylabel(name)
            ax.set_title(f'{name} (Maior é Melhor)')

        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.4f}', ha='center', va='bottom')

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return results, comparison_df

def plot_comprehensive_analysis(y_true, y_pred_proba, y_pred, model_name="CatBoost"):
    """Análise compreensiva do modelo"""
    print(f"\n📊 ANÁLISE COMPREENSIVA - {model_name}")
    print("=" * 50)

    # 1. Métricas detalhadas
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"📈 Métricas de Performance:")
    print(f"   Acurácia: {accuracy:.4f}")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   Log Loss: {logloss:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")

    # 2. Análise por threshold
    print(f"\n🎯 Análise por Threshold:")
    thresholds = [0.4, 0.5, 0.6]
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred_thresh)
        print(f"   Threshold {threshold}: Acurácia = {acc:.4f}")

    # 3. Plot ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curve
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution of predictions
    axes[0, 1].hist(y_pred_proba, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
    axes[0, 1].set_xlabel('Probabilidade Prevista')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição das Probabilidades')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    axes[1, 0].plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True, alpha=0.3)

    # Feature importance (placeholder - será preenchido depois)
    axes[1, 1].text(0.5, 0.5, 'Feature Importance\n(Será plotado separadamente)',
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy,
        'auc': auc_roc,
        'log_loss': logloss,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def advanced_adaptive_otw(team_data, current_date, team_name, df_players=None):
    BASE_WINDOW = 10
    MIN_WINDOW = 5
    MAX_WINDOW = 20

    if len(team_data) < MIN_WINDOW:
        return len(team_data)

    season_start = pd.to_datetime(f'{current_date.year}-10-01', utc=True)
    if current_date.month < 6:
        season_start = pd.to_datetime(f'{current_date.year - 1}-10-01', utc=True)

    days_into_season = (current_date - season_start).days

    if days_into_season < 30:
        window = max(MIN_WINDOW, min(len(team_data), 8))
    elif days_into_season > 150:
        window = min(MAX_WINDOW, 15)
    else:
        window = BASE_WINDOW

    if len(team_data) >= 10:
        recent_games = team_data.tail(10)
        if 'net_rating' in recent_games.columns:
            recent_std = recent_games['net_rating'].std()
            recent_mean = abs(recent_games['net_rating'].mean())
            if recent_mean > 0:
                cv = recent_std / recent_mean
                if cv > 1.5:
                    window = max(MIN_WINDOW, window - 3)
                elif cv < 0.5:
                    window = min(MAX_WINDOW, window + 2)

    if df_players is not None:
        roster_change_detected = detect_roster_change(
            df_players, team_name, current_date, lookback_days=14
        )
        if roster_change_detected:
            window = MIN_WINDOW

    games_last_week = team_data[
        team_data['date'] >= (current_date - timedelta(days=7))
        ].shape[0]

    if games_last_week >= 4:
        window = max(MIN_WINDOW, window - 2)
    elif games_last_week <= 1:
        window = min(MAX_WINDOW, window + 2)

    if len(team_data) >= window + 5:
        recent = team_data.tail(window)
        historical = team_data.tail(window * 2).head(window)
        if 'net_rating' in team_data.columns:
            recent_perf = recent['net_rating'].mean()
            hist_perf = historical['net_rating'].mean()
            if abs(recent_perf - hist_perf) > 10:
                window = max(MIN_WINDOW, window - 2)

    return int(window)

def detect_roster_change(df_players, team_name, current_date, lookback_days=14):
    if df_players is None or df_players.empty:
        return False

    team_players = df_players[
        (df_players['team_code'] == team_name) &
        (df_players['date'] >= (current_date - timedelta(days=lookback_days))) &
        (df_players['date'] < current_date)
        ]

    if team_players.empty:
        return False

    mid_date = current_date - timedelta(days=lookback_days / 2)
    first_half = team_players[team_players['date'] < mid_date]
    second_half = team_players[team_players['date'] >= mid_date]

    if first_half.empty or second_half.empty:
        return False

    top_first = set(
        first_half.groupby('player_id')['min']
        .mean()
        .nlargest(8)
        .index
    )

    top_second = set(
        second_half.groupby('player_id')['min']
        .mean()
        .nlargest(8)
        .index
    )

    overlap = len(top_first & top_second) / len(top_first | top_second)
    return overlap < 0.7

def calculate_dynamic_weights(dates, ref_date, volatility=None, game_importance=None):
    days_ago = (ref_date - dates).dt.days.clip(lower=0)
    if volatility is not None:
        lambda_param = 0.05 * (1 + volatility / 10)
    else:
        lambda_param = 0.05

    weights = np.exp(-lambda_param * days_ago)
    if game_importance is not None:
        weights *= (1 + 0.2 * np.array(game_importance))
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights

def calculate_confidence_interval(predictions, features, model_type='catboost'):
    base_pred = predictions
    uncertainty_factors = []

    if 'actual_window' in features:
        if features['actual_window'] < 8:
            uncertainty_factors.append(0.15)

    if 'home_score_volatility' in features and 'visitor_score_volatility' in features:
        avg_vol = (features['home_score_volatility'] + features['visitor_score_volatility']) / 2
        if avg_vol > 15:
            uncertainty_factors.append(0.10)

    if 'days_into_season' in features:
        if features['days_into_season'] < 30:
            uncertainty_factors.append(0.10)

    base_margin = 0.05
    total_margin = base_margin + sum(uncertainty_factors)
    lower_bound = max(0, base_pred - total_margin)
    upper_bound = min(1, base_pred + total_margin)
    confidence_score = 1 - total_margin
    return lower_bound, upper_bound, confidence_score

def safe_temporal_filter(df, current_date, buffer_days=1):
    safe_date = current_date - timedelta(days=buffer_days)
    return df[df['date'] < safe_date]

def wavg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or len(weights) == 0:
        return 0.0
    return float(np.average(values, weights=weights))

def wstd(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or len(weights) == 0:
        return 0.0
    m = np.average(values, weights=weights)
    var = np.average((values - m) ** 2, weights=weights)
    return float(np.sqrt(var))

def calculate_player_features(df_players, df_games, current_date, team,
                              window_days=30, min_games=5, buffer_hours=24):
    safe_date = current_date - timedelta(hours=buffer_hours)
    team_players_history = df_players[
        (df_players['team_code'] == team) &
        (df_players['date'] < safe_date)
        ]

    team_players_history = safe_temporal_filter(df_players, current_date)
    team_players_history = team_players_history[team_players_history['team_code'] == team]

    if len(team_players_history) == 0:
        return {}

    numeric_cols = ['min', 'points', 'totReb', 'assists', 'steals', 'blocks', 'turnovers', 'plusMinus', 'fgp', 'tpp', 'ftp']
    for col in numeric_cols:
        if col in team_players_history.columns:
            team_players_history[col] = pd.to_numeric(team_players_history[col], errors='coerce')

    team_players_history = team_players_history.dropna(subset=['min', 'points'], how='any')
    team_players_history[numeric_cols] = team_players_history[numeric_cols].fillna(0)

    date_cutoff = current_date - timedelta(days=window_days)
    recent_players = team_players_history[team_players_history['date'] >= date_cutoff]

    if len(recent_players) < min_games * 8:
        recent_players = team_players_history.tail(min_games * 10)

    if len(recent_players) == 0:
        return {}

    weights = calculate_dynamic_weights(recent_players['date'], current_date)
    player_stats = []
    for player_id in recent_players['player_id'].unique():
        player_data = recent_players[recent_players['player_id'] == player_id]
        if len(player_data) < 3:
            continue

        player_weights = calculate_dynamic_weights(player_data['date'], current_date)
        try:
            player_stat = {
                'min_avg': wavg(player_data['min'].astype(float), player_weights),
                'points_avg': wavg(player_data['points'].astype(float), player_weights),
                'rebounds_avg': wavg(player_data['totReb'].astype(float), player_weights),
                'assists_avg': wavg(player_data['assists'].astype(float), player_weights),
                'steals_avg': wavg(player_data['steals'].astype(float), player_weights),
                'blocks_avg': wavg(player_data['bvvou ks'].astype(float), player_weights),
                'turnovers_avg': wavg(player_data['turnovers'].astype(float), player_weights),
                'plus_minus_avg': wavg(player_data['plusMinus'].astype(float), player_weights),
                'fgp_avg': wavg(player_data['fgp'].astype(float), player_weights),
                'tpp_avg': wavg(player_data['tpp'].astype(float), player_weights),
                'ftp_avg': wavg(player_data['ftp'].astype(float), player_weights),
                'games_played': len(player_data)
            }
            player_stats.append(player_stat)
        except:
            continue

    if len(player_stats) == 0:
        return {}

    player_df = pd.DataFrame(player_stats)
    player_weights = np.array([p['min_avg'] * p['games_played'] for p in player_stats])
    if player_weights.sum() == 0:
        player_weights = np.ones(len(player_stats))

    features = {}
    for col in player_df.columns:
        if col != 'games_played':
            features[f'player_{col}'] = wavg(player_df[col], player_weights)

    top_players = player_df.nlargest(5, 'min_avg')
    for i, (_, player) in enumerate(top_players.iterrows(), 1):
        features[f'top{i}_points'] = player['points_avg']
        features[f'top{i}_rebounds'] = player['rebounds_avg']
        features[f'top{i}_assists'] = player['assists_avg']
        features[f'top{i}_plus_minus'] = player['plus_minus_avg']

    if len(player_df) > 1 and player_df['points_avg'].mean() > 0:
        features['player_consistency'] = 1.0 - (player_df['points_avg'].std() / player_df['points_avg'].mean())
    else:
        features['player_consistency'] = 0.5

    features['player_depth'] = len([p for p in player_stats if p['points_avg'] > 8.0])
    features['player_offensive_rating'] = (
            features['player_points_avg'] * features['player_fgp_avg'] / 100.0 +
            features['player_assists_avg'] * 0.5
    )
    features['player_defensive_rating'] = (
            features['player_steals_avg'] * 2.0 +
            features['player_blocks_avg'] * 2.0 -
            features['player_turnovers_avg'] * 0.5
    )
    return features

def calculate_advanced_features_with_players(df_games, df_players, window=10, use_adaptive_otw=True,
                                             include_player_features=True):
    features_list = []
    print(
        f"\n🔄 Calculando features avançadas (window={window}, adaptive={use_adaptive_otw}, players={include_player_features})...")
    total_games = len(df_games)
    df_games_sorted = df_games.sort_values('date').reset_index(drop=True)

    for idx, game in df_games_sorted.iterrows():
        if idx % 100 == 0:
            print(f"  Processando jogo {idx}/{total_games} ({idx / total_games * 100:.1f}%)")

        game_date = game['date']
        home_team = game['home_team']
        visitor_team = game['visitor_team']
        past_games = safe_temporal_filter(df_games_sorted, game_date, buffer_days=1)
        home_prev_games = past_games[
            ((past_games['home_team'] == home_team) |
             (past_games['visitor_team'] == home_team))
        ].sort_values('date')
        visitor_prev_games = past_games[
            ((past_games['home_team'] == visitor_team) |
             (past_games['visitor_team'] == visitor_team))
        ].sort_values('date')

        min_games_required = 5
        if len(home_prev_games) < min_games_required or len(visitor_prev_games) < min_games_required:
            continue

        home_net_ratings = []
        for _, g in home_prev_games.iterrows():
            if g['home_team'] == home_team:
                net_rating = g['home_score'] - g['visitor_score']
            else:
                net_rating = g['visitor_score'] - g['home_score']
            home_net_ratings.append(net_rating)

        visitor_net_ratings = []
        for _, g in visitor_prev_games.iterrows():
            if g['home_team'] == visitor_team:
                net_rating = g['home_score'] - g['visitor_score']
            else:
                net_rating = g['visitor_score'] - g['home_score']
            visitor_net_ratings.append(net_rating)

        home_prev_games = home_prev_games.copy()
        visitor_prev_games = visitor_prev_games.copy()
        home_prev_games['net_rating'] = home_net_ratings
        visitor_prev_games['net_rating'] = visitor_net_ratings

        if use_adaptive_otw:
            home_window = advanced_adaptive_otw(home_prev_games, game_date, home_team, df_players)
            visitor_window = advanced_adaptive_otw(visitor_prev_games, game_date, visitor_team, df_players)
            actual_window = min(home_window, visitor_window)
        else:
            actual_window = window

        home_recent = home_prev_games.tail(actual_window).copy()
        visitor_recent = visitor_prev_games.tail(actual_window).copy()
        home_vol = home_recent['net_rating'].std() if 'net_rating' in home_recent and not home_recent.empty else None
        visitor_vol = visitor_recent[
            'net_rating'].std() if 'net_rating' in visitor_recent and not visitor_recent.empty else None

        w_home = calculate_dynamic_weights(home_recent['date'], game_date, volatility=home_vol)
        w_vis = calculate_dynamic_weights(visitor_recent['date'], game_date, volatility=visitor_vol)
        features = {
            'game_id': game.get('game_id', f'idx_{idx}'),
            'date': game_date,
            'home_team': home_team,
            'visitor_team': visitor_team,
            'home_win': game['home_win'],
            'actual_window': actual_window
        }

        def acc_points(team_recent, team_name):
            wins = 0
            pts_scored, pts_allowed = [], []
            for _, g in team_recent.iterrows():
                if g['home_team'] == team_name:
                    won = g['home_score'] > g['visitor_score']
                    wins += int(won)
                    pts_scored.append(g['home_score'])
                    pts_allowed.append(g['visitor_score'])
                else:
                    won = g['visitor_score'] > g['home_score']
                    wins += int(won)
                    pts_scored.append(g['visitor_score'])
                    pts_allowed.append(g['home_score'])
            return wins, pts_scored, pts_allowed

        hw, hps, hpa = acc_points(home_recent, home_team)
        vw, vps, vpa = acc_points(visitor_recent, visitor_team)
        features['home_win_rate'] = hw / len(home_recent) if len(home_recent) > 0 else 0.5
        features['visitor_win_rate'] = vw / len(visitor_recent) if len(visitor_recent) > 0 else 0.5
        hps_vals = np.array(hps, dtype=float)
        hpa_vals = np.array(hpa, dtype=float)
        vps_vals = np.array(vps, dtype=float)
        vpa_vals = np.array(vpa, dtype=float)
        features['home_avg_points'] = wavg(hps_vals, w_home) if len(hps_vals) else 0.0
        features['visitor_avg_points'] = wavg(vps_vals, w_vis) if len(vps_vals) else 0.0
        features['home_avg_points_allowed'] = wavg(hpa_vals, w_home) if len(hpa_vals) else 0.0
        features['visitor_avg_points_allowed'] = wavg(vpa_vals, w_vis) if len(vpa_vals) else 0.0
        features['home_net_rating'] = features['home_avg_points'] - features['home_avg_points_allowed']
        features['visitor_net_rating'] = features['visitor_avg_points'] - features['visitor_avg_points_allowed']

        if len(home_prev_games) > 0:
            last_home_game_date = home_prev_games.iloc[-1]['date']
            days_rest_home = int((game_date - last_home_game_date).days)
            features['home_rest_days'] = days_rest_home
            features['home_back_to_back'] = 1 if days_rest_home <= 1 else 0
        else:
            features['home_rest_days'] = 7
            features['home_back_to_back'] = 0

        if len(visitor_prev_games) > 0:
            last_visitor_game_date = visitor_prev_games.iloc[-1]['date']
            days_rest_visitor = int((game_date - last_visitor_game_date).days)
            features['visitor_rest_days'] = days_rest_visitor
            features['visitor_back_to_back'] = 1 if days_rest_visitor <= 1 else 0
        else:
            features['visitor_rest_days'] = 7
            features['visitor_back_to_back'] = 0

        week_ago = game_date - timedelta(days=7)
        features['home_games_last_week'] = int((home_prev_games['date'] >= week_ago).sum())
        features['visitor_games_last_week'] = int((visitor_prev_games['date'] >= week_ago).sum())

        def calculate_streak(team_games, team_name, max_games=10):
            streak = 0
            for _, g in team_games.tail(max_games).iloc[::-1].iterrows():
                if g['home_team'] == team_name:
                    won = g['home_score'] > g['visitor_score']
                else:
                    won = g['visitor_score'] > g['home_score']
                if streak == 0:
                    streak = 1 if won else -1
                elif streak > 0 and won:
                    streak += 1
                elif streak < 0 and not won:
                    streak -= 1
                else:
                    break
            return streak

        features['home_streak'] = calculate_streak(home_prev_games, home_team)
        features['visitor_streak'] = calculate_streak(visitor_prev_games, visitor_team)
        h2h_cutoff = game_date - timedelta(days=7)
        h2h_games = past_games[
            (past_games['date'] < h2h_cutoff) &
            (((past_games['home_team'] == home_team) & (past_games['visitor_team'] == visitor_team)) |
             ((past_games['home_team'] == visitor_team) & (past_games['visitor_team'] == home_team)))
            ].tail(5)

        if len(h2h_games) > 0:
            h2h_home_wins = 0
            for _, g in h2h_games.iterrows():
                if g['home_team'] == home_team:
                    h2h_home_wins += int(g['home_score'] > g['visitor_score'])
                else:
                    h2h_home_wins += int(g['visitor_score'] > g['home_score'])
            features['h2h_home_win_rate'] = h2h_home_wins / len(h2h_games)
        else:
            features['h2h_home_win_rate'] = 0.5

        home_at_home = past_games[past_games['home_team'] == home_team].tail(15)
        if len(home_at_home) > 0:
            features['home_team_home_win_rate'] = (home_at_home['home_score'] > home_at_home['visitor_score']).mean()
            features['home_team_home_avg_score'] = home_at_home['home_score'].mean()
            features['home_team_home_avg_allowed'] = home_at_home['visitor_score'].mean()
        else:
            features['home_team_home_win_rate'] = 0.5
            features['home_team_home_avg_score'] = 110.0
            features['home_team_home_avg_allowed'] = 110.0

        visitor_away = past_games[past_games['visitor_team'] == visitor_team].tail(15)
        if len(visitor_away) > 0:
            features['visitor_team_away_win_rate'] = (visitor_away['visitor_score'] > visitor_away['home_score']).mean()
            features['visitor_team_away_avg_score'] = visitor_away['visitor_score'].mean()
            features['visitor_team_away_avg_allowed'] = visitor_away['home_score'].mean()
        else:
            features['visitor_team_away_win_rate'] = 0.5
            features['visitor_team_away_avg_score'] = 110.0
            features['visitor_team_away_avg_allowed'] = 110.0

        def calculate_momentum(team_games, team_name, games=3):
            last_games = team_games.tail(games)
            if len(last_games) == 0:
                return 0.0
            momentum = 0
            for _, g in last_games.iterrows():
                if g['home_team'] == team_name:
                    momentum += 1 if g['home_score'] > g['visitor_score'] else -1
                else:
                    momentum += 1 if g['visitor_score'] > g['home_score'] else -1
            return momentum / len(last_games)

        features['home_momentum'] = calculate_momentum(home_prev_games, home_team)
        features['visitor_momentum'] = calculate_momentum(visitor_prev_games, visitor_team)
        features['month'] = int(game_date.month)
        features['day_of_week'] = int(game_date.dayofweek)
        features['is_weekend'] = 1 if game_date.dayofweek >= 5 else 0
        season_start = pd.to_datetime(f'{game_date.year}-10-01', utc=True)
        if game_date.month < 6:
            season_start = pd.to_datetime(f'{game_date.year - 1}-10-01', utc=True)
        days_into_season = int((game_date - season_start).days)
        features['days_into_season'] = days_into_season
        features['season_phase'] = int(min(3, days_into_season // 60))
        features['win_rate_diff'] = features['home_win_rate'] - features['visitor_win_rate']
        features['net_rating_diff'] = features['home_net_rating'] - features['visitor_net_rating']
        features['points_diff'] = features['home_avg_points'] - features['visitor_avg_points']
        features['rest_advantage'] = features['home_rest_days'] - features['visitor_rest_days']
        features['momentum_diff'] = features['home_momentum'] - features['visitor_momentum']
        features['streak_diff'] = features['home_streak'] - features['visitor_streak']

        if len(home_recent) >= 3:
            home_pts_series = np.array([
                g['home_score'] if g['home_team'] == home_team else g['visitor_score']
                for _, g in home_recent.iterrows()
            ], dtype=float)
            features['home_score_volatility'] = wstd(home_pts_series, w_home)
        else:
            features['home_score_volatility'] = 10.0

        if len(visitor_recent) >= 3:
            vis_pts_series = np.array([
                g['home_score'] if g['home_team'] == visitor_team else g['visitor_score']
                for _, g in visitor_recent.iterrows()
            ], dtype=float)
            features['visitor_score_volatility'] = wstd(vis_pts_series, w_vis)
        else:
            features['visitor_score_volatility'] = 10.0

        features['home_off_vs_visitor_def'] = features['home_avg_points'] - features['visitor_avg_points_allowed']
        features['visitor_off_vs_home_def'] = features['visitor_avg_points'] - features['home_avg_points_allowed']

        if include_player_features:
            try:
                home_player_features = calculate_player_features(df_players, df_games, game_date, home_team)
                visitor_player_features = calculate_player_features(df_players, df_games, game_date, visitor_team)
                for key, value in home_player_features.items():
                    features[f'home_{key}'] = value
                for key, value in visitor_player_features.items():
                    features[f'visitor_{key}'] = value
                for key in home_player_features.keys():
                    home_key = f'home_{key}'
                    visitor_key = f'visitor_{key}'
                    if home_key in features and visitor_key in features:
                        features[f'player_{key}_diff'] = features[home_key] - features[visitor_key]
            except:
                pass

        features_list.append(features)

    df_feat = pd.DataFrame(features_list)
    return df_feat.replace([np.inf, -np.inf], np.nan).dropna()


class TemporalValidator:
    def __init__(self, n_splits=3, embargo_days=1, min_train=50, min_test=10):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.min_train = min_train
        self.min_test = min_test

    def split(self, df):
        if 'date' not in df.columns or len(df) == 0:
            return []

        # Criar uma cópia com índice resetado para trabalhar
        df_sorted = df.sort_values('date').reset_index(drop=True)
        date_norm = df_sorted['date'].dt.normalize()
        unique_days = date_norm.unique()

        if len(unique_days) < self.n_splits + 2:
            return []

        fold_len = max(1, len(unique_days) // (self.n_splits + 1))
        splits = []

        for i in range(self.n_splits):
            test_start_idx = len(unique_days) - (i + 1) * fold_len
            test_end_idx = len(unique_days) - 1 - i * fold_len

            if test_start_idx < 0 or test_end_idx < 0:
                continue

            test_start_day = pd.to_datetime(unique_days[test_start_idx])
            test_end_day = pd.to_datetime(unique_days[test_end_idx])

            embargo_cut = test_start_day - pd.Timedelta(days=self.embargo_days)

            # Usar o índice do DataFrame resetado
            train_idx = df_sorted[df_sorted['date'] < embargo_cut].index
            test_idx = df_sorted[
                (df_sorted['date'] >= test_start_day) &
                (df_sorted['date'] <= test_end_day)
                ].index

            if len(train_idx) >= self.min_train and len(test_idx) >= self.min_test:
                splits.append((train_idx.tolist(), test_idx.tolist()))

        return splits

def calcular_acuracia_por_time(test_df, y_test, y_pred):
    """Calcula acurácia por time"""
    resultados_time = []
    test_df_com_pred = test_df.copy()
    test_df_com_pred['prediction'] = y_pred
    test_df_com_pred['actual'] = y_test
    test_df_com_pred['correct'] = (test_df_com_pred['prediction'] == test_df_com_pred['actual']).astype(int)
    home_teams = test_df_com_pred['home_team'].unique()
    visitor_teams = test_df_com_pred['visitor_team'].unique()
    all_teams = set(home_teams) | set(visitor_teams)

    for team in sorted(all_teams):
        home_games = test_df_com_pred[test_df_com_pred['home_team'] == team]
        home_correct = home_games['correct'].sum() if len(home_games) > 0 else 0
        home_total = len(home_games)
        home_accuracy = home_correct / home_total if home_total > 0 else 0
        visitor_games = test_df_com_pred[test_df_com_pred['visitor_team'] == team]
        visitor_correct = visitor_games['correct'].sum() if len(visitor_games) > 0 else 0
        visitor_total = len(visitor_games)
        visitor_accuracy = visitor_correct / visitor_total if visitor_total > 0 else 0
        team_games = test_df_com_pred[
            (test_df_com_pred['home_team'] == team) |
            (test_df_com_pred['visitor_team'] == team)
            ]
        team_correct = team_games['correct'].sum()
        team_total = len(team_games)
        team_accuracy = team_correct / team_total if team_total > 0 else 0
        resultados_time.append({
            'time': team,
            'jogos_como_home': home_total,
            'acuracia_home': home_accuracy,
            'jogos_como_visitor': visitor_total,
            'acuracia_visitor': visitor_accuracy,
            'total_jogos': team_total,
            'total_acertos': team_correct,
            'acuracia_geral_time': team_accuracy
        })

    return pd.DataFrame(resultados_time)

def avaliar_predictions(y_true, proba, thresh=0.5, titulo="Sem calibração"):
    """Avalia predictions com threshold específico"""
    preds = (proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    print(f"\n🔎 Avaliação {titulo} (threshold={thresh:.2f})")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, preds, digits=4))

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS
# =============================================================================

# Carregar dados
games_file = 'nba_all_games_2021_present_2025-09-26_17-45-08.xlsx'
players_file = 'nba_all_player_stats_2021_present_2025-09-26_17-45-08.xlsx'

print("\n📊 Carregando dados...")
df_games = pd.read_excel(games_file)
df_players = pd.read_excel(players_file)

# Pré-processamento
numeric_columns = ['min', 'points', 'totReb', 'assists', 'steals', 'blocks', 'turnovers', 'plusMinus', 'fgp', 'tpp', 'ftp']

for col in numeric_columns:
    if col in df_players.columns:
        df_players[col] = pd.to_numeric(df_players[col], errors='coerce')

df_players = df_players.dropna(subset=['min', 'points'], how='any')
df_players[numeric_columns] = df_players[numeric_columns].fillna(0)

df_games['date'] = pd.to_datetime(df_games['date'], utc=True, errors='coerce')
df_players['date'] = pd.to_datetime(df_players['date'], utc=True, errors='coerce')
df_games = df_games.dropna(subset=['date'])
df_players = df_players.dropna(subset=['date'])

regular_end_date = pd.to_datetime('2025-04-15', utc=True)
df_games = df_games[df_games['date'] <= regular_end_date].copy()
df_players = df_players[df_players['date'] <= regular_end_date].copy()

print(f"✅ Dados carregados: {len(df_games)} jogos, {len(df_players)} stats de jogadores")
print(f"✅ Dados dos jogadores limpos: {len(df_players)} registros válidos")

df_games = df_games.rename(columns={
    'away_score': 'visitor_score',
    'home_team_code': 'home_team',
    'away_team_code': 'visitor_team'
})

df_games['home_win'] = (df_games['home_score'] > df_games['visitor_score']).astype(int)
print(f"✅ Target criado. Distribuição: {df_games['home_win'].value_counts(normalize=True).to_dict()}")

# =============================================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS
# =============================================================================

print("\n🔍 Otimizando Hiperparâmetros com Validação Temporal...")

configs = [
    {'window': 8, 'adaptive': False, 'players': True},
    {'window': 10, 'adaptive': False, 'players': True},
    {'window': 12, 'adaptive': False, 'players': True},
    {'window': 15, 'adaptive': False, 'players': True},
    {'window': 10, 'adaptive': True, 'players': True},
    {'window': 12, 'adaptive': True, 'players': True}
]

best_config = None
best_auc = 0.0
config_results = {}

for config in configs:
    print(f"\nTestando configuração: {config}")
    df_features = calculate_advanced_features_with_players(
        df_games, df_players,
        window=config['window'],
        use_adaptive_otw=config['adaptive'],
        include_player_features=config['players']
    )

    if len(df_features) < 100:
        print("  ❌ Poucos dados, pulando...")
        continue

    numerical_features = [
        'home_win_rate', 'visitor_win_rate',
        'home_avg_points', 'visitor_avg_points',
        'home_avg_points_allowed', 'visitor_avg_points_allowed',
        'home_net_rating', 'visitor_net_rating',
        'home_rest_days', 'visitor_rest_days',
        'home_back_to_back', 'visitor_back_to_back',
        'home_games_last_week', 'visitor_games_last_week',
        'home_streak', 'visitor_streak', 'h2h_home_win_rate',
        'home_team_home_win_rate', 'visitor_team_away_win_rate',
        'home_team_home_avg_score', 'home_team_home_avg_allowed',
        'visitor_team_away_avg_score', 'visitor_team_away_avg_allowed',
        'home_momentum', 'visitor_momentum', 'days_into_season',
        'win_rate_diff', 'net_rating_diff', 'points_diff', 'rest_advantage',
        'momentum_diff', 'streak_diff', 'home_score_volatility', 'visitor_score_volatility',
        'home_off_vs_visitor_def', 'visitor_off_vs_home_def'
    ]

    if config['players']:
        player_features = [col for col in df_features.columns if col.startswith('home_player_') or
                           col.startswith('visitor_player_') or col.startswith('player_')]
        numerical_features.extend(player_features)

    categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend', 'season_phase']
    all_features = numerical_features + categorical_features
    target = 'home_win'
    available_features = [f for f in all_features if f in df_features.columns]
    missing_features = set(all_features) - set(available_features)
    if missing_features:
        print(f"  ⚠️ Features faltantes: {missing_features}")

    validator = TemporalValidator(n_splits=3, embargo_days=1)
    splits = validator.split(df_features)

    if len(splits) == 0:
        print("  ❌ Nenhum split válido")
        continue

    fold_aucs = []
    for train_idx, val_idx in splits:
        train_df = df_features.iloc[train_idx]
        val_df = df_features.iloc[val_idx]
        X_train = train_df[available_features]
        y_train = train_df[target]
        X_val = val_df[available_features]
        y_val = val_df[target]
        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=10,
            cat_features=[f for f in categorical_features if f in available_features],
            random_seed=42,
            verbose=False,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=20,
            subsample=0.8
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        fold_aucs.append(auc_score)

    mean_auc = float(np.mean(fold_aucs))
    config_results[str(config)] = {'auc': mean_auc, 'std': float(np.std(fold_aucs))}
    print(f"  AUC médio: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_config = config

if best_config is None:
    print("⚠️ Nenhuma configuração válida. Usando fallback {'window': 10, 'adaptive': True, 'players': True}.")
    best_config = {'window': 10, 'adaptive': True, 'players': True}

print(f"\n✅ Melhor configuração: {best_config} (AUC: {best_auc:.4f})")

# =============================================================================
# TREINAMENTO DO MODELO FINAL
# =============================================================================

print("\n🤖 Treinamento final com configuração otimizada...")

df_features_final = calculate_advanced_features_with_players(
    df_games, df_players,
    window=best_config['window'],
    use_adaptive_otw=best_config['adaptive'],
    include_player_features=True
)

print(f"\n🔍 Dataset final: {len(df_features_final)} jogos")
print(f"Período: {df_features_final['date'].min().date()} até {df_features_final['date'].max().date()})")
print(f"Distribuição target: {df_features_final['home_win'].value_counts(normalize=True).to_dict()}")

numerical_features = [
    'home_win_rate', 'visitor_win_rate',
    'home_avg_points', 'visitor_avg_points',
    'home_avg_points_allowed', 'visitor_avg_points_allowed',
    'home_net_rating', 'visitor_net_rating',
    'home_rest_days', 'visitor_rest_days',
    'home_back_to_back', 'visitor_back_to_back',
    'home_games_last_week', 'visitor_games_last_week',
    'home_streak', 'visitor_streak', 'h2h_home_win_rate',
    'home_team_home_win_rate', 'visitor_team_away_win_rate',
    'home_team_home_avg_score', 'home_team_home_avg_allowed',
    'visitor_team_away_avg_score', 'visitor_team_away_avg_allowed',
    'home_momentum', 'visitor_momentum', 'days_into_season',
    'win_rate_diff', 'net_rating_diff', 'points_diff', 'rest_advantage',
    'momentum_diff', 'streak_diff', 'home_score_volatility', 'visitor_score_volatility',
    'home_off_vs_visitor_def', 'visitor_off_vs_home_def'
]

player_features = [col for col in df_features_final.columns if col.startswith('home_player_') or
                   col.startswith('visitor_player_') or col.startswith('player_')]
numerical_features.extend(player_features)
categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend', 'season_phase']
all_features = numerical_features + categorical_features
target = 'home_win'
available_features = [f for f in all_features if f in df_features_final.columns]
split_date = df_features_final['date'].quantile(0.80)
train_mask = df_features_final['date'] < split_date
test_mask = ~train_mask

if test_mask.sum() < 100 and len(df_features_final) >= 300:
    split_date = df_features_final['date'].quantile(0.85)
    train_mask = df_features_final['date'] < split_date
    test_mask = ~train_mask

train_df = df_features_final[train_mask]
test_df = df_features_final[test_mask]

print(f"\n📅 Split temporal:")
print(f"Treino: {len(train_df)} jogos ({train_df['date'].min().date()} a {train_df['date'].max().date()})")
print(f"Teste: {len(test_df)} jogos ({test_df['date'].min().date()} a {test_df['date'].max().date()})")

X_train = train_df[available_features]
y_train = train_df[target]
X_test = test_df[available_features]
y_test = test_df[target]

pos_rate = float(y_train.mean()) if len(y_train) > 0 else 0.5
scale_pos_weight = float((1 - pos_rate) / pos_rate) if 0 < pos_rate < 1 else 1.0

final_model = CatBoostClassifier(
    iterations=400,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=8,
    cat_features=[f for f in categorical_features if f in available_features],
    random_seed=42,
    verbose=False,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=30,
    subsample=0.85,
    rsm=0.8,
    scale_pos_weight=scale_pos_weight
)

print("Treinando modelo final...")
final_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=50
)

y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# =============================================================================
# ANÁLISE DE CONFIANÇA
# =============================================================================

cis = []
for i in range(len(test_df)):
    pred = y_pred_proba[i]
    feat = test_df.iloc[i].to_dict()
    lower, upper, conf = calculate_confidence_interval(pred, feat)
    cis.append((lower, upper, conf))

avg_conf = np.mean([c[2] for c in cis])
print(f"\nConfiança média das previsões: {avg_conf:.4f}")
print("Exemplos de CI (primeiros 5 jogos):")
for i in range(min(5, len(cis))):
    lower, upper, conf = cis[i]
    print(f"Jogo {i}: Prob {y_pred_proba[i]:.4f} | CI [{lower:.4f}, {upper:.4f}] | Conf {conf:.4f}")

# =============================================================================
# AVALIAÇÃO DO MODELO
# =============================================================================

acc = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)

print("\n📈 RESULTADOS FINAIS:")
print("=" * 60)
print(f"Acurácia: {acc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Log Loss: {ll:.4f}")

p = y_test.mean()
baseline_acc = max(p, 1 - p) if 0 < p < 1 else 1.0
print(f"Baseline (classe majoritária): {baseline_acc:.4f}")
print(f"Melhoria sobre baseline: +{(acc - baseline_acc):.4f}")

home_preds_pct = float((y_pred_proba > 0.5).mean())
print(f"\nViés do modelo: {home_preds_pct:.1%} previsões para time da casa")

# Calibração
calibrated_model = None
y_pred_proba_cal = y_pred_proba

if home_preds_pct > 0.65 or home_preds_pct < 0.45:
    print("⚠️ Aplicando calibração devido a viés detectado...")
    calib_cut = train_df['date'].quantile(0.85)
    X_calib = train_df[train_df['date'] >= calib_cut][available_features]
    y_calib = train_df[train_df['date'] >= calib_cut][target]
    if len(X_calib) < 100:
        X_calib = X_train
        y_calib = y_train

    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_calib, y_calib)
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_pred_proba_cal >= 0.5).astype(int)
    acc_cal = accuracy_score(y_test, y_pred_cal)
    auc_roc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    ll_cal = log_loss(y_test, y_pred_proba_cal)

    print("\n📈 Resultados após calibração isotônica:")
    print("=" * 60)
    print(f"Acurácia calibrada: {acc_cal:.4f}")
    print(f"AUC-ROC calibrada: {auc_roc_cal:.4f}")
    print(f"Log Loss calibrado: {ll_cal:.4f}")

    try:
        prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='quantile')
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_pred_proba_cal, n_bins=10, strategy='quantile')
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfeita')
        plt.plot(prob_pred_raw, prob_true_raw, marker='o', label='Sem calibração')
        plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='Calibrada')
        plt.xlabel('Probabilidade prevista')
        plt.ylabel('Fração positiva')
        plt.title('Curva de Confiabilidade')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except:
        pass
else:
    print("Sem calibração adicional.")

# Avaliações detalhadas
avaliar_predictions(y_test, y_pred_proba, 0.5, "Bruta")
if calibrated_model is not None:
    avaliar_predictions(y_test, y_pred_proba_cal, 0.5, "Calibrada")

# =============================================================================
# ANÁLISE DE FEATURES
# =============================================================================

fi_df = None
try:
    importances = final_model.get_feature_importance()
    feat_names = list(X_train.columns)
    fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
    print("\n🏅 Top 20 importâncias de features:")
    print(fi_df.head(20).to_string(index=False))
    plt.figure(figsize=(8, 8))
    top_n = 20
    plt.barh(fi_df.head(top_n)['feature'][::-1], fi_df.head(top_n)['importance'][::-1])
    plt.title('Importância de Features CatBoost')
    plt.tight_layout()
    plt.show()
except:
    pass

# SHAP analysis
try:
    n_sample = min(500, len(X_test))
    if n_sample >= 50:
        X_shap = X_test.sample(n=n_sample, random_state=42)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_shap)
        print("\n📊 Plot SHAP summary das principais variáveis")
        shap.summary_plot(shap_values, X_shap, show=True, plot_type='bar', max_display=20)
        shap.summary_plot(shap_values, X_shap, show=True, max_display=20)
except:
    pass

# =============================================================================
# ANÁLISE POR TIME
# =============================================================================

print("\n📊 Calculando acurácia por time...")
df_acuracia_time = calcular_acuracia_por_time(test_df, y_test, y_pred)
df_acuracia_time = df_acuracia_time.sort_values('acuracia_geral_time', ascending=False)

print("\n🏆 TOP 10 Times por Acurácia:")
print(df_acuracia_time.head(10).to_string(index=False))
print("\n📉 BOTTOM 10 Times por Acurácia:")
print(df_acuracia_time.tail(10).to_string(index=False))

# =============================================================================
# ANÁLISES COMPLEMENTARES
# =============================================================================

print("\n" + "=" * 80)
print("ANÁLISES COMPLEMENTARES E COMPARAÇÃO DE MODELOS")
print("=" * 80)

# 1. Comparação com modelos baseline
print("\n1. 🔄 COMPARANDO COM MODELOS BASELINE...")
categorical_features_clean = [f for f in categorical_features if f in available_features]
baseline_results, comparison_df = compare_with_baseline_models(
    X_train, y_train, X_test, y_test, categorical_features_clean
)

# Adicionar CatBoost aos resultados baseline
y_pred_proba_catboost = final_model.predict_proba(X_test)[:, 1]
y_pred_catboost = final_model.predict(X_test)

baseline_results['CatBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_catboost),
    'auc_roc': roc_auc_score(y_test, y_pred_proba_catboost),
    'log_loss': log_loss(y_test, y_pred_proba_catboost),
    'f1': f1_score(y_test, y_pred_catboost),
    'precision': precision_score(y_test, y_pred_catboost),
    'recall': recall_score(y_test, y_pred_catboost)
}

# 2. Análise compreensiva do CatBoost
print("\n2. 📊 ANÁLISE COMPREENSIVA DO CATBOOST...")
catboost_metrics = plot_comprehensive_analysis(
    y_test, y_pred_proba_catboost, y_pred_catboost, "CatBoost"
)

# 3. Análise de calibração
print("\n3. 🎯 ANÁLISE DE CALIBRAÇÃO...")
plot_calibration_analysis(y_test, y_pred_proba_catboost, None, "CatBoost")

# 4. Matriz de confusão
print("\n4. 📋 MATRIZ DE CONFUSÃO...")
plot_confusion_matrix_analysis(y_test, y_pred_catboost, None, "CatBoost")

# 5. Análise de ablação de features
print("\n5. 🔍 ANÁLISE DE ABLAÇÃO DE FEATURES...")
feature_groups = {
    'Estatísticas Básicas': ['home_win_rate', 'visitor_win_rate', 'home_avg_points', 'visitor_avg_points'],
    'Estatísticas Defensivas': ['home_avg_points_allowed', 'visitor_avg_points_allowed', 'home_net_rating',
                                'visitor_net_rating'],
    'Descanso e Agenda': ['home_rest_days', 'visitor_rest_days', 'home_back_to_back', 'visitor_back_to_back'],
    'Momentum': ['home_streak', 'visitor_streak', 'home_momentum', 'visitor_momentum'],
    'Head-to-Head': ['h2h_home_win_rate'],
    'Home/Away Performance': ['home_team_home_win_rate', 'visitor_team_away_win_rate'],
    'Features de Jogadores': [f for f in available_features if 'player_' in f],
    'Diferenciais': ['win_rate_diff', 'net_rating_diff', 'points_diff', 'rest_advantage']
}

# Filtrar apenas features que existem
filtered_groups = {}
for group_name, features in feature_groups.items():
    existing_features = [f for f in features if f in available_features]
    if existing_features:
        filtered_groups[group_name] = existing_features

ablation_results = plot_feature_ablation_analysis(final_model, X_train, y_train, filtered_groups, "CatBoost")

# 6. Validação Purged K-Fold
print("\n6. 🔄 VALIDAÇÃO PURGED K-FOLD...")

# Criar uma cópia do DataFrame para trabalhar com índices consistentes
df_features_final_sorted = df_features_final.sort_values('date').reset_index(drop=True)
purged_folds = purged_kfold_validation(df_features_final_sorted, n_splits=5, embargo_days=3)

if purged_folds:
    purged_aucs = []
    for fold, (train_idx, val_idx) in enumerate(purged_folds):
        print(f"  Fold {fold + 1}/{len(purged_folds)}...")

        # Usar .iloc para acessar pelas posições, já que resetamos o índice
        X_train_fold = df_features_final_sorted.iloc[train_idx][available_features]
        y_train_fold = df_features_final_sorted.iloc[train_idx][target]
        X_val_fold = df_features_final_sorted.iloc[val_idx][available_features]
        y_val_fold = df_features_final_sorted.iloc[val_idx][target]

        # Pular fold se não houver dados suficientes
        if len(X_train_fold) < 50 or len(X_val_fold) < 10:
            print(f"    ⚠️ Fold {fold + 1} ignorado: dados insuficientes")
            continue

        fold_model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            cat_features=[f for f in categorical_features_clean if f in available_features],
            random_seed=SEED,
            verbose=False
        )

        fold_model.fit(X_train_fold, y_train_fold, verbose=False)
        y_pred_fold = fold_model.predict_proba(X_val_fold)[:, 1]
        fold_auc = roc_auc_score(y_val_fold, y_pred_fold)
        purged_aucs.append(fold_auc)
        print(f"    Fold {fold + 1} AUC: {fold_auc:.4f}")

    if purged_aucs:
        print(f"✅ AUC médio Purged K-Fold: {np.mean(purged_aucs):.4f} ± {np.std(purged_aucs):.4f}")
    else:
        print("❌ Nenhum fold válido para cálculo do AUC")
else:
    print("❌ Não foi possível realizar Purged K-Fold (poucos dados ou splits inválidos)")
# =============================================================================
# SALVAMENTO DE RESULTADOS
# =============================================================================

acuracia_geral = accuracy_score(y_test, y_pred)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    excel_path = f"resultados_completos_nba_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Dataset completo com features
        df_features_final.to_excel(writer, sheet_name='dataset_features', index=False)

        # 2. Acurácia por time
        df_acuracia_time.to_excel(writer, sheet_name='acuracia_por_time', index=False)

        # 3. Métricas gerais
        metricas_gerais_data = {
            'Metrica': [
                'Acurácia Geral', 'AUC-ROC', 'Log Loss', 'F1-Score',
                'Precision', 'Recall', 'Baseline (classe majoritária)',
                'Melhoria sobre baseline', 'Total de Jogos no Teste',
                'Configuração Otimizada', 'AUC Purged K-Fold'
            ],
            'Valor': [
                f"{catboost_metrics['accuracy']:.4f}",
                f"{catboost_metrics['auc']:.4f}",
                f"{catboost_metrics['log_loss']:.4f}",
                f"{catboost_metrics['f1']:.4f}",
                f"{catboost_metrics['precision']:.4f}",
                f"{catboost_metrics['recall']:.4f}",
                f"{baseline_acc:.4f}",
                f"{(catboost_metrics['accuracy'] - baseline_acc):.4f}",
                f"{len(test_df)}",
                str(best_config),
                f"{np.mean(purged_aucs) if purged_folds else 'N/A':.4f}"
            ]
        }
        pd.DataFrame(metricas_gerais_data).to_excel(writer, sheet_name='metricas_gerais', index=False)

        # 4. Feature importances
        if fi_df is not None:
            fi_df.to_excel(writer, sheet_name='feature_importances', index=False)

        # 5. Comparação de modelos
        comparison_df.to_excel(writer, sheet_name='comparacao_modelos')

        # 6. Resultados de ablação
        ablation_results.to_excel(writer, sheet_name='ablation_analysis', index=False)

        # 7. Métricas detalhadas por threshold
        thresholds_data = []
        for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
            y_pred_thresh = (y_pred_proba_catboost >= threshold).astype(int)
            thresholds_data.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_test, y_pred_thresh),
                'precision': precision_score(y_test, y_pred_thresh),
                'recall': recall_score(y_test, y_pred_thresh),
                'f1': f1_score(y_test, y_pred_thresh)
            })
        pd.DataFrame(thresholds_data).to_excel(writer, sheet_name='threshold_analysis', index=False)

    print(f"💾 Arquivo Excel completo salvo: {excel_path}")
    print("   Abas incluídas:")
    print("   - dataset_features: Dataset completo com todas as features")
    print("   - acuracia_por_time: Acurácia detalhada por time")
    print("   - metricas_gerais: Métricas gerais do modelo")
    print("   - feature_importances: Importância das features")
    print("   - comparacao_modelos: Comparação com modelos baseline")
    print("   - ablation_analysis: Resultados da ablação de features")
    print("   - threshold_analysis: Análise por diferentes thresholds")

except Exception as e:
    print(f"⚠️ Erro ao salvar Excel: {e}")

# Salvar gráficos adicionais
try:
    # Feature importance gráfico detalhado
    if fi_df is not None:
        plt.figure(figsize=(12, 10))
        top_features = fi_df.head(25)
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.title('Top 25 Feature Importances - CatBoost', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # SHAP summary plot
    if len(X_test) >= 100:
        print("\n📊 Gerando análise SHAP...")
        sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_idx]
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, plot_type='bar', max_display=20)
        plt.title('SHAP Feature Importance (Global)')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

except Exception as e:
    print(f"⚠️ Erro ao gerar gráficos adicionais: {e}")

# Gerar requirements
generate_requirements()

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETO CONCLUÍDO!")
print("=" * 80)
print(f"📈 Performance Final do CatBoost:")
print(f"   • Acurácia: {catboost_metrics['accuracy']:.4f}")
print(f"   • AUC-ROC: {catboost_metrics['auc']:.4f}")
print(f"   • F1-Score: {catboost_metrics['f1']:.4f}")
print(f"🎯 Comparação com Baseline:")
print(f"   • Melhoria: +{(catboost_metrics['accuracy'] - baseline_acc):.4f}")
print(f"📊 Modelos Comparados: {list(baseline_results.keys())}")
print(f"💾 Arquivos Salvos:")
print(f"   • Excel completo: {excel_path}")
print(f"   • Requirements: requirements.txt")
print(f"   • Gráficos: feature_importance_{timestamp}.png, shap_summary_{timestamp}.png")
print("=" * 80)