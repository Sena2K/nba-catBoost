import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

print("=" * 60)
print("MODELO DE PREVISÃO NBA - VERSÃO CORRIGIDA (SEM DATA LEAKAGE)")
print("=" * 60)

games_file = 'nba_games_2025-09-25_01-54-11.xlsx'
players_file = 'nba_player_stats_2025-09-25_01-54-11.xlsx'

print("\n📊 Carregando dados...")
df_games = pd.read_excel(games_file)
df_players = pd.read_excel(players_file)

# Datas timezone-aware
df_games['date'] = pd.to_datetime(df_games['date'], utc=True, errors='coerce')
df_players['date'] = pd.to_datetime(df_players['date'], utc=True, errors='coerce')
df_games = df_games.dropna(subset=['date'])
df_players = df_players.dropna(subset=['date'])

# Temporada regular
regular_end_date = pd.to_datetime('2025-04-15', utc=True)
df_games = df_games[df_games['date'] <= regular_end_date].copy()
df_players = df_players[df_players['date'] <= regular_end_date].copy()

print(f"✅ Dados carregados: {len(df_games)} jogos, {len(df_players)} stats de jogadores")

# Renomear colunas
df_games = df_games.rename(columns={
    'away_score': 'visitor_score',
    'home_team_code': 'home_team',
    'away_team_code': 'visitor_team'
})

# Target
df_games['home_win'] = (df_games['home_score'] > df_games['visitor_score']).astype(int)
print(f"✅ Target criado. Distribuição: {df_games['home_win'].value_counts(normalize=True).to_dict()}")

def safe_temporal_filter(df, current_date, buffer_days=1):
    safe_date = current_date - timedelta(days=buffer_days)
    return df[df['date'] < safe_date]

def adaptive_otw(team_data, base_window=10, min_window=5, max_window=25):
    if len(team_data) < base_window:
        return max(min_window, min(len(team_data), base_window))
    recent_games = team_data.tail(base_window)
    if 'net_rating' in recent_games.columns and len(recent_games) >= 5:
        volatility = recent_games['net_rating'].std()
        mean_nr = recent_games['net_rating'].mean()
        thr = abs(mean_nr) * 0.1
        if volatility > thr:
            return max(min_window, base_window - 3)
        return min(max_window, base_window + 5)
    return base_window

def calculate_advanced_features_fixed(df_games, df_players, window=10, use_adaptive_otw=True):
    features_list = []

    print(f"\n🔄 Calculando features avançadas (window={window}, adaptive={use_adaptive_otw})...")
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

        # Net ratings históricos
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
            home_window = adaptive_otw(home_prev_games, window)
            visitor_window = adaptive_otw(visitor_prev_games, window)
            actual_window = min(home_window, visitor_window)
        else:
            actual_window = window

        home_recent = home_prev_games.tail(actual_window)
        visitor_recent = visitor_prev_games.tail(actual_window)

        features = {
            'game_id': game['game_id'],
            'date': game_date,
            'home_team': home_team,
            'visitor_team': visitor_team,
            'home_win': game['home_win'],
            'actual_window': actual_window
        }

        # 1) Win rate e pontos
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
        features['home_avg_points'] = float(np.mean(hps)) if hps else 0.0
        features['visitor_avg_points'] = float(np.mean(vps)) if vps else 0.0
        features['home_avg_points_allowed'] = float(np.mean(hpa)) if hpa else 0.0
        features['visitor_avg_points_allowed'] = float(np.mean(vpa)) if vpa else 0.0

        features['home_net_rating'] = features['home_avg_points'] - features['home_avg_points_allowed']
        features['visitor_net_rating'] = features['visitor_avg_points'] - features['visitor_avg_points_allowed']

        # 2) Descanso
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

        # 3) Densidade de jogos
        week_ago = game_date - timedelta(days=7)
        features['home_games_last_week'] = int((home_prev_games['date'] >= week_ago).sum())
        features['visitor_games_last_week'] = int((visitor_prev_games['date'] >= week_ago).sum())

        # 4) Streak
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

        # 5) Head-to-head
        h2h_games = past_games[
            ((past_games['home_team'] == home_team) & (past_games['visitor_team'] == visitor_team)) |
            ((past_games['home_team'] == visitor_team) & (past_games['visitor_team'] == home_team))
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

        # 6) Splits casa/fora
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

        # 7) Momentum recente
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

        # 8) Temporais
        features['month'] = int(game_date.month)
        features['day_of_week'] = int(game_date.dayofweek)
        features['is_weekend'] = 1 if game_date.dayofweek >= 5 else 0

        season_start = pd.to_datetime(f'{game_date.year}-10-01', utc=True)
        if game_date.month < 6:
            season_start = pd.to_datetime(f'{game_date.year - 1}-10-01', utc=True)
        days_into_season = int((game_date - season_start).days)
        features['days_into_season'] = days_into_season
        features['season_phase'] = int(min(3, days_into_season // 60))

        # 9) Diferenças
        features['win_rate_diff'] = features['home_win_rate'] - features['visitor_win_rate']
        features['net_rating_diff'] = features['home_net_rating'] - features['visitor_net_rating']
        features['points_diff'] = features['home_avg_points'] - features['visitor_avg_points']
        features['rest_advantage'] = features['home_rest_days'] - features['visitor_rest_days']
        features['momentum_diff'] = features['home_momentum'] - features['visitor_momentum']
        features['streak_diff'] = features['home_streak'] - features['visitor_streak']

        # 10) Volatilidade
        if len(home_recent) >= 3:
            features['home_score_volatility'] = float(np.std([
                g['home_score'] if g['home_team'] == home_team else g['visitor_score']
                for _, g in home_recent.iterrows()
            ]))
        else:
            features['home_score_volatility'] = 10.0

        if len(visitor_recent) >= 3:
            features['visitor_score_volatility'] = float(np.std([
                g['home_score'] if g['home_team'] == visitor_team else g['visitor_score']
                for _, g in visitor_recent.iterrows()
            ]))
        else:
            features['visitor_score_volatility'] = 10.0

        # 11) Matchup específico
        features['home_off_vs_visitor_def'] = features['home_avg_points'] - features['visitor_avg_points_allowed']
        features['visitor_off_vs_home_def'] = features['visitor_avg_points'] - features['home_avg_points_allowed']

        features_list.append(features)

    df_feat = pd.DataFrame(features_list)
    # Limpeza básica
    return df_feat.replace([np.inf, -np.inf], np.nan).dropna()

class TemporalValidator:
    def __init__(self, n_splits=5, test_size=30, gap_days=1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap_days = gap_days

    def split(self, df):
        splits = []
        df_sorted = df.sort_values('date')
        total_samples = len(df_sorted)
        if total_samples < (self.n_splits + 1) * 20:
            return splits
        test_samples_per_split = total_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start_idx = total_samples - (i + 1) * test_samples_per_split
            test_end_idx = total_samples - i * test_samples_per_split

            test_indices = df_sorted.index[test_start_idx:test_end_idx]

            test_start_date = df_sorted.iloc[test_start_idx]['date']
            train_cutoff = test_start_date - timedelta(days=self.gap_days)
            train_indices = df_sorted[df_sorted['date'] < train_cutoff].index

            if len(train_indices) > 50 and len(test_indices) > 10:
                splits.append((train_indices, test_indices))

        return splits

print("\n🔍 Otimizando Hiperparâmetros com Validação Temporal...")

configs = [
    {'window': 8, 'adaptive': False},
    {'window': 10, 'adaptive': False},
    {'window': 12, 'adaptive': False},
    {'window': 15, 'adaptive': False},
    {'window': 10, 'adaptive': True},
    {'window': 12, 'adaptive': True}
]

best_config = None
best_auc = 0.0
config_results = {}

for config in configs:
    print(f"\nTestando configuração: {config}")

    df_features = calculate_advanced_features_fixed(
        df_games, df_players,
        window=config['window'],
        use_adaptive_otw=config['adaptive']
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

    categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend', 'season_phase']
    all_features = numerical_features + categorical_features
    target = 'home_win'

    validator = TemporalValidator(n_splits=3, gap_days=1)
    splits = validator.split(df_features)

    if len(splits) == 0:
        print("  ❌ Nenhum split válido")
        continue

    fold_aucs = []
    for train_idx, val_idx in splits:
        train_df = df_features.loc[train_idx]
        val_df = df_features.loc[val_idx]

        X_train = train_df[all_features]
        y_train = train_df[target]
        X_val = val_df[all_features]
        y_val = val_df[target]

        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=10,
            cat_features=categorical_features,
            random_seed=42,
            verbose=False,
            eval_metric='AUC',
            early_stopping_rounds=20,
            subsample=0.8
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        fold_aucs.append(auc)

    mean_auc = float(np.mean(fold_aucs))
    config_results[str(config)] = {'auc': mean_auc, 'std': float(np.std(fold_aucs))}
    print(f"  AUC médio: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_config = config

print(f"\n✅ Melhor configuração: {best_config} (AUC: {best_auc:.4f})")

print("\n🤖 Treinamento final com configuração otimizada...")

df_features_final = calculate_advanced_features_fixed(
    df_games, df_players,
    window=best_config['window'],
    use_adaptive_otw=best_config['adaptive']
)

print(f"\n🔍 Dataset final: {len(df_features_final)} jogos")
print(f"Período: {df_features_final['date'].min().date()} até {df_features_final['date'].max().date()}")
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

categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend', 'season_phase']
all_features = numerical_features + categorical_features
target = 'home_win'

split_date = df_features_final['date'].quantile(0.8)
train_mask = df_features_final['date'] < split_date
test_mask = df_features_final['date'] >= split_date

train_df = df_features_final[train_mask]
test_df = df_features_final[test_mask]

print(f"\n📅 Split temporal:")
print(f"Treino: {len(train_df)} jogos ({train_df['date'].min().date()} a {train_df['date'].max().date()})")
print(f"Teste: {len(test_df)} jogos ({test_df['date'].min().date()} a {test_df['date'].max().date()})")

X_train = train_df[all_features]
y_train = train_df[target]
X_test = test_df[all_features]
y_test = test_df[target]

final_model = CatBoostClassifier(
    iterations=400,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=8,
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
    eval_metric='AUC',
    early_stopping_rounds=30,
    subsample=0.85,
    rsm=0.8  # amostragem de colunas correta para CatBoost
)

print("Treinando modelo final...")
final_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=50
)

# Avaliação final
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)

print("\n📈 RESULTADOS FINAIS:")
print("=" * 60)
print(f"Acurácia: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Log Loss: {ll:.4f}")

# Baseline correto: classe majoritária
p = y_test.mean()
baseline_acc = max(p, 1 - p)
print(f"Baseline (classe majoritária): {baseline_acc:.4f}")
print(f"Melhoria sobre baseline: +{(acc - baseline_acc):.4f}")

# Análise de viés
home_preds_pct = float((y_pred_proba > 0.5).mean())
print(f"\nViés do modelo: {home_preds_pct:.1%} previsões para time da casa")

# Calibração isotônica opcional
calibrated_model = None
y_pred_proba_cal = y_pred_proba

if home_preds_pct > 0.65 or home_preds_pct < 0.45:
    print("⚠️ Aplicando calibração devido a viés detectado...")

    # Subconjunto temporal do treino para calibrar
    calib_cut = train_df['date'].quantile(0.85)
    X_calib = train_df[train_df['date'] >= calib_cut][all_features]
    y_calib = train_df[train_df['date'] >= calib_cut][target]
    if len(X_calib) < 100:
        X_calib = X_train
        y_calib = y_train

    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_calib, y_calib)

    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_pred_proba_cal >= 0.5).astype(int)

    acc_cal = accuracy_score(y_test, y_pred_cal)
    auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
    ll_cal = log_loss(y_test, y_pred_proba_cal)

    print("\n📈 Resultados após calibração isotônica:")
    print("=" * 60)
    print(f"Acurácia calibrada: {acc_cal:.4f}")
    print(f"AUC-ROC calibrada: {auc_cal:.4f}")
    print(f"Log Loss calibrado: {ll_cal:.4f}")

    # Curvas de confiabilidade
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
    except Exception as e:
        print(f"Falha ao plotar curva de confiabilidade: {e}")
else:
    print("Sem calibração adicional.")

# Matriz de confusão e relatório
from sklearn.metrics import ConfusionMatrixDisplay
def avaliar_predictions(y_true, proba, thresh=0.5, titulo="Sem calibração"):
    preds = (proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    print(f"\n🔎 Avaliação {titulo} (threshold={thresh:.2f})")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, preds, digits=4))

avaliar_predictions(y_test, y_pred_proba, 0.5, "Bruta")
if calibrated_model is not None:
    avaliar_predictions(y_test, y_pred_proba_cal, 0.5, "Calibrada")

# Importância de features
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
except Exception as e:
    print(f"Falha ao calcular ou plotar importâncias: {e}")

# SHAP
try:
    n_sample = min(500, len(X_test))
    if n_sample >= 50:
        X_shap = X_test.sample(n=n_sample, random_state=42)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_shap)

        print("\n📊 Plot SHAP summary das principais variáveis")
        shap.summary_plot(shap_values, X_shap, show=True, plot_type='bar', max_display=20)
        shap.summary_plot(shap_values, X_shap, show=True, max_display=20)
    else:
        print("Amostra muito pequena para SHAP, pulando...")
except Exception as e:
    print(f"Falha ao gerar SHAP: {e}")

# Salvar artefatos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
try:
    final_model.save_model(f"catboost_nba_model_{timestamp}.txt")
    fi_df.to_csv(f"feature_importances_{timestamp}.csv", index=False)
    df_features_final.to_csv(f"dataset_features_{timestamp}.csv", index=False)
    print(f"\n💾 Artefatos salvos:")
    print(f"Modelo: catboost_nba_model_{timestamp}.cbm")
    print(f"Importâncias: feature_importances_{timestamp}.csv")
    print(f"Dataset de features: dataset_features_{timestamp}.csv")
except Exception as e:
    print(f"Falha ao salvar artefatos: {e}")

print("\n✅ Pipeline concluído.")
