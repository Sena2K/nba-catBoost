import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import shap
import warnings
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# ===== CONFIGURAÇÃO INICIAL =====
print("=" * 60)
print("MODELO DE PREVISÃO NBA - VERSÃO BALANCEADA")
print("=" * 60)

# Caminhos dos arquivos
games_file = 'nba_games_2024_2025_full_2025-09-20_03-53-42.xlsx'
players_file = 'nba_player_stats_2024_2025_full_2025-09-20_03-53-42.xlsx'

# Carregar dados
print("\n📊 Carregando dados...")
df_games = pd.read_excel(games_file)
df_players = pd.read_excel(players_file)

# Converter datas
df_games['date'] = pd.to_datetime(df_games['date'], utc=True)
df_players['date'] = pd.to_datetime(df_players['date'], utc=True)

# Filtrar temporada regular
regular_end_date = pd.to_datetime('2025-04-15', utc=True)
df_games = df_games[df_games['date'] <= regular_end_date]
df_players = df_players[df_players['date'] <= regular_end_date]

print(f"✅ Dados carregados: {len(df_games)} jogos, {len(df_players)} stats de jogadores")

# Criar target
df_games['home_win'] = (df_games['home_score'] > df_games['visitor_score']).astype(int)


# ===== FUNÇÃO APRIMORADA DE FEATURES =====
def calculate_advanced_features(df_games, df_players, window=10):
    """
    Calcula features avançadas incluindo:
    - Back-to-back games
    - Streak (sequência de vitórias/derrotas)
    - Head-to-head histórico
    - Performance em casa vs fora
    - Fadiga acumulada
    - Momentum recente
    """
    features_list = []

    print("\n🔄 Calculando features avançadas...")
    total_games = len(df_games)

    for idx, game in df_games.iterrows():
        if idx % 100 == 0:
            print(f"  Processando jogo {idx}/{total_games} ({idx / total_games * 100:.1f}%)")

        game_date = game['date']
        home_team = game['home_team']
        visitor_team = game['visitor_team']

        # ===== IMPORTANTE: FILTRAR APENAS JOGOS ANTERIORES À DATA ATUAL =====
        past_games = df_games[df_games['date'] < game_date]

        # Jogos anteriores de cada time
        home_prev_games = past_games[
            ((past_games['home_team'] == home_team) |
             (past_games['visitor_team'] == home_team))
        ].sort_values('date')

        visitor_prev_games = past_games[
            ((past_games['home_team'] == visitor_team) |
             (past_games['visitor_team'] == visitor_team))
        ].sort_values('date')

        # Requer mínimo de jogos históricos
        if len(home_prev_games) < 5 or len(visitor_prev_games) < 5:
            continue

        # Pegar últimos jogos para cálculo de médias
        home_recent = home_prev_games.tail(window)
        visitor_recent = visitor_prev_games.tail(window)

        features = {
            'game_id': game['game_id'],
            'date': game_date,
            'home_team': home_team,
            'visitor_team': visitor_team,
            'home_win': game['home_win']
        }

        # ===== 1. WIN RATE E PERFORMANCE BÁSICA =====
        home_wins = 0
        home_points_scored = []
        home_points_allowed = []

        for _, g in home_recent.iterrows():
            if g['home_team'] == home_team:
                won = g['home_score'] > g['visitor_score']
                home_wins += won
                home_points_scored.append(g['home_score'])
                home_points_allowed.append(g['visitor_score'])
            else:
                won = g['visitor_score'] > g['home_score']
                home_wins += won
                home_points_scored.append(g['visitor_score'])
                home_points_allowed.append(g['home_score'])

        visitor_wins = 0
        visitor_points_scored = []
        visitor_points_allowed = []

        for _, g in visitor_recent.iterrows():
            if g['home_team'] == visitor_team:
                won = g['home_score'] > g['visitor_score']
                visitor_wins += won
                visitor_points_scored.append(g['home_score'])
                visitor_points_allowed.append(g['visitor_score'])
            else:
                won = g['visitor_score'] > g['home_score']
                visitor_wins += won
                visitor_points_scored.append(g['visitor_score'])
                visitor_points_allowed.append(g['home_score'])

        features['home_win_rate'] = home_wins / len(home_recent) if len(home_recent) > 0 else 0.5
        features['visitor_win_rate'] = visitor_wins / len(visitor_recent) if len(visitor_recent) > 0 else 0.5
        features['home_avg_points'] = np.mean(home_points_scored) if home_points_scored else 0
        features['visitor_avg_points'] = np.mean(visitor_points_scored) if visitor_points_scored else 0
        features['home_avg_points_allowed'] = np.mean(home_points_allowed) if home_points_allowed else 0
        features['visitor_avg_points_allowed'] = np.mean(visitor_points_allowed) if visitor_points_allowed else 0

        # Net rating
        features['home_net_rating'] = features['home_avg_points'] - features['home_avg_points_allowed']
        features['visitor_net_rating'] = features['visitor_avg_points'] - features['visitor_avg_points_allowed']

        # ===== 2. BACK-TO-BACK GAMES =====
        if len(home_prev_games) > 0:
            last_home_game_date = home_prev_games.iloc[-1]['date']
            days_rest_home = (game_date - last_home_game_date).days
            features['home_rest_days'] = days_rest_home
            features['home_back_to_back'] = 1 if days_rest_home <= 1 else 0

            # Jogos nos últimos 7 dias (fadiga acumulada)
            week_ago = game_date - timedelta(days=7)
            home_games_last_week = len(home_prev_games[home_prev_games['date'] >= week_ago])
            features['home_games_last_week'] = home_games_last_week
        else:
            features['home_rest_days'] = 7
            features['home_back_to_back'] = 0
            features['home_games_last_week'] = 0

        if len(visitor_prev_games) > 0:
            last_visitor_game_date = visitor_prev_games.iloc[-1]['date']
            days_rest_visitor = (game_date - last_visitor_game_date).days
            features['visitor_rest_days'] = days_rest_visitor
            features['visitor_back_to_back'] = 1 if days_rest_visitor <= 1 else 0

            # Jogos nos últimos 7 dias
            visitor_games_last_week = len(visitor_prev_games[visitor_prev_games['date'] >= week_ago])
            features['visitor_games_last_week'] = visitor_games_last_week
        else:
            features['visitor_rest_days'] = 7
            features['visitor_back_to_back'] = 0
            features['visitor_games_last_week'] = 0

        # ===== 3. WINNING/LOSING STREAK =====
        # Home team streak
        home_streak = 0
        for g in home_prev_games.tail(10).iloc[::-1].iterrows():
            g = g[1]
            if g['home_team'] == home_team:
                won = g['home_score'] > g['visitor_score']
            else:
                won = g['visitor_score'] > g['home_score']

            if home_streak == 0:
                home_streak = 1 if won else -1
            elif home_streak > 0 and won:
                home_streak += 1
            elif home_streak < 0 and not won:
                home_streak -= 1
            else:
                break

        # Visitor team streak
        visitor_streak = 0
        for g in visitor_prev_games.tail(10).iloc[::-1].iterrows():
            g = g[1]
            if g['home_team'] == visitor_team:
                won = g['home_score'] > g['visitor_score']
            else:
                won = g['visitor_score'] > g['home_score']

            if visitor_streak == 0:
                visitor_streak = 1 if won else -1
            elif visitor_streak > 0 and won:
                visitor_streak += 1
            elif visitor_streak < 0 and not won:
                visitor_streak -= 1
            else:
                break

        features['home_streak'] = home_streak
        features['visitor_streak'] = visitor_streak

        # ===== 4. HEAD-TO-HEAD HISTÓRICO =====
        h2h_games = past_games[
            ((past_games['home_team'] == home_team) & (past_games['visitor_team'] == visitor_team)) |
            ((past_games['home_team'] == visitor_team) & (past_games['visitor_team'] == home_team))
            ].tail(5)

        if len(h2h_games) > 0:
            h2h_home_wins = 0
            for _, g in h2h_games.iterrows():
                if g['home_team'] == home_team:
                    h2h_home_wins += 1 if g['home_score'] > g['visitor_score'] else 0
                else:
                    h2h_home_wins += 0 if g['home_score'] > g['visitor_score'] else 1
            features['h2h_home_win_rate'] = h2h_home_wins / len(h2h_games)
        else:
            features['h2h_home_win_rate'] = 0.5

        # ===== 5. PERFORMANCE EM CASA VS FORA =====
        # Home team quando joga em casa
        home_at_home = past_games[past_games['home_team'] == home_team].tail(10)
        if len(home_at_home) > 0:
            home_at_home_wins = (home_at_home['home_score'] > home_at_home['visitor_score']).sum()
            features['home_team_home_win_rate'] = home_at_home_wins / len(home_at_home)
        else:
            features['home_team_home_win_rate'] = 0.5

        # Visitor team quando joga fora
        visitor_away = past_games[past_games['visitor_team'] == visitor_team].tail(10)
        if len(visitor_away) > 0:
            visitor_away_wins = (visitor_away['visitor_score'] > visitor_away['home_score']).sum()
            features['visitor_team_away_win_rate'] = visitor_away_wins / len(visitor_away)
        else:
            features['visitor_team_away_win_rate'] = 0.5

        # ===== 6. MOMENTUM (últimos 3 jogos) =====
        home_last_3 = home_prev_games.tail(3)
        visitor_last_3 = visitor_prev_games.tail(3)

        home_momentum = 0
        for _, g in home_last_3.iterrows():
            if g['home_team'] == home_team:
                home_momentum += 1 if g['home_score'] > g['visitor_score'] else -1
            else:
                home_momentum += 1 if g['visitor_score'] > g['home_score'] else -1
        features['home_momentum'] = home_momentum / 3 if len(home_last_3) > 0 else 0

        visitor_momentum = 0
        for _, g in visitor_last_3.iterrows():
            if g['home_team'] == visitor_team:
                visitor_momentum += 1 if g['home_score'] > g['visitor_score'] else -1
            else:
                visitor_momentum += 1 if g['visitor_score'] > g['home_score'] else -1
        features['visitor_momentum'] = visitor_momentum / 3 if len(visitor_last_3) > 0 else 0

        # ===== 7. FEATURES DE CALENDÁRIO =====
        features['month'] = game_date.month
        features['day_of_week'] = game_date.dayofweek
        features['is_weekend'] = 1 if game_date.dayofweek >= 5 else 0

        # ===== 8. NOVAS FEATURES DIFERENCIAIS =====
        # Diferença de qualidade entre os times
        features['win_rate_diff'] = features['home_win_rate'] - features['visitor_win_rate']
        features['net_rating_diff'] = features['home_net_rating'] - features['visitor_net_rating']
        features['points_diff'] = features['home_avg_points'] - features['visitor_avg_points']

        # Vantagem de descanso
        features['rest_advantage'] = features['home_rest_days'] - features['visitor_rest_days']

        features_list.append(features)

    return pd.DataFrame(features_list)


# ===== OTW OPTIMIZATION =====
print("\n🔍 Otimizando Optimal Time Window (OTW)...")
windows = [5, 10, 15, 20, 25, 30]
otw_results = {}

# Criar datasets para diferentes windows
df_features_dict = {}
for w in windows:
    print(f"Calculando features para window={w}...")
    df_features_dict[w] = calculate_advanced_features(df_games, df_players, window=w)
    df_features_dict[w] = df_features_dict[w].sort_values('date').reset_index(drop=True)

# Definir features (REMOVENDO home_advantage)
numerical_features = [
    'home_win_rate', 'visitor_win_rate',
    'home_avg_points', 'visitor_avg_points',
    'home_avg_points_allowed', 'visitor_avg_points_allowed',
    'home_net_rating', 'visitor_net_rating',
    'home_rest_days', 'visitor_rest_days',
    'home_back_to_back', 'visitor_back_to_back',
    'home_games_last_week', 'visitor_games_last_week',
    'home_streak', 'visitor_streak',
    'h2h_home_win_rate',
    'home_team_home_win_rate', 'visitor_team_away_win_rate',
    'home_momentum', 'visitor_momentum',
    # REMOVIDO: 'home_advantage' - estava enviesando o modelo
    # NOVAS FEATURES:
    'win_rate_diff', 'net_rating_diff', 'points_diff', 'rest_advantage'
]

categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend']
all_features = numerical_features + categorical_features
target = 'home_win'

# Para cada window, fazer split temporal e treinar CatBoost
for w in windows:
    df_features = df_features_dict[w]

    if len(df_features) == 0:
        continue

    split_date = df_features['date'].quantile(0.8)
    train_mask = df_features['date'] < split_date
    val_mask = df_features['date'] >= split_date  # Usando o que era test como val para OTW

    train_df = df_features[train_mask]
    val_df = df_features[val_mask]

    if len(train_df) < 10 or len(val_df) < 5:
        continue

    X_train = train_df[all_features]
    y_train = train_df[target]
    X_val = val_df[all_features]
    y_val = val_df[target]

    # Usar parâmetros para reduzir overfitting
    cat_model = CatBoostClassifier(
        iterations=300,  # Reduzido
        depth=5,  # Profundidade menor
        learning_rate=0.03,  # Taxa de aprendizado menor
        l2_leaf_reg=5,  # Mais regularização
        cat_features=categorical_features,
        random_seed=42,
        verbose=False,
        eval_metric='AUC',
        early_stopping_rounds=30,
        subsample=0.8,  # Subsample para reduzir overfitting
        colsample_bylevel=0.8
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=0
    )

    y_pred_proba = cat_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, cat_model.predict(X_val))
    ll = log_loss(y_val, y_pred_proba)

    otw_results[w] = {'auc': auc, 'acc': acc, 'll': ll}
    print(f"Window {w}: AUC={auc:.4f}, Acc={acc:.4f}, LogLoss={ll:.4f}")

# Selecionar melhor window baseado em AUC
if otw_results:
    best_window = max(otw_results, key=lambda x: otw_results[x]['auc'])
    print(f"\n✅ Melhor OTW: {best_window} jogos (AUC={otw_results[best_window]['auc']:.4f})")
else:
    best_window = 10
    print("\n⚠️ Nenhum window válido encontrado, usando default 10")

# Usar o melhor window para o dataset final
df_features = df_features_dict[best_window]

# ===== VERIFICAÇÃO DE DATA LEAKAGE =====
print("\n🔍 Verificando integridade dos dados...")
print(f"Período dos dados: {df_features['date'].min().date()} até {df_features['date'].max().date()}")
print(f"Distribuição do target: {df_features['home_win'].value_counts(normalize=True).to_dict()}")

# ===== SPLIT TEMPORAL CORRETO =====
split_date = df_features['date'].quantile(0.8)
train_mask = df_features['date'] < split_date
test_mask = df_features['date'] >= split_date

train_df = df_features[train_mask]
test_df = df_features[test_mask]

print(f"\n📅 Data de corte: {split_date.date()}")
print(f"Treino: {len(train_df)} jogos ({train_df['date'].min().date()} a {train_df['date'].max().date()})")
print(f"Teste: {len(test_df)} jogos ({test_df['date'].min().date()} a {test_df['date'].max().date()})")

# Preparar dados
X_train = train_df[all_features]
y_train = train_df[target]
X_test = test_df[all_features]
y_test = test_df[target]

print(f"\n📊 Features utilizadas: {len(all_features)}")
print(f"  - Numéricas: {len(numerical_features)}")
print(f"  - Categóricas: {len(categorical_features)}")

# ===== TREINAR MODELOS =====
print(f"\n🤖 Treinando modelos com OTW={best_window}...")

# 1. CatBoost com parâmetros ajustados
print("\n1️⃣ CatBoost...")
cat_model = CatBoostClassifier(
    iterations=300,
    depth=5,
    learning_rate=0.03,
    l2_leaf_reg=5,
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
    eval_metric='AUC',
    early_stopping_rounds=30,
    subsample=0.8,
    colsample_bylevel=0.8
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=100
)

# Predições
y_pred_cat = cat_model.predict(X_test)
y_pred_proba_cat = cat_model.predict_proba(X_test)[:, 1]

# 2. Logistic Regression (baseline)
print("\n2️⃣ Logistic Regression...")
logreg = LogisticRegression(max_iter=1000, C=1.0)
logreg.fit(X_train[numerical_features], y_train)
y_pred_logreg = logreg.predict(X_test[numerical_features])
y_pred_proba_logreg = logreg.predict_proba(X_test[numerical_features])[:, 1]

# 3. Random Forest
print("\n3️⃣ Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train[numerical_features], y_train)
y_pred_rf = rf.predict(X_test[numerical_features])
y_pred_proba_rf = rf.predict_proba(X_test[numerical_features])[:, 1]

# ===== MÉTRICAS DE AVALIAÇÃO =====
print("\n📈 RESULTADOS DOS MODELOS:")
print("=" * 60)

models = {
    'CatBoost': (y_pred_cat, y_pred_proba_cat),
    'Logistic Regression': (y_pred_logreg, y_pred_proba_logreg),
    'Random Forest': (y_pred_rf, y_pred_proba_rf)
}

for name, (y_pred, y_pred_proba) in models.items():
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)

    print(f"\n{name}:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")

# Baseline: sempre prever time da casa
baseline_home = np.ones(len(y_test))
baseline_acc = accuracy_score(y_test, baseline_home)
print(f"\n📊 Baseline (sempre home): {baseline_acc:.4f}")

# ===== ANALISAR VIÉS =====
print("\n🔍 Analisando viés do modelo:")
home_preds = y_pred_proba_cat
home_win_rate = sum(home_preds > 0.5) / len(home_preds)
print(f"Percentual de previsões para time da casa: {home_win_rate:.1%}")

# Se ainda estiver muito alto (>65%), aplicar calibração
if home_win_rate > 0.65:
    print("⚠️  Viés alto detectado, aplicando calibração...")
    calibrated_model = CalibratedClassifierCV(cat_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    y_pred_proba_cat = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_cat = (y_pred_proba_cat > 0.5).astype(int)

    # Recalcular métricas
    acc = accuracy_score(y_test, y_pred_cat)
    auc = roc_auc_score(y_test, y_pred_proba_cat)
    ll = log_loss(y_test, y_pred_proba_cat)

    print(f"✅ Após calibração - Acurácia: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {ll:.4f}")

# ===== FEATURE IMPORTANCE =====
print("\n🎯 Top 15 Features Mais Importantes (CatBoost):")
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(15).to_string())

# ===== SHAP EXPLAINABILITY =====
print("\n🔍 Calculando SHAP values...")
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_test)

# Resumo global
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_feature_importance.png', dpi=150)
plt.close()

# Exemplo local para o primeiro jogo de teste
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], show=False)
shap.save_html("shap_force_plot.html", force_plot)

print("✅ SHAP analysis saved: shap_feature_importance.png and shap_force_plot.html")

# ===== PREVISÃO DOS PRÓXIMOS JOGOS =====
print("\n🔮 PREVISÃO DOS PRÓXIMOS JOGOS:")
print("=" * 60)

# Pegar os últimos jogos do dataset de teste como "próximos jogos"
next_games = test_df.tail(10).copy()
next_games_features = next_games[all_features]

# Fazer previsões
predictions = cat_model.predict_proba(next_games_features)[:, 1]

# Mostrar previsões
print("\nPróximos jogos e previsões:\n")
for idx, (_, game) in enumerate(next_games.iterrows()):
    prob_home = predictions[idx]
    prob_visitor = 1 - prob_home
    predicted_winner = game['home_team'] if prob_home > 0.5 else game['visitor_team']
    confidence = max(prob_home, prob_visitor)
    actual_winner = game['home_team'] if game['home_win'] == 1 else game['visitor_team']

    print(f"📅 {game['date'].date()}")
    print(f"   {game['home_team']} vs {game['visitor_team']}")
    print(f"   Previsão: {predicted_winner} vence (confiança: {confidence:.1%})")
    print(f"   Probabilidades: {game['home_team']} {prob_home:.1%} | {game['visitor_team']} {prob_visitor:.1%}")
    print(f"   Resultado real: {actual_winner} {'✅' if predicted_winner == actual_winner else '❌'}")
    print()

# ===== ANÁLISE DE CALIBRAÇÃO =====
print("\n📊 Análise de Calibração do Modelo...")
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_proba_cat, n_bins=10
)

# Plot de calibração
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Curva de calibração
axes[0, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="CatBoost")
axes[0, 0].plot([0, 1], [0, 1], "k:", label="Perfeitamente calibrado")
axes[0, 0].set_xlabel("Probabilidade média predita")
axes[0, 0].set_ylabel("Fração de positivos")
axes[0, 0].set_title("Curva de Calibração")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribuição das probabilidades
axes[0, 1].hist(y_pred_proba_cat[y_test == 0], bins=20, alpha=0.5, label='Visitor ganhou', color='red')
axes[0, 1].hist(y_pred_proba_cat[y_test == 1], bins=20, alpha=0.5, label='Home ganhou', color='blue')
axes[0, 1].set_xlabel("Probabilidade predita (Home ganha)")
axes[0, 1].set_ylabel("Frequência")
axes[0, 1].set_title("Distribuição das Probabilidades")
axes[0, 1].legend()

# 3. Feature importance (top 10)
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel("Importância")
axes[1, 0].set_title("Top 10 Features")
axes[1, 0].invert_yaxis()

# 4. Performance ao longo do tempo
test_df['week'] = test_df['date'].dt.isocalendar().week
test_df['predicted'] = y_pred_cat
weekly_acc = test_df.groupby('week').apply(
    lambda x: accuracy_score(x['home_win'], x['predicted'])
)
axes[1, 1].plot(weekly_acc.index, weekly_acc.values, 'o-')
axes[1, 1].axhline(y=accuracy_score(y_test, y_pred_cat), color='r', linestyle='--', label='Média')
axes[1, 1].set_xlabel("Semana do ano")
axes[1, 1].set_ylabel("Acurácia")
axes[1, 1].set_title("Performance Semanal")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_analysis_complete.png', dpi=150)
plt.close()

# ===== SALVAR RESULTADOS =====
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

# Adicionar previsões ao dataframe de teste
test_df['predicted_prob'] = y_pred_proba_cat
test_df['predicted'] = y_pred_cat

# Converter timezone para None antes de salvar
test_df['date'] = pd.to_datetime(test_df['date']).dt.tz_localize(None)

# Salvar Excel com múltiplas abas
output_file = f'nba_predictions_balanced_{timestamp}.xlsx'
with pd.ExcelWriter(output_file) as writer:
    test_df.to_excel(writer, sheet_name='Predictions', index=False)
    feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)

    # OTW results
    otw_df = pd.DataFrame.from_dict(otw_results, orient='index')
    otw_df.index.name = 'window'
    otw_df.to_excel(writer, sheet_name='OTW_Optimization')

    # Resumo do modelo
    summary_data = {
        'Metric': ['Accuracy', 'AUC-ROC', 'Log Loss', 'Total Games', 'Train Period', 'Test Period', 'Best OTW',
                   'Home Prediction Bias'],
        'Value': [
            f"{accuracy_score(y_test, y_pred_cat):.4f}",
            f"{roc_auc_score(y_test, y_pred_proba_cat):.4f}",
            f"{log_loss(y_test, y_pred_proba_cat):.4f}",
            len(test_df),
            f"{train_df['date'].min().date()} to {train_df['date'].max().date()}",
            f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
            str(best_window),
            f"{home_win_rate:.1%}"
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Model_Summary', index=False)

print(f"\n✅ Resultados salvos em '{output_file}'")
print("\n🎉 Análise completa finalizada com sucesso!")
print("=" * 60)


print("\n📌 Matriz de confusão geral (CatBoost):")
print(confusion_matrix(y_test, y_pred_cat))
print("\n📌 Relatório de classificação geral:")
print(classification_report(y_test, y_pred_cat, digits=3))

test_tmp = test_df.copy()
test_tmp['pred_home_win'] = y_pred_cat
acc_home_games = (test_tmp[test_tmp['home_win']==1]['pred_home_win']==1).mean()
acc_away_games = (test_tmp[test_tmp['home_win']==0]['pred_home_win']==0).mean()
print(f"\nAcurácia quando mandante vence: {acc_home_games:.3f}")
print(f"Acurácia quando visitante vence: {acc_away_games:.3f}")
