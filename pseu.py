import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import shap

warnings.filterwarnings('ignore')


# ===== FUNÇÃO calculate_advanced_features =====
def calculate_advanced_features(df_games, df_players, window=10):
    """
    Calcula features avançadas incluindo:
    - Back-to-back games
    - Streak (sequência de vitórias/derrotas)
    - Head-to-head histórico
    - Performance em casa vs fora
    - Fadiga acumulada
    - Momentum recente

    Garante que não há data leakage usando apenas dados anteriores
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

        # ===== 3. WINNING/LOSING STREAK (CORRIGIDO) =====
        # Home team streak - usar apenas jogos anteriores à data atual
        home_streak = 0
        home_recent_games = home_prev_games.tail(min(10, len(home_prev_games)))

        for _, g in home_recent_games.iloc[::-1].iterrows():
            if g['home_team'] == home_team:
                won = g['home_score'] > g['visitor_score']
            else:
                won = g['visitor_score'] > g['home_score']

            # Primeira iteração
            if home_streak == 0:
                home_streak = 1 if won else -1
            # Continua a streak
            elif (home_streak > 0 and won) or (home_streak < 0 and not won):
                home_streak = home_streak + 1 if home_streak > 0 else home_streak - 1
            # Quebra a streak
            else:
                break

        # Visitor team streak
        visitor_streak = 0
        visitor_recent_games = visitor_prev_games.tail(min(10, len(visitor_prev_games)))

        for _, g in visitor_recent_games.iloc[::-1].iterrows():
            if g['home_team'] == visitor_team:
                won = g['home_score'] > g['visitor_score']
            else:
                won = g['visitor_score'] > g['home_score']

            # Primeira iteração
            if visitor_streak == 0:
                visitor_streak = 1 if won else -1
            # Continua a streak
            elif (visitor_streak > 0 and won) or (visitor_streak < 0 and not won):
                visitor_streak = visitor_streak + 1 if visitor_streak > 0 else visitor_streak - 1
            # Quebra a streak
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

        # ===== 8. HOME COURT ADVANTAGE =====
        features['home_advantage'] = 1

        features_list.append(features)

    return pd.DataFrame(features_list)


# ===== CONFIGURAÇÃO PARA PREVISÃO PSEUDO-AO-VIVO =====
print("=" * 60)
print("PREVISÃO PSEUDO-AO-VIVO - ÚLTIMO JOGO")
print("=" * 60)

# Carregar dados
games_file = 'nba_games_2024_2025_full_2025-09-20_03-53-42.xlsx'
players_file = 'nba_player_stats_2024_2025_full_2025-09-20_03-53-42.xlsx'

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

# Calcular features com janela ótima de 25 jogos (conforme determinado anteriormente)
optimal_window = 25
print(f"\n🚀 Criando features avançadas com janela ótima ({optimal_window} jogos)...")
df_features = calculate_advanced_features(df_games, df_players, window=optimal_window)
print(f"✅ Dataset criado com {len(df_features)} jogos válidos")

# Ordenar por data
df_features = df_features.sort_values('date').reset_index(drop=True)

# ===== SEPARAR ÚLTIMO JOGO PARA PREVISÃO PSEUDO-AO-VIVO =====
# Pegar todos os jogos exceto o último para treino
train_df = df_features.iloc[:-1].copy()
last_game = df_features.iloc[[-1]].copy()

print(f"\n📅 Separando dados:")
print(f"Treino: {len(train_df)} jogos ({train_df['date'].min().date()} a {train_df['date'].max().date()})")
print(f"Teste (pseudo-ao-vivo): 1 jogo ({last_game['date'].iloc[0].date()})")

# ===== DEFINIR FEATURES =====
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
    'home_advantage'
]

categorical_features = ['home_team', 'visitor_team', 'month', 'day_of_week', 'is_weekend']
all_features = numerical_features + categorical_features
target = 'home_win'

# Preparar dados
X_train = train_df[all_features]
y_train = train_df[target]
X_test = last_game[all_features]
y_test = last_game[target]

print(f"\n📊 Features utilizadas: {len(all_features)}")

# ===== TREINAR MODELO COM TODOS OS DADOS EXCETO O ÚLTIMO JOGO =====
print("\n🤖 Treinando modelo CatBoost...")
cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    cat_features=categorical_features,
    random_seed=42,
    verbose=False,
    eval_metric='AUC'
)

# Treinar sem validação (usando todos os dados disponíveis)
cat_model.fit(X_train, y_train, verbose=100)

# ===== FAZER PREVISÃO PARA O ÚLTIMO JOGO =====
print("\n🔮 Fazendo previsão pseudo-ao-vivo...")
y_pred_proba = cat_model.predict_proba(X_test)[:, 1]
y_pred = cat_model.predict(X_test)

# Resultado real
actual_result = "Casa vence" if y_test.iloc[0] == 1 else "Visitante vence"
actual_winner = last_game['home_team'].iloc[0] if y_test.iloc[0] == 1 else last_game['visitor_team'].iloc[0]

# Previsão
predicted_winner = last_game['home_team'].iloc[0] if y_pred[0] == 1 else last_game['visitor_team'].iloc[0]
confidence = y_pred_proba[0] if y_pred[0] == 1 else 1 - y_pred_proba[0]

print(f"\n🎯 PREVISÃO PSEUDO-AO-VIVO:")
print("=" * 50)
print(f"📅 Data: {last_game['date'].iloc[0].date()}")
print(f"🏠 Casa: {last_game['home_team'].iloc[0]}")
print(f"🛫 Visitante: {last_game['visitor_team'].iloc[0]}")
print(f"🔮 Previsão: {predicted_winner} vence (confiança: {confidence:.1%})")
print(
    f"📊 Probabilidades: {last_game['home_team'].iloc[0]} {y_pred_proba[0]:.1%} | {last_game['visitor_team'].iloc[0]} {1 - y_pred_proba[0]:.1%}")
print(f"🏆 Resultado real: {actual_winner} venceu {'✅' if predicted_winner == actual_winner else '❌'}")

# ===== ANÁLISE DETALHADA DA PREVISÃO =====
print(f"\n📈 Análise detalhada da previsão:")
print("=" * 50)

# Obter importância das features para esta previsão específica
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_test)

# Criar dataframe com as contribuições de cada feature
feature_contributions = pd.DataFrame({
    'feature': all_features,
    'contribution': shap_values[0]
}).sort_values('contribution', key=abs, ascending=False)

print("\n🔍 Top 10 features que mais influenciaram esta previsão:")
for i, row in feature_contributions.head(10).iterrows():
    direction = "➕" if row['contribution'] > 0 else "➖"
    print(f"  {direction} {row['feature']}: {row['contribution']:.4f}")

# ===== SALVAR RESULTADOS =====
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
output_file = f'nba_pseudo_live_prediction_{timestamp}.xlsx'

with pd.ExcelWriter(output_file) as writer:
    # Salvar a previsão
    prediction_df = last_game.copy()
    prediction_df['predicted_prob'] = y_pred_proba
    prediction_df['predicted'] = y_pred
    prediction_df['correct'] = (y_pred == y_test.values).astype(int)
    prediction_df.to_excel(writer, sheet_name='Prediction', index=False)

    # Salvar as contribuições das features
    feature_contributions.to_excel(writer, sheet_name='Feature_Contributions', index=False)

    # Salvar resumo
    summary_data = {
        'Metric': ['Date', 'Home Team', 'Visitor Team', 'Predicted Winner',
                   'Confidence', 'Actual Winner', 'Correct Prediction'],
        'Value': [
            last_game['date'].iloc[0].date(),
            last_game['home_team'].iloc[0],
            last_game['visitor_team'].iloc[0],
            predicted_winner,
            f"{confidence:.1%}",
            actual_winner,
            "Yes" if predicted_winner == actual_winner else "No"
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

print(f"\n✅ Resultados salvos em '{output_file}'")

# ===== PLOT DA ANÁLISE SHAP =====
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=all_features, plot_type='bar', show=False)
plt.title('Feature Contributions to Prediction')
plt.tight_layout()
plt.savefig(f'shap_analysis_pseudo_live_{timestamp}.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n📊 Gráfico de análise SHAP salvo como 'shap_analysis_pseudo_live_{timestamp}.png'")
print("\n🎉 Análise pseudo-ao-vivo finalizada com sucesso!")
print("=" * 60)