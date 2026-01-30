import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt



def train_ranker(train_final, val_final, feature_cols, ):
    """
    Addestra un modello di Learning to Rank usando LightGBM Ranker.
    
    Args:
        train_final (pd.DataFrame): DataFrame di training con feature e label.
        val_final (pd.DataFrame): DataFrame di validazione con feature e label.
    """
    # 1. Definiamo quali feature usare per il ranking

    print("Preparazione dati per il Ranking...")

    # --- A. ORDINAMENTO ---
    # Il Ranker DEVE avere le righe raggruppate per query_id contigui
    train_final = train_final.sort_values('qid')
    val_final = val_final.sort_values('qid')

    # --- B. Creazione Input (X) e Target (y) ---
    X_train = train_final[feature_cols]
    y_train = train_final['label'] # 1=Rilevante, 0=Non Rilevante

    X_val = val_final[feature_cols]
    y_val = val_final['label']

    # --- C. Calcolo dei Gruppi ---
    q_train = train_final.groupby('qid', sort=False).size().to_numpy()
    q_val = val_final.groupby('qid', sort=False).size().to_numpy()

    # Controllo che i conti tornino
    print(f"Righe Train: {len(X_train)} | Somma Gruppi Train: {sum(q_train)}")
    assert sum(q_train) == len(X_train)
    assert sum(q_val) == len(X_val)

    # --- D. ADDESTRAMENTO ---
    print("Inizio Training del Modello...")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",        # Ottimizziamo MAP (Mean Average Precision)
        eval_at=[10],        # top 10
        n_estimators=1000,   # Max alberi
        learning_rate=0.05,  # Velocità di apprendimento
        importance_type='gain',
        random_state=42,
        verbose=-1
    )

    # Facciamo partire l'addestramento
    ranker.fit(
        X_train, y_train,
        group=q_train,
        eval_set=[(X_val, y_val)],
        eval_group=[q_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100), # Si ferma se non migliora per 50 giri
            lgb.log_evaluation(period=100)           # Stampa ogni 100 giri
        ]
    )

    # --- E. RISULTATI ---
    print("\n--- RISULTATO FINALE ---")
    # Best score è un dizionario tipo {'valid_0': {'map@10': 0.45}}
    best_score = ranker.best_score_['valid_0']['map@10']
    print(f"Miglior MAP@10 sul Validation Set: {best_score:.4f}")

    # Grafico importanza feature
    lgb.plot_importance(ranker, importance_type='gain', title='Quali feature contano di più?')
    plt.show()
    return ranker, best_score

def train_ranker_tuned(train_final, val_final, feature_cols):
    """
    Addestra un modello di Learning to Rank usando LightGBM Ranker.

    Args:
        train_final (pd.DataFrame): DataFrame di training con feature e label.
        val_final (pd.DataFrame): DataFrame di validazione con feature e label.
    """
    # 1. Definiamo quali feature usare per il ranking

    #print("Preparazione dati per il Ranking...")

    # --- A. ORDINAMENTO ---
    # Il Ranker DEVE avere le righe raggruppate per query_id contigui
    train_final = train_final.sort_values('qid')
    val_final = val_final.sort_values('qid')

    # --- B. Creazione Input (X) e Target (y) ---
    X_train = train_final[feature_cols]
    y_train = train_final['label'] # 1=Rilevante, 0=Non Rilevante

    X_val = val_final[feature_cols]
    y_val = val_final['label']

    # --- C. Calcolo dei Gruppi ---
    q_train = train_final.groupby('qid', sort=False).size().to_numpy()
    q_val = val_final.groupby('qid', sort=False).size().to_numpy()

    # Controllo che i conti tornino
    print(f"Righe Train: {len(X_train)} | Somma Gruppi Train: {sum(q_train)}")
    assert sum(q_train) == len(X_train)
    assert sum(q_val) == len(X_val)

    # --- D. ADDESTRAMENTO ---
    #print("Inizio Training del Modello...")

    ranker = lgb.LGBMRanker(

        objective="lambdarank",
        metric="map",
        eval_at=[10],
        n_estimators=6000,      # Aumentato: lasciamolo imparare a fondo
        learning_rate=0.01,     # LENTISSIMO: per non memorizzare subito i duplicati
        num_leaves=60,          # Più "intelligente"
        max_depth=-1,

        # PARAMETRI ANTI-DUPLICATI
        min_child_samples=100,  # Obbliga ogni foglia ad avere almeno 100 esempi (ignora gruppetti piccoli di duplicati)
        subsample=0.7,          # Bagging: usa solo il 70% dei dati a ogni albero (così i duplicati variano)
        subsample_freq=1,       # Bagging a ogni iterazione
        colsample_bytree=0.8,   # Usa solo l'80% delle feature a ogni albero

        reg_lambda=10,          # Regolarizzazione forte per evitare overfitting
        random_state=42,
        n_jobs=-1
    )

    # Facciamo partire l'addestramento
    ranker.fit(
        X_train, y_train,
        group=q_train,
        eval_set=[(X_val, y_val)],
        eval_group=[q_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300) # Si ferma se non migliora per 50 giri
            lgb.log_evaluation(period=100)           # Stampa ogni 100 giri
        ]
    )

    # --- E. RISULTATI ---
    #print("\n--- RISULTATO FINALE ---")
    # Best score è un dizionario tipo {'valid_0': {'map@10': 0.45}}
    best_score = ranker.best_score_['valid_0']['map@10']
    print(f"Miglior MAP@10 sul Validation Set: {best_score:.4f}")

    # Grafico importanza feature
    lgb.plot_importance(ranker, importance_type='gain', title='Quali feature contano di più?')
    plt.show()
    return ranker, best_score
