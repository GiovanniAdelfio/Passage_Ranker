import math
from collections import Counter
import re

def group_min_max(x):
    min_val = x.min()
    max_val = x.max()
    if max_val == min_val:
        return x - min_val # O return 0
    return (x - min_val) / (max_val - min_val)

def build_idf_dict(tokenized_corpus):
    print("Calcolo IDF manuale...")
    # 1. Conta in quanti documenti appare ogni parola
    doc_freqs = Counter()
    for doc_tokens in tokenized_corpus:
        # Usiamo set() perché ci interessa se appare nel doc, non quante volte
        doc_freqs.update(set(doc_tokens))

    # 2. Calcola IDF per ogni parola
    # Formula BM25 Standard: log((N - n + 0.5) / (n + 0.5) + 1)
    N = len(tokenized_corpus)
    idf_dict = {}

    for word, freq in doc_freqs.items():
        idf = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
        idf_dict[word] = idf

    return idf_dict

def get_weighted_overlap(query, passage, idf_map, default_val=1.0):
    # Tokenizzazione semplice (adattala se usi tokenizer specifici)
    q_tokens = set(str(query).lower().split())
    p_tokens = set(str(passage).lower().split())

    # Intersezione (parole in comune)
    intersection = q_tokens.intersection(p_tokens)

    if not q_tokens:
        return 0.0

    # Somma dei pesi delle parole in comune
    intersection_weight = sum(idf_map.get(w, default_val) for w in intersection)

    # Somma dei pesi di TUTTA la query (per normalizzare tra 0 e 1)
    query_weight = sum(idf_map.get(w, default_val) for w in q_tokens)

    # Evitiamo divisione per zero
    if query_weight == 0:
        return 0.0

    return intersection_weight / query_weight


def check_exact_code_match(query, passage):
    # Cerca stringhe alfanumeriche o numeri che appaiono in entrambi
    # Questa regex cerca "parole" che contengono almeno un numero (es. A5, 2024, art.1)
    q_tokens = set(t for t in query.split() if any(c.isdigit() for c in t))
    if not q_tokens:
        return 0 # Nessun codice nella query

    # Controlla se ALMENO UNO dei codici della query è nel passaggio
    for token in q_tokens:
        if token in passage: # match esatto (case sensitive o no, decidi tu)
            return 1
    return 0

# Calcolo Overlap (parole in comune)
def get_overlap(row):
    q_tokens = set(row['query'].split())
    p_tokens = set(row['passage'].split())
    return len(q_tokens.intersection(p_tokens))

def add_features(train_final, val_final = None):
    """ Crea i dataset di training e validation con feature semplici."""
    from tqdm import tqdm
    print("1. Aggiunta feature semplici al Train Set...")
    # Pulizia stringhe
    train_final['query'] = train_final['query'].astype(str).str.lower().fillna("")
    train_final['passage'] = train_final['passage'].astype(str).str.lower().fillna("")

    # Calcolo Lunghezze
    train_final['query_len'] = train_final['query'].str.split().str.len()
    train_final['passage_len'] = train_final['passage'].str.split().str.len()

    tqdm.pandas(desc="Calcolo Overlap Train")
    train_final['overlap_count'] = train_final.progress_apply(get_overlap, axis=1)

    # Calcolo Ratio
    train_final['overlap_ratio'] = train_final['overlap_count'] / train_final['query_len']
    train_final['overlap_ratio'] = train_final['overlap_ratio'].fillna(0.0)
    # Rinominiamo per chiarezza: questo è il dataset finale per il training
    train_final = train_final.copy()

    # Creazione VALIDATION SET
    if val_final is not None:
        print("\n2. Creazione Validation Set (questo ci metterà qualche minuto)...")
        # Pulizia
        val_final['query'] = val_final['query'].astype(str).str.lower().fillna("")
        val_final['passage'] = val_final['passage'].astype(str).str.lower().fillna("")

        # Calcolo Feature Semplici per il Validation
        print("   Calcolo Feature Semplici sul Validation...")
        val_final['query_len'] = val_final['query'].str.split().str.len()
        val_final['passage_len'] = val_final['passage'].str.split().str.len()
        val_final['overlap_count'] = val_final.progress_apply(get_overlap, axis=1)
        val_final['overlap_ratio'] = val_final['overlap_count'] / val_final['query_len']
        val_final['overlap_ratio'] = val_final['overlap_ratio'].fillna(0.0)
        val_final = val_final.copy()

        print("\n--- FINITO! ---")
        print("Train shape:", train_final.shape)
        print("Val shape:", val_final.shape)
        print(train_final.head(2))
    
        return train_final, val_final
    return train_final

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def get_advanced_features(df):
    print("Calcolo Advanced Features (N-grams, Position, LCS)...")
    
    # Pre-tokenizzazione veloce (split spazi)
    # Assumiamo che le colonne siano 'query' e 'passage'
    # Convertiamo in stringhe minuscole per sicurezza
    queries = df['query'].astype(str).str.lower().str.split()
    docs = df['passage'].astype(str).str.lower().str.split()
    
    # Liste per i risultati
    bigram_overlaps = []
    first_pos_list = []
    lcs_list = []
    
    for q_tokens, d_tokens in zip(queries, docs):
        # --- 1. Bigram Overlap ---
        if len(q_tokens) < 2:
            bigram_overlaps.append(0)
        else:
            # Crea set di bigrammi (coppie)
            q_bigrams = set(zip(q_tokens, q_tokens[1:]))
            d_bigrams = set(zip(d_tokens, d_tokens[1:]))
            if len(q_bigrams) > 0:
                overlap = len(q_bigrams.intersection(d_bigrams)) / len(q_bigrams)
            else:
                overlap = 0
            bigram_overlaps.append(overlap)
            
        # --- 2. First Match Position ---
        # Troviamo la posizione della PRIMA parola della query che appare nel doc
        min_pos = 1000 # Valore alto di default (non trovato)
        q_set = set(q_tokens)
        
        for i, word in enumerate(d_tokens):
            if word in q_set:
                min_pos = i
                break # Trovata la prima, ci fermiamo
        
        # Normalizziamo (es. posizione / lunghezza doc) o lasciamo raw. 
        # Raw è meglio per LightGBM, ma capiamo a 50 se non c'è.
        if min_pos == 1000:
            first_pos_list.append(len(d_tokens)) # Penalità: lunghezza doc
        else:
            first_pos_list.append(min_pos)

        # --- 3. LCS (Longest Common Subsequence) ---
        # Usiamo SequenceMatcher (lento su testi lunghi, ma ok su passaggi brevi)
        # Per velocità, facciamo una versione semplificata sui token
        # (Se è troppo lento, togli questa parte)
        if len(q_tokens) > 0 and len(d_tokens) > 0:
            sm = SequenceMatcher(None, q_tokens, d_tokens)
            match = sm.find_longest_match(0, len(q_tokens), 0, len(d_tokens))
            lcs_list.append(match.size) # Lunghezza della sequenza comune
        else:
            lcs_list.append(0)

    # Aggiungi al DF
    df['bigram_overlap'] = bigram_overlaps
    df['first_match_pos'] = first_pos_list
    df['lcs_score'] = lcs_list
    
    return df