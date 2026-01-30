
import pandas as pd
import torch
from tqdm import tqdm
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disabilita componenti non necessari

def preprocess_text(text, use_lemma=True, remove_stopwords=True, min_token_len=2):

    """
    Preprocessing avanzato per query e documenti
    
    Args:
        text: testo da preprocessare
        use_lemma: usa lemmatization (True) o lascia tokens originali (False)
        remove_stopwords: rimuovi stopwords (meno aggressive)
        min_token_len: lunghezza minima token (default 2, evita 'a', 'i', etc.)
    
    Returns:
        testo preprocessato come stringa
    """

    CUSTOM_STOP_WORDS = STOP_WORDS - {
    'no', 'not', 'nor', 'none', 'nobody', 'nothing', 'neither', 'never',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'first', 'last', 'top', 'bottom', 'back', 'front',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such'
}
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Rimuovi URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Rimuovi email
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Sostituisci caratteri speciali con spazi (mantieni solo alfanumerici e spazi)
    # IMPORTANTE: mantieni gli apostrofi per contrazioni (don't, it's, etc.)
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    
    # 5. Rimuovi numeri isolati (opzionale, dipende dal dominio)
    # Se i numeri sono importanti (es. anni, versioni), commenta questa riga
    # text = re.sub(r'\b\d+\b', '', text)
    
    # 6. Normalizza spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Lemmatization con spaCy (più veloce e accurato di NLTK)
    if use_lemma:
        doc = nlp(text)
        tokens = []
        
        for token in doc:
            # Filtra:
            # - stopwords (se richiesto)
            # - punteggiatura
            # - token troppo corti
            # - token che sono solo spazi
            if (not remove_stopwords or token.text not in CUSTOM_STOP_WORDS) and \
               not token.is_punct and \
               len(token.text) >= min_token_len and \
               token.text.strip():
                # Usa il lemma (forma base della parola)
                tokens.append(token.lemma_)
        
        text = ' '.join(tokens)
    else:
        # Se non usiamo lemma, almeno rimuovi stopwords
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in CUSTOM_STOP_WORDS and len(w) >= min_token_len]
            text = ' '.join(words)
    
    return text.strip()


def preprocess_collection(collection_df, text_column='passage', cache_path=None):
    """
    Preprocessa l'intera collezione di documenti
    Può salvare/caricare da cache per velocizzare
    """
    import os
    
    # Prova a caricare da cache
    if cache_path and os.path.exists(cache_path):
        print(f"Caricamento collezione preprocessata da cache: {cache_path}")
        return pd.read_parquet(cache_path)
    
    print("Preprocessing collezione documenti...")
    collection_processed = collection_df.copy()
    
    # Preprocessa in batch per efficienza con spaCy
    texts = collection_processed[text_column].fillna("").astype(str).tolist()
    
    # Process in batches con spaCy pipe (molto più veloce)
    processed_texts = []
    batch_size = 1000
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Preprocessing docs"):
        batch = texts[i:i+batch_size]
        batch_processed = [preprocess_text(t) for t in batch]
        processed_texts.extend(batch_processed)
    
    collection_processed[text_column + '_processed'] = processed_texts
    
    # Salva cache
    if cache_path:
        print(f"Salvataggio cache in: {cache_path}")
        collection_processed.to_parquet(cache_path)
    
    return collection_processed

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

def generate_hybrid_negatives(
    df, 
    pre_collection,  # Collection con preprocessing BM25
    emb_collection,  # Collection con embeddings precalcolati
    bm25,            # Modello BM25 già inizializzato
    BATCH_SIZE=200, 
    NUM_NEGATIVES=99
):
    """
    Genera negativi combinando BM25 e Cosine Similarity usando dati precalcolati.
    
    Strategia:
    1. Per ogni query, recupera top-K candidati con BM25
    2. Per ogni query, recupera top-K candidati con Cosine Similarity
    3. Unisce le due liste senza duplicati
    4. Calcola ENTRAMBI gli score per tutte le coppie finali
    
    Args:
        df: DataFrame originale con coppie positive (label=1)
        pre_collection: DataFrame con ['pid', 'passage', 'passage_processed']
        emb_collection: DataFrame con ['pid', 'embedding'] (embeddings precalcolati)
        bm25: Modello BM25GPU già inizializzato
        BATCH_SIZE: Dimensione batch per processing
        NUM_NEGATIVES: Numero di negativi per query
        
    Returns:
        DataFrame con colonne: qid, pid, query, passage, bm25_score, cosine_sim, label
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. PREPARAZIONE DATI
    print("Preparazione strutture dati...")
    
    # Dizionari per accesso rapido
    pid_to_text = pd.Series(
        pre_collection['passage_processed'].values, 
        index=pre_collection['pid']
    ).to_dict()
    
    pid_to_index = {pid: i for i, pid in enumerate(pre_collection['pid'].values)}
    index_to_pid = pre_collection['pid'].values
    
    # Embedding matrix (assumendo che emb_collection sia allineato con pre_collection)
    # Se gli embeddings sono già tensori, altrimenti converti
    emb_matrix = torch.tensor(
        np.stack(emb_collection['embedding'].values), 
        device=device, 
        dtype=torch.float32
    )
    
    # 2. PREPROCESSING QUERY
    print("Preprocessing query...")
    new_df = df[df['label'] == 1].reset_index(drop=True).copy()
    new_df['query_processed'] = new_df['query'].apply(preprocess_text)
    
    # Encode query embeddings (usa il tuo modello)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    print("Encoding query...")
    instruction = "Represent this sentence for searching relevant passages: "
    queries = new_df['query_processed'].tolist()
    queries = [instruction + q  for q in queries]
    query_embeddings = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
    
    # 3. PROCESSING PER BATCH
    augmented_rows = []
    num_batches = (len(new_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(new_df))
        
        batch_df = new_df.iloc[start:end]
        batch_q_emb = query_embeddings[start:end]
        
        with torch.no_grad():
            # --- A. RETRIEVAL BM25 ---
            bm25_scores_matrix = bm25.score_batch(
                batch_df['query_processed'].tolist()
            )
            bm25_top_scores, bm25_top_indices = torch.topk(
                bm25_scores_matrix, 
                k=NUM_NEGATIVES + 10
            )
            
            # --- B. RETRIEVAL COSINE SIMILARITY ---
            # Calcola similarità con TUTTI i passaggi
            cosine_scores_matrix = torch.mm(
                batch_q_emb, 
                emb_matrix.T
            )  # Shape: (batch_size, num_passages)
            
            cos_top_scores, cos_top_indices = torch.topk(
                cosine_scores_matrix,
                k=NUM_NEGATIVES + 10
            )
        
        # Sposta su CPU
        bm25_top_indices = bm25_top_indices.cpu().numpy()
        bm25_top_scores = bm25_top_scores.cpu().numpy()
        cos_top_indices = cos_top_indices.cpu().numpy()
        cos_top_scores = cos_top_scores.cpu().numpy()
        bm25_scores_full = bm25_scores_matrix.cpu().numpy()
        cosine_scores_full = cosine_scores_matrix.cpu().numpy()
        
        # --- C. COSTRUZIONE RIGHE ---
        for j, (_, row) in enumerate(batch_df.iterrows()):
            qid = row['qid']
            query_txt = row['query_processed']
            pos_pid = row['pid']
            pos_idx = pid_to_index[pos_pid]
            
            # 1. DOCUMENTO POSITIVO
            pos_text = pid_to_text.get(pos_pid, "")
            pos_bm25 = float(bm25_scores_full[j, pos_idx])
            pos_cosine = float(cosine_scores_full[j, pos_idx])
            
            augmented_rows.append({
                'qid': qid,
                'pid': pos_pid,
                'query': query_txt,
                'passage': pos_text,
                'bm25_score': pos_bm25,
                'cosine_sim': pos_cosine,
                'label': 1
            })
            
            # 2. NEGATIVI: UNIONE BM25 + COSINE
            # Raccogli candidati unici da entrambe le fonti
            bm25_candidates = set(
                index_to_pid[idx] 
                for idx in bm25_top_indices[j] 
                if index_to_pid[idx] != pos_pid
            )
            
            cosine_candidates = set(
                index_to_pid[idx]
                for idx in cos_top_indices[j]
                if index_to_pid[idx] != pos_pid
            )
            
            # Unione senza duplicati
            all_candidates = list(bm25_candidates | cosine_candidates)
            
            # Limita a NUM_NEGATIVES
            # Opzione: ordina per score combinato (es. media normalizzata)
            # Per semplicità, prendiamo i primi NUM_NEGATIVES
            candidate_scores = []
            for pid in all_candidates:
                idx = pid_to_index[pid]
                bm25_sc = float(bm25_scores_full[j, idx])
                cos_sc = float(cosine_scores_full[j, idx])
                candidate_scores.append((pid, bm25_sc, cos_sc))
            
            # Ordina per score combinato (es. somma normalizzata)
            # Normalizza BM25 e Cosine separatamente prima di combinare
            if candidate_scores:
                bm25_vals = [s[1] for s in candidate_scores]
                cos_vals = [s[2] for s in candidate_scores]
                
                bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
                cos_min, cos_max = min(cos_vals), max(cos_vals)
                
                # Evita divisione per zero
                bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
                cos_range = cos_max - cos_min if cos_max != cos_min else 1
                
                scored_candidates = []
                for pid, bm25_sc, cos_sc in candidate_scores:
                    bm25_norm = (bm25_sc - bm25_min) / bm25_range
                    cos_norm = (cos_sc - cos_min) / cos_range
                    combined = 0.5 * bm25_norm + 0.5 * cos_norm
                    scored_candidates.append((pid, bm25_sc, cos_sc, combined))
                
                # Ordina per score combinato decrescente
                scored_candidates.sort(key=lambda x: x[3], reverse=True)
                
                # Prendi i top NUM_NEGATIVES
                for pid, bm25_sc, cos_sc, _ in scored_candidates[:NUM_NEGATIVES]:
                    passage = pid_to_text.get(pid, "")
                    augmented_rows.append({
                        'qid': qid,
                        'pid': pid,
                        'query': query_txt,
                        'passage': passage,
                        'bm25_score': bm25_sc,
                        'cosine_sim': cos_sc,
                        'label': 0
                    })
    
    # Cleanup
    del query_embeddings, emb_matrix
    torch.cuda.empty_cache()
    gc.collect()
    
    result_df = pd.DataFrame(augmented_rows)
    print(f"\nDataset generato: {len(result_df)} righe")
    print(f"Positivi: {(result_df['label']==1).sum()}")
    print(f"Negativi: {(result_df['label']==0).sum()}")
    
    return result_df

def generate_hybrid_negatives_test(
    test_df,
    pre_collection,  # Collection con preprocessing BM25
    emb_collection,  # Collection con embeddings precalcolati
    bm25,            # Modello BM25 già inizializzato
    BATCH_SIZE=200, 
    TOP_K=100        # Numero di documenti da recuperare per query
):
    """
    Genera coppie query-document per test set combinando BM25 e Cosine Similarity.
    
    Strategia:
    1. Per ogni query, recupera top-K candidati con BM25
    2. Per ogni query, recupera top-K candidati con Cosine Similarity
    3. Unisce le due liste senza duplicati
    4. Calcola ENTRAMBI gli score per tutte le coppie finali
    
    Args:
        test_df: DataFrame con colonne ['qid', 'query'] (senza label)
        pre_collection: DataFrame con ['pid', 'passage', 'passage_processed']
        emb_collection: DataFrame con ['pid', 'embedding']
        bm25: Modello BM25GPU già inizializzato
        BATCH_SIZE: Dimensione batch per processing
        TOP_K: Numero di documenti da recuperare per query
        
    Returns:
        DataFrame con colonne: qid, pid, query, passage, bm25_score, cosine_sim
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. PREPARAZIONE DATI
    print("Preparazione strutture dati...")
    
    pid_to_text = pd.Series(
        pre_collection['passage_processed'].values, 
        index=pre_collection['pid']
    ).to_dict()
    
    pid_to_index = {pid: i for i, pid in enumerate(pre_collection['pid'].values)}
    index_to_pid = pre_collection['pid'].values
    
    # Embedding matrix
    emb_matrix = torch.tensor(
        np.stack(emb_collection['embedding'].values), 
        device=device, 
        dtype=torch.float32
    )
    
    # 2. PREPROCESSING QUERY
    print("Preprocessing query...")
    test_df = test_df.reset_index(drop=True).copy()
    test_df['query_processed'] = test_df['query'].apply(preprocess_text)
    
    # Encode query embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    print("Encoding query...")
    instruction = "Represent this sentence for searching relevant passages: "
    queries = test_df['query_processed'].tolist()
    queries = [instruction + q  for q in queries]
    query_embeddings = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
    
    # 3. PROCESSING PER BATCH
    test_rows = []
    num_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Processing Test Batches"):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(test_df))
        
        batch_df = test_df.iloc[start:end]
        batch_q_emb = query_embeddings[start:end]
        
        with torch.no_grad():
            # --- A. RETRIEVAL BM25 ---
            bm25_scores_matrix = bm25.score_batch(
                batch_df['query_processed'].tolist()
            )
            bm25_top_scores, bm25_top_indices = torch.topk(
                bm25_scores_matrix, 
                k=TOP_K + 10  # Recupera qualche extra per sicurezza
            )
            
            # --- B. RETRIEVAL COSINE SIMILARITY ---
            cosine_scores_matrix = torch.mm(
                batch_q_emb, 
                emb_matrix.T
            )
            
            cos_top_scores, cos_top_indices = torch.topk(
                cosine_scores_matrix,
                k=TOP_K + 10
            )
        
        # Sposta su CPU
        bm25_top_indices = bm25_top_indices.cpu().numpy()
        bm25_top_scores = bm25_top_scores.cpu().numpy()
        cos_top_indices = cos_top_indices.cpu().numpy()
        cos_top_scores = cos_top_scores.cpu().numpy()
        bm25_scores_full = bm25_scores_matrix.cpu().numpy()
        cosine_scores_full = cosine_scores_matrix.cpu().numpy()
        
        # --- C. COSTRUZIONE RIGHE ---
        for j, (_, row) in enumerate(batch_df.iterrows()):
            qid = row['qid']
            query_txt = row['query_processed']
            
            # UNIONE BM25 + COSINE candidati
            bm25_candidates = set(
                index_to_pid[idx] 
                for idx in bm25_top_indices[j]
            )
            
            cosine_candidates = set(
                index_to_pid[idx]
                for idx in cos_top_indices[j]
            )
            
            # Unione senza duplicati
            all_candidates = list(bm25_candidates | cosine_candidates)
            
            # Calcola score per tutti i candidati
            candidate_scores = []
            for pid in all_candidates:
                idx = pid_to_index[pid]
                bm25_sc = float(bm25_scores_full[j, idx])
                cos_sc = float(cosine_scores_full[j, idx])
                candidate_scores.append((pid, bm25_sc, cos_sc))
            
            # Ordina per score combinato
            if candidate_scores:
                bm25_vals = [s[1] for s in candidate_scores]
                cos_vals = [s[2] for s in candidate_scores]
                
                bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
                cos_min, cos_max = min(cos_vals), max(cos_vals)
                
                bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
                cos_range = cos_max - cos_min if cos_max != cos_min else 1
                
                scored_candidates = []
                for pid, bm25_sc, cos_sc in candidate_scores:
                    bm25_norm = (bm25_sc - bm25_min) / bm25_range
                    cos_norm = (cos_sc - cos_min) / cos_range
                    combined = 0.5 * bm25_norm + 0.5 * cos_norm
                    scored_candidates.append((pid, bm25_sc, cos_sc, combined))
                
                # Ordina per score combinato decrescente
                scored_candidates.sort(key=lambda x: x[3], reverse=True)
                
                # Prendi i top TOP_K
                for pid, bm25_sc, cos_sc, _ in scored_candidates[:TOP_K]:
                    passage = pid_to_text.get(pid, "")
                    test_rows.append({
                        'qid': qid,
                        'pid': pid,
                        'query': query_txt,
                        'passage': passage,
                        'bm25_score': bm25_sc,
                        'cosine_sim': cos_sc
                    })
    
    # Cleanup
    del query_embeddings, emb_matrix
    torch.cuda.empty_cache()
    gc.collect()
    
    result_df = pd.DataFrame(test_rows)
    print(f"\nTest dataset generato: {len(result_df)} righe")
    print(f"Query uniche: {result_df['qid'].nunique()}")
    print(f"Media documenti per query: {len(result_df) / result_df['qid'].nunique():.1f}")
    
    return result_df
def generate_hybrid_negatives_validation(
    val_df,
    pre_collection,  # Collection con preprocessing BM25
    emb_collection,  # Collection con embeddings precalcolati
    bm25,            # Modello BM25 già inizializzato
    BATCH_SIZE=200, 
    TOP_K=100        # Numero di documenti da recuperare per query
):
    """
    Genera coppie query-document per validation set combinando BM25 e Cosine Similarity.
    
    Strategia REALISTICA:
    1. Recupera top-K candidati ignorando completamente i label
    2. Se il documento positivo è tra i top-K, avrà label=1
    3. Tutti gli altri documenti recuperati hanno label=0
    4. Se il positivo NON è recuperato, quella query semplicemente non avrà positivi
    
    Questo simula esattamente il comportamento del test set.
    
    Args:
        val_df: DataFrame con colonne ['qid', 'query', 'pid', 'label']
        pre_collection: DataFrame con ['pid', 'passage', 'passage_processed']
        emb_collection: DataFrame con ['pid', 'embedding']
        bm25: Modello BM25GPU già inizializzato
        BATCH_SIZE: Dimensione batch per processing
        TOP_K: Numero di documenti da recuperare per query
        
    Returns:
        DataFrame con colonne: qid, pid, query, passage, bm25_score, cosine_sim, label
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. PREPARAZIONE DATI
    print("Preparazione strutture dati...")
    
    pid_to_text = pd.Series(
        pre_collection['passage_processed'].values, 
        index=pre_collection['pid']
    ).to_dict()
    
    pid_to_index = {pid: i for i, pid in enumerate(pre_collection['pid'].values)}
    index_to_pid = pre_collection['pid'].values
    
    # Embedding matrix
    emb_matrix = torch.tensor(
        np.stack(emb_collection['embedding'].values), 
        device=device, 
        dtype=torch.float32
    )
    
    # 2. ESTRAI SOLO DOCUMENTI POSITIVI (label=1)
    print("Estrazione documenti positivi...")
    val_positives = val_df[val_df['label'] == 1].reset_index(drop=True).copy()
    
    # Mappa qid -> positive pid
    qid_to_pos_pid = dict(zip(val_positives['qid'], val_positives['pid']))
    
    # 3. PREPROCESSING QUERY
    print("Preprocessing query...")
    val_positives['query_processed'] = val_positives['query'].apply(preprocess_text)
    
    # Encode query embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    print("Encoding query...")
    instruction = "Represent this sentence for searching relevant passages: "
    queries = val_positives['query_processed'].tolist()
    queries = [instruction + q  for q in queries]
    query_embeddings = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
    
    # 4. PROCESSING PER BATCH
    val_rows = []
    num_batches = (len(val_positives) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Processing Validation Batches"):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(val_positives))
        
        batch_df = val_positives.iloc[start:end]
        batch_q_emb = query_embeddings[start:end]
        
        with torch.no_grad():
            # --- A. RETRIEVAL BM25 ---
            bm25_scores_matrix = bm25.score_batch(
                batch_df['query_processed'].tolist()
            )
            bm25_top_scores, bm25_top_indices = torch.topk(
                bm25_scores_matrix, 
                k=TOP_K + 10
            )
            
            # --- B. RETRIEVAL COSINE SIMILARITY ---
            cosine_scores_matrix = torch.mm(
                batch_q_emb, 
                emb_matrix.T
            )
            
            cos_top_scores, cos_top_indices = torch.topk(
                cosine_scores_matrix,
                k=TOP_K + 10
            )
        
        # Sposta su CPU
        bm25_top_indices = bm25_top_indices.cpu().numpy()
        bm25_top_scores = bm25_top_scores.cpu().numpy()
        cos_top_indices = cos_top_indices.cpu().numpy()
        cos_top_scores = cos_top_scores.cpu().numpy()
        bm25_scores_full = bm25_scores_matrix.cpu().numpy()
        cosine_scores_full = cosine_scores_matrix.cpu().numpy()
        
        # --- C. COSTRUZIONE RIGHE ---
        for j, (_, row) in enumerate(batch_df.iterrows()):
            qid = row['qid']
            query_txt = row['query_processed']
            pos_pid = row['pid']  # Documento corretto
            
            # UNIONE BM25 + COSINE candidati (SENZA guardare pos_pid)
            bm25_candidates = set(
                index_to_pid[idx] 
                for idx in bm25_top_indices[j]
            )
            
            cosine_candidates = set(
                index_to_pid[idx]
                for idx in cos_top_indices[j]
            )
            
            # Unione senza duplicati
            all_candidates = list(bm25_candidates | cosine_candidates)
            
            # Calcola score per tutti i candidati
            candidate_scores = []
            for pid in all_candidates:
                idx = pid_to_index[pid]
                bm25_sc = float(bm25_scores_full[j, idx])
                cos_sc = float(cosine_scores_full[j, idx])
                candidate_scores.append((pid, bm25_sc, cos_sc))
            
            # Ordina per score combinato
            if candidate_scores:
                bm25_vals = [s[1] for s in candidate_scores]
                cos_vals = [s[2] for s in candidate_scores]
                
                bm25_min, bm25_max = min(bm25_vals), max(bm25_vals)
                cos_min, cos_max = min(cos_vals), max(cos_vals)
                
                bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
                cos_range = cos_max - cos_min if cos_max != cos_min else 1
                
                scored_candidates = []
                for pid, bm25_sc, cos_sc in candidate_scores:
                    bm25_norm = (bm25_sc - bm25_min) / bm25_range
                    cos_norm = (cos_sc - cos_min) / cos_range
                    combined = 0.5 * bm25_norm + 0.5 * cos_norm
                    scored_candidates.append((pid, bm25_sc, cos_sc, combined))
                
                # Ordina per score combinato decrescente
                scored_candidates.sort(key=lambda x: x[3], reverse=True)
                
                # Prendi i top TOP_K e basta
                for pid, bm25_sc, cos_sc, _ in scored_candidates[:TOP_K]:
                    passage = pid_to_text.get(pid, "")
                    label = 1 if pid == pos_pid else 0
                    
                    val_rows.append({
                        'qid': qid,
                        'pid': pid,
                        'query': query_txt,
                        'passage': passage,
                        'bm25_score': bm25_sc,
                        'cosine_sim': cos_sc,
                        'label': label
                    })
    
    # Cleanup
    del query_embeddings, emb_matrix
    torch.cuda.empty_cache()
    gc.collect()
    
    result_df = pd.DataFrame(val_rows)
    
    # Statistiche
    print(f"\nValidation dataset generato: {len(result_df)} righe")
    print(f"Query uniche: {result_df['qid'].nunique()}")
    print(f"Positivi totali: {(result_df['label']==1).sum()}")
    print(f"Negativi totali: {(result_df['label']==0).sum()}")
    print(f"Recall@{TOP_K}: {(result_df['label']==1).sum() / len(val_positives) * 100:.2f}%")
    
    return result_df
