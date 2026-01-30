from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import gc # Garbage collector per pulire la RAM
from tqdm import tqdm
import pandas as pd
import numpy as np


def add_BERT_feature(df, or_df, coll, model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

    if device == "cuda":
        model_cross_encoder.model.half()

    #pid_to_idx = {pid: i for i, pid in enumerate(collection['pid'])}
    # 1. Prepara gli ID come stringhe
    qids = df['qid'].astype(str).tolist()
    pids = df['pid'].astype(str).tolist()

    # 2. Crea dizionari di lookup (Molto veloce: O(1) per l'accesso)
    # Trasformiamo i dataframe in dizionari {id: text}
    print("Creazione indici di ricerca...")
    q_map = dict(zip(or_df['qid'].astype(str), or_df['query']))
    p_map = dict(zip(coll['pid'].astype(str), coll['passage']))

    # 3. Estrazione con Progress Bar
    print("Estrazione Queries...")
    # Usa .get(q) per evitare errori se un ID non esiste, oppure q_map[q] se sei sicuro
    queries = [q_map.get(q, "") for q in tqdm(qids)]

    print("Estrazione Passages...")
    passages = [p_map.get(p, "") for p in tqdm(pids)]

    # 2. Encode in Batch (Molto veloce su GPU)
    print("Cross Encoding...")
    cross_encoder_scores = model_cross_encoder.predict(list(zip(queries, passages)), batch_size=64, show_progress_bar=True)

    # 4. Aggiungiamo al DataFrame (spostando su CPU)

    df['cross'] = cross_encoder_scores

    # Pulizia memoria GPU
    del passages, queries, cross_encoder_scores
    torch.cuda.empty_cache()
    gc.collect()

    return df



def collection_encoder(coll, batch_size=128):
    """
    Encode collection passages e ritorna DataFrame con embeddings.
    
    Args:
        coll: DataFrame con colonne ['pid', 'passage']
        batch_size: Dimensione batch per encoding
        
    Returns:
        DataFrame con colonne ['pid', 'embedding']
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
    
    print("Encoding Collection Passages...")
    passages = coll['passage'].astype(str).tolist()
    p_embeddings = model.encode(
        passages, 
        batch_size=batch_size, 
        convert_to_tensor=True, 
        show_progress_bar=True
    )
    
    # Converti a numpy
    coll_embeddings = p_embeddings.cpu().numpy()
    
    # Crea DataFrame con pid e embedding
    emb_df = pd.DataFrame({
        'pid': coll['pid'].values,
        'embedding': list(coll_embeddings)  # Lista di array numpy
    })
    
    # Cleanup
    del p_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Embeddings generati: {len(emb_df)} documenti")
    
    return emb_df