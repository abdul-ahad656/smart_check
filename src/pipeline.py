from sentence_transformers import SentenceTransformer
import numpy as np

def clean_text(text):
    return text.lower().strip()

def process_document_pair(text1, text2):
    model = SentenceTransformer("models/fine_tuned_MiniLM")

    # Clean inputs
    text1_clean = clean_text(text1)
    text2_clean = clean_text(text2)

    emb1 = model.encode(text1_clean)
    emb2 = model.encode(text2_clean)

    # Cosine similarity
    sim_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return sim_score
