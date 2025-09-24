from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os, fitz, hashlib, pickle, numpy as np, joblib
from sentence_transformers import SentenceTransformer
from docx import Document
from src.pipeline import clean_text

app = Flask(__name__)
CORS(app)

# === Lazy-loaded BERT ===
bert_model = None
def get_bert_model():
    global bert_model
    if bert_model is None:
        bert_model = SentenceTransformer("models/fine_tuned_MiniLM")
    return bert_model

def extract_text(file):
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".txt":
            return file.read().decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            text = ""
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text
        elif ext == ".docx":
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def interpret_score(score):
    if score > 0.85: return "Highly similar"
    elif score > 0.6: return "Possibly related"
    return "Likely original"

@app.route("/smart-check", methods=["POST"])
def smart_check():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "File is required"}), 400
    try:
        text = clean_text(extract_text(file))
        filename = secure_filename(file.filename)
        new_embedding = get_bert_model().encode(text)

        corpus_path = "models/embeddings/corpus_embeddings.pkl"
        if os.path.exists(corpus_path):
            with open(corpus_path, "rb") as f:
                filenames, embeddings = pickle.load(f)
        else:
            filenames, embeddings = [], np.empty((0, 384))

        if len(embeddings) > 0:
            scores = np.dot(embeddings, new_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(new_embedding)
            )
            max_score = float(np.max(scores))
            most_similar_doc = filenames[np.argmax(scores)]
        else:
            max_score = 0.0
            most_similar_doc = None

        response = {
            "similarity_score": float(round(max_score, 4)),
            "verdict": interpret_score(max_score),
            "most_similar_doc": most_similar_doc,
            "added_to_corpus": False
        }

        if max_score < 0.8:
            hash_name = hashlib.sha1(text.encode()).hexdigest()[:10]
            new_filename = f"{hash_name}_{filename}"
            os.makedirs("data/corpus", exist_ok=True)
            save_path = os.path.join("data/corpus", new_filename)
            with open(save_path, "w", encoding="utf-8") as f_out:
                f_out.write(text)

            filenames.append(new_filename)
            embeddings = np.vstack([embeddings, new_embedding])

            with open(corpus_path, "wb") as f_out:
                pickle.dump((filenames, embeddings), f_out)

            response["added_to_corpus"] = True

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Smart Check backend running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
