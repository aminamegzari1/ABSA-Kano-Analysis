from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import pandas as pd
from io import BytesIO

# Import internes
from absa_model import predict_sentiment
from kano_fusion import compute_kano_scores
from kano_plot import draw_custom_kano_plot
from utils import extract_comments, extract_comments_from_url


app = Flask(__name__)
CORS(app)  # Autorise le frontend (React, etc.)

# ---------------------------------------------------------
#                 ROUTE DE TEST
# ---------------------------------------------------------
@app.route("/")
def home():
    return "ABSA + Kano API is running ✔️"


# ---------------------------------------------------------
# 1) Prédiction ABSA sur une seule phrase
# ---------------------------------------------------------
@app.route("/absa/predict", methods=["GET"])
def absa_predict():
    review = request.args.get("review")
    aspect = request.args.get("aspect")

    if not review or not aspect:
        return jsonify({"error": "Parameters 'review' and 'aspect' are required"}), 400

    sentiment = predict_sentiment(review, aspect)

    return jsonify({
        "review": review,
        "aspect": aspect,
        "sentiment": sentiment
    })


# ---------------------------------------------------------
# 2) EXTRACTION DES COMMENTAIRES À PARTIR D’UN FICHIER
# ---------------------------------------------------------
@app.route("/extract-comments", methods=["POST"])
def extract_comments_route():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]
    suffix = os.path.splitext(file.filename)[-1]

    temp_path = f"temp_uploaded{suffix}"
    file.save(temp_path)

    try:
        comments = extract_comments(temp_path, suffix)
    except Exception as e:
        return jsonify({"error": f"Erreur extraction : {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({"comments": comments})


# ---------------------------------------------------------
# 3) ANALYSE ABSA → KANO À PARTIR D'UN FICHIER CSV
# ---------------------------------------------------------
@app.route("/absa/analyze-file", methods=["POST"])
def absa_analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Erreur lecture CSV : {str(e)}"}), 400

    if not {"review", "aspect"}.issubset(df.columns):
        return jsonify({"error": "Le CSV doit contenir 'review' et 'aspect'"}), 400

    # Analyse ABSA
    df["sentiment"] = df.apply(
        lambda row: predict_sentiment(str(row["review"]), str(row["aspect"])),
        axis=1
    )

    # Scores Kano
    kano_df = compute_kano_scores(df)

    # Diagramme
    fig = draw_custom_kano_plot(kano_df)

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "kano_diagram.png")

    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    with open(output_path, "wb") as f:
        f.write(img_buf.read())

    return jsonify({
        "kano_scores": kano_df.reset_index().to_dict(orient="records"),
        "diagram_url": "/download-diagram"
    })


# ---------------------------------------------------------
# 4) ANALYSE ABSA → KANO À PARTIR D’UNE URL
# ---------------------------------------------------------
@app.route("/absa/analyze-url", methods=["POST"])
def absa_analyze_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "Aucune URL fournie"}), 400

    try:
        comments = extract_comments_from_url(url)
    except Exception as e:
        return jsonify({"error": f"Erreur extraction URL : {str(e)}"}), 500

    df = pd.DataFrame([{"review": c, "aspect": "general"} for c in comments])

    df["sentiment"] = df["review"].apply(lambda r: predict_sentiment(r, "general"))

    kano_df = compute_kano_scores(df)

    fig = draw_custom_kano_plot(kano_df)

    os.makedirs("static", exist_ok=True)
    output_path = os.path.join("static", "kano_diagram.png")

    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    with open(output_path, "wb") as f:
        f.write(img_buf.read())

    return jsonify({
        "kano_scores": kano_df.reset_index().to_dict(orient="records"),
        "diagram_url": "/download-diagram"
    })


# ---------------------------------------------------------
# 5) TÉLÉCHARGER LE DIAGRAMME
# ---------------------------------------------------------
@app.route("/download-diagram", methods=["GET"])
def download_diagram():
    path = os.path.join("static", "kano_diagram.png")
    if not os.path.exists(path):
        return jsonify({"error": "Aucun diagramme généré"}), 404

    return send_from_directory("static", "kano_diagram.png")


# ---------------------------------------------------------
# LANCEMENT DU SERVEUR
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
