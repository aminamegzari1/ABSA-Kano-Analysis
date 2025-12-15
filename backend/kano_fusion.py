# kano_fusion.py
import pandas as pd

def compute_kano_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    df contient :
    - review
    - aspect
    - sentiment (positive | negative)

    Retourne un DataFrame contenant :
    cs+ (normalisé)
    cs- (normalisé)
    category (Kano)
    """

    # -----------------------------
    # 1. Calcul des ratios
    # -----------------------------
    ratio = (
        df.groupby("aspect")["sentiment"]
        .value_counts(normalize=True)
        .rename("ratio")
        .reset_index()
        .pivot(index="aspect", columns="sentiment", values="ratio")
        .fillna(0.0)
    )

    ratio["pos_ratio"] = ratio.get("positive", 0.0)
    ratio["neg_ratio"] = ratio.get("negative", 0.0)

    # -----------------------------
    # 2. cs+ et cs- EXACTEMENT comme dans votre notebook
    # -----------------------------
    max_pos = ratio["pos_ratio"].max() or 1
    max_neg = ratio["neg_ratio"].max() or 1

    ratio["cs+"] = ratio["pos_ratio"] / max_pos
    ratio["cs-"] = ratio["neg_ratio"] / max_neg

    # -----------------------------
    # 3. Classification Kano (mêmes règles que notebook LLM)
    # -----------------------------
    def classify(row):
        csplus = row["cs+"]
        csminus = row["cs-"]

        if csplus >= 0.5 and csminus < 0.5:
            return "Attractive"
        elif csplus < 0.5 and csminus >= 0.5:
            return "Must-be"
        elif csplus >= 0.5 and csminus >= 0.5:
            return "One-dimensional"
        else:
            return "Indifferent"

    ratio["category"] = ratio.apply(classify, axis=1)

    return ratio
