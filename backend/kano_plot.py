# kano_plot.py

import matplotlib.pyplot as plt
from adjustText import adjust_text

def draw_custom_kano_plot(kano_df):
    """
    kano_df contient :
    cs+
    cs-
    category
    """

    # Récupération sous forme dict
    cs_plus = {aspect: kano_df.loc[aspect, "cs+"] for aspect in kano_df.index}
    cd_moins = {aspect: -(kano_df.loc[aspect, "cs-"]) for aspect in kano_df.index}

    # Normalisation identique au notebook
    max_cs = max(cs_plus.values()) or 1
    min_cd = min(cd_moins.values()) or -1

    cs_plus_norm = {k: v / max_cs for k, v in cs_plus.items()}
    cd_moins_norm = {k: v / abs(min_cd) for k, v in cd_moins.items()}

    # Couleurs notebook
    category_colors = {
        "Attractive": "blue",
        "Must-be": "red",
        "One-dimensional": "orange",
        "Indifferent": "purple"
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    texts = []

    for aspect in kano_df.index:
        cs_val = cs_plus_norm[aspect]
        cd_val = cd_moins_norm[aspect]
        category = kano_df.loc[aspect, "category"]
        color = category_colors.get(category, "black")

        ax.scatter(cs_val, cd_val, color=color, s=80)
        texts.append(ax.text(cs_val, cd_val, aspect, fontsize=9, ha='center', va='center', color=color))

    # Quadrants identiques au notebook
    ax.axhline(-0.5, color='black', linestyle='dashed')
    ax.axvline(0.5, color='black', linestyle='dashed')


    # --- ÉTIQUETTES DES QUADRANTS KANO ---
    ax.text(0.25, -0.25, "Indifferent", fontsize=12, ha='center', color="purple")
    ax.text(0.75, -0.25, "Attractive", fontsize=12, ha='center', color="blue")
    ax.text(0.25, -0.75, "Must-be", fontsize=12, ha='center', color="red")
    ax.text(0.75, -0.75, "One-dimensional", fontsize=12, ha='center', color="orange")


    ax.set_xlabel("Satisfaction CS+ (normalisé)")
    ax.set_ylabel("Dissatisfaction CD− (normalisé)")
    ax.set_title("Diagramme Kano basé sur ABSA")

    # Légères marges
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.1, 0.1)

    adjust_text(texts, ax=ax)
    return fig
