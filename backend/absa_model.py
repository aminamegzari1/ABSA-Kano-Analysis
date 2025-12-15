import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification

# Détection du device (GPU si dispo, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Charger le tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# 2) Charger le modèle Camembert pour classification
model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=2
)

# 3) Charger vos poids fine-tuned
state_dict = torch.load("best_camembert_fixed.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 4) Mapping label -> sentiment
label_map = {0: "negative", 1: "positive"}

def predict_sentiment(review_text: str, aspect: str):
    """
    Prend un texte + un aspect et retourne 'positive' ou 'negative'
    """
    input_text = f"{review_text} [SEP] {aspect}"

    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return label_map[prediction]
