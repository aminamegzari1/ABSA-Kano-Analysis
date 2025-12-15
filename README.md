# ABSA–Kano Analysis Backend

This repository contains the backend implementation of a **customer satisfaction analysis system** combining  
**Aspect-Based Sentiment Analysis (ABSA)** with the **Kano Model**.

The system analyzes user reviews at the **aspect level**, aggregates sentiment information, computes Kano satisfaction coefficients, and automatically generates a **Kano diagram** to support product feature prioritization.

This project was developed in an **academic context** and focuses exclusively on the backend logic.

---

## Project Objectives

The main objectives of this project are:

- Perform **Aspect-Based Sentiment Analysis** on customer reviews
- Identify how each product feature contributes to **satisfaction** and **dissatisfaction**
- Apply the **Kano Model** to classify features into:
  - Must-be
  - Attractive
  - One-dimensional
  - Indifferent
- Generate a **visual Kano diagram** for interpretation and decision support

---

## System Overview

The backend pipeline follows these steps:

1. **Input reviews** (CSV file or extracted text)
2. **ABSA inference** using a fine-tuned CamemBERT model
3. **Aggregation of sentiments by aspect**
4. **Kano score computation (CS+ and CS−)**
5. **Automatic generation of the Kano diagram**

---

## Aspect-Based Sentiment Analysis (ABSA)

### Model
- Base model: **CamemBERT (camembert-base)**
- Framework: **Hugging Face Transformers**
- Task: **Aspect-Based Sentiment Classification**
- Output labels:
  - `positive`
  - `negative`

### Input Representation
Each prediction is performed on a concatenated input:

