# 🧠 Product Pricer: Estimating Product Price from Text Descriptions

A GenAI-powered pipeline to predict product prices based solely on textual features such as product descriptions, features, and titles. The project benchmarks both classical ML and LLM-based approaches, using curated datasets from the Amazon Reviews corpus.

---

## 🚀 Project Workflow

### 1. 📥 Data Loading
- Source: McAuley’s Amazon Reviews dataset (2.8M items across categories).
- Features extracted: `rating`, `title`, `features`, `description`, `price`.

### 2. 🧹 Data Curation & Cleaning
- Combined relevant textual fields into a unified prompt.
- Filtered items:
  - Minimum of 300 characters in combined text.
  - At least 15 tokens post-tokenization.
- Balanced dataset across price buckets and categories (e.g., Electronics, Automotive, Toys).

### 3. 🏗️ Feature Engineering & Baselines
- Created baseline regression models to predict price:
  - ✅ `Word2Vec + Random Forest` → **Best Baseline: $97 avg error**
  - Bag of Words + Linear Regression
  - Word2Vec + SVR
  - Feature Encoding + Linear Regression

### 4. 🤖 GPT Frontier Models (Zero-shot)
- Tested models without training (only on test set with description):
  - GPT-4o → **$76**
  - GPT-4-mini → $80
  - Claude → $101

### 5. 🔓 Open Source Models (Fine-Tuned)
- Base: `LLaMA 3.1 8B` (4-bit quantized)
- Used LORA + Tokenization + Data Collator for fine-tuning
- Achieved **$46.67 avg error**, beating GPT-4o

---

## 📊 Model Comparison Summary

| Model                  | Avg Price Error ($) |
|------------------------|---------------------|
| Word2Vec + RF          | 97                  |
| GPT-4o (Zero-shot)     | 76                  |
| LLaMA 3.1 8B (Fine-tuned) | **46.67**         |

---

## 🛠️ Tools & Frameworks
- Python, Scikit-learn
- Gensim for Word2Vec
- OpenAI API (GPT-4o, mini)
- LLaMA 3.1 via HuggingFace
- PEFT, LoRA, Transformers

---

## 📁 Folder Structure
```bash
├── data_loading.py
├── data_curation.py

```

---

## 📌 Highlights
- Fine-tuned open-source LLaMA model outperformed GPT-4o.
- Built robust preprocessing and prompt generation pipeline.
- Balanced price distribution with curated category samples.

---

> ⚡ A full-stack GenAI pipeline that showcases LLM alignment with structured price prediction goals.

