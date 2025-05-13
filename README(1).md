# 🧠 Product Pricer: Estimating Product Price from Text Descriptions

A full-stack GenAI pipeline that predicts the price of Amazon products using only their text descriptions. From classical ML to cutting-edge open-source LLMs, we benchmark a range of models to explore how well language alone can reveal value.

---

## 🚀 Workflow Overview

This project unfolds over four main phases:

### 1️⃣ Import & Load Data
- 📦 Source: McAuley-Lab Amazon Reviews 2023 Dataset via HuggingFace Datasets
- 📊 Categories: Home & Kitchen, Automotive, Video Games, and more
- ⚙️ Uses load_dataset() to fetch “raw_meta” product metadata (title, features, description, price)

```python
load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}", split="full")
```

---

### 2️⃣ Advanced Data Curation
- 🧽 Combine product title, description, and features into a unified content string
- ✂️ Clean unnecessary patterns (e.g., "Batteries Included?", serial codes)
- 🔍 Filter items:
  - ≥ 300 characters of content
  - ≥ 150 tokens after LLaMA-3 tokenization
  - Price range between $0.50 – $999.49

> Prompts are structured as:
>
> ❓ “How much does this cost to the nearest dollar?”
>
> Followed by product details and a label: “Price is $X.00”

---

### 3️⃣ Preprocessing, Tokenization & LoRA Setup
- 🤖 Tokenizer: Meta’s LLaMA 3.1 8B via Hugging Face
- 🪶 Prompts formatted for both zero-shot and supervised fine-tuning
- 🔧 Ready for parameter-efficient finetuning (LoRA) with PEFT and Transformers

---

### 4️⃣ Modeling & Evaluation

We test three tiers of models:

| 🧪 Model Type        | Model Name               | Avg Prediction Error ($) |
|---------------------|--------------------------|---------------------------|
| 📉 Classical ML      | Word2Vec + Random Forest | 97                        |
| 🌐 Frontier LLM      | GPT-4o (Zero-shot)       | 76                        |
| 🌱 Open Source LLM   | LLaMA 3.1 8B (LoRA-tuned) | 🚀 46.67                 |

---

## 🔍 Detailed Model Highlights

### ✅ Baseline Models
- Word2Vec embeddings with scikit-learn regressors
- RF outperforms other baselines like SVR and Linear Regression
- Performance: ~97 dollar mean error

### 🤯 Frontier LLMs (Zero-shot)
- GPT-4o and Claude tested directly with prompt-only inference
- GPT-4o achieves $76 average error — surprisingly solid!

### 🔓 Open-Source + Fine-Tuning
- LLaMA 3.1 8B, 4-bit quantized
- Finetuned on prompt-label pairs using LoRA (PEFT)
- Beats GPT-4o with $46.67 average error!

---

## 📁 Repository Structure

```
.
├── the_product_pricer.py      # Model benchmarking and testing
├── loaders.py                 # Parallelized data ingestion
├── items.py                   # Data cleaning & prompt builder
├── llama_lora_config.json     # LoRA fine-tuning setup
├── README.md                  # This file ✨
```

---

## 🛠️ Tech Stack

- 🧠 LLMs: GPT-4o (via OpenAI API), Meta LLaMA 3.1 8B
- 🛠️ ML: Scikit-learn, Gensim Word2Vec
- 🔧 Tokenization: Hugging Face Transformers
- ⚙️ Finetuning: PEFT, LoRA, BitsAndBytes
- 🔄 Parallel Processing: ProcessPoolExecutor

---

## ✨ Why This Project Rocks

- Built from scratch with production-ready data pipelines
- Proves open-source LLMs can outperform proprietary models on numeric regression
- Showcases structured prompt formatting and aligned prediction behavior

---

## 📌 Sample Prompt

```
How much does this cost to the nearest dollar?

Sony WH-1000XM4 Wireless Noise Canceling Overhead Headphones
Industry-leading noise cancelation with Dual Noise Sensor technology
Up to 30-hour battery life
Touch Sensor controls

Price is $
```

---

## 📍 Future Enhancements

- Integrate few-shot GPT examples to test in-context learning
- Explore more categories and cross-category generalization
- Deploy on Gradio / Streamlit for live inference

---

> 🧠 From tokenized text to price prediction — The Product Pricer bridges NLP and numerical understanding with state-of-the-art models.
