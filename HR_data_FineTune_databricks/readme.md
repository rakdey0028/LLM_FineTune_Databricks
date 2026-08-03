# 🚀 LLM Fine-Tuning with Databricks

> End-to-end pipeline for fine-tuning Large Language Models (LLMs) using Databricks, Unsloth, Hugging Face Transformers, PEFT (LoRA/QLoRA), MLflow, and Databricks Model Serving.

---

## 📌 Overview

This repository demonstrates a complete LLM fine-tuning workflow on Databricks using an HR Policy use case.

The project covers the entire lifecycle:

- 📖 Non-Instruction (Continued Pretraining)
- 💬 Supervised Instruction Fine-Tuning (SFT)
- 👍 Preference Fine-Tuning (DPO/Preference Training)
- 📊 LLM Evaluation
- 📦 MLflow Model Registration
- 🚀 Databricks Model Serving
- 🧪 Inference Testing

The objective is to build an enterprise-ready LLM capable of answering HR policy questions with high accuracy.

---

# 🏗 Project Structure

```
HR_data_FineTune_databricks/
│
├── 00_setup.py
│
├── FineTune/
│   ├── Non_Instruct_FT.py
│   ├── Instruction_FT.py
│   └── Preference_FT.py
│
├── utilities/
│   ├── config.py
│   ├── inference.py
│   ├── model_deploy.py
│   ├── model_serving.py
│   ├── model_evaluate.py
│   └── __init__.py
│
├── LLM_Evaluation/
│   └── generate_evaluate_report.py
│
├── Model_Serving/
│   ├── model_deploy_serving.py
│   └── inference_test.py
│
├── requirements.txt
└── setup.py
```

---

# ✨ Features

- End-to-end LLM Fine-Tuning
- LoRA / QLoRA Training
- Continued Pretraining
- Instruction Fine-Tuning
- Preference Optimization
- MLflow Tracking
- Model Registration
- Databricks Unity Catalog Support
- Databricks Model Serving
- Automated Evaluation
- Enterprise Deployment Ready

---

# 🧠 Fine-Tuning Pipeline

```text
Base LLM
     │
     ▼
Non-Instruction Fine-Tuning
     │
     ▼
Instruction Fine-Tuning (SFT)
     │
     ▼
Preference Fine-Tuning
     │
     ▼
LLM Evaluation
     │
     ▼
MLflow Registration
     │
     ▼
Databricks Model Serving
     │
     ▼
Inference API
```

---

# 📂 Modules

## 1. Non-Instruct Fine-Tuning

```
FineTune/Non_Instruct_FT.py
```

Performs continued pretraining using domain-specific HR policy data to adapt the base model to enterprise knowledge.

---

## 2. Instruction Fine-Tuning

```
FineTune/Instruction_FT.py
```

Fine-tunes the model using instruction-following datasets in Alpaca format to improve question-answering capabilities.

---

## 3. Preference Fine-Tuning

```
FineTune/Preference_FT.py
```

Optimizes response quality using preference datasets to align the model with human-preferred responses.

---

## 4. Model Evaluation

```
LLM_Evaluation/
```

Evaluates the fine-tuned model using LLM-as-a-Judge techniques and generates evaluation reports.

---

## 5. Model Deployment

Utilities for:

- MLflow Model Registration
- Unity Catalog Registration
- Databricks Serving Endpoint Deployment

---

## 6. Inference

Contains scripts for:

- Endpoint Testing
- Batch Inference
- Local Prediction

---

# ⚙️ Technology Stack

| Component | Technology |
|------------|------------|
| Platform | Databricks |
| Framework | PyTorch |
| Transformers | Hugging Face |
| Fine-Tuning | Unsloth |
| PEFT | LoRA / QLoRA |
| Training | TRL |
| Experiment Tracking | MLflow |
| Deployment | Databricks Model Serving |
| Model Registry | Unity Catalog |

---

# 📦 Requirements

Main packages used:

```text
torch
transformers
trl
peft
accelerate
bitsandbytes
datasets
huggingface_hub
unsloth
unsloth_zoo
sentencepiece
protobuf
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

## Step 1

Setup environment

```bash
python 00_setup.py
```

---

## Step 2

Run Non-Instruction Fine-Tuning

```bash
python FineTune/Non_Instruct_FT.py
```

---

## Step 3

Run Instruction Fine-Tuning

```bash
python FineTune/Instruction_FT.py
```

---

## Step 4

Run Preference Fine-Tuning

```bash
python FineTune/Preference_FT.py
```

---

## Step 5

Evaluate the model

```bash
python LLM_Evaluation/generate_evaluate_report.py
```

---

## Step 6

Deploy the model

```bash
python Model_Serving/model_deploy_serving.py
```

---

## Step 7

Test inference

```bash
python Model_Serving/inference_test.py
```

---

# 📊 MLflow Integration

The project logs:

- Parameters
- Training Metrics
- Loss Curves
- Artifacts
- Registered Models
- Model Versions

using MLflow.

---

# 🚀 Databricks Model Serving

After training:

- Register model to Unity Catalog
- Deploy to Databricks Serving Endpoint
- Query via REST API
- Perform online inference

---

# 🎯 Use Case

This repository demonstrates how to build an enterprise HR assistant capable of answering questions such as:

- Leave Policy
- Attendance Policy
- Benefits
- Travel Policy
- Working Hours
- Holidays
- Employee Guidelines
- HR Procedures

The pipeline can easily be adapted for other enterprise knowledge bases by replacing the training dataset.


---

# ⭐ Support

If you found this repository useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further improvements.
