# LLMOps with Kubernetes

**Hands-on labs to fine-tune, deploy, observe, and scale GenAI workloads on Kubernetes — from zero to production.**

This course walks you through building a complete **LLMOps workflow** for the *Atharva Dental Clinic Assistant* — a domain-adapted chatbot powered by **Retrieval-Augmented Generation (RAG)** and a **LoRA-fine-tuned LLM**.  
You’ll start from a blank KIND cluster and progress all the way to a fully automated, observable, and scalable GenAI system powered by **KServe**, **vLLM**, **Prometheus**, and **ArgoCD**.

---

## What You’ll Learn

- Deploy and operate **LLM-based applications** on Kubernetes  
- Generate **synthetic domain data** and build a **FAISS retriever**
- **Fine-tune** open LLMs with **LoRA** on CPU-only environments  
- Package models as **OCI artifacts** and mount them via **ImageVolumes**
- Serve models using **KServe (RawDeployment)** + **vLLM**
- Instrument and visualize **LLM observability metrics** in Prometheus & Grafana  
- Implement **autoscaling (HPA, VPA, KEDA)** for GenAI services  
- Automate continuous delivery using **ArgoCD GitOps**

---

## Lab Index

Each lab builds on the previous one, progressively assembling the full GenAI stack.

| # | Lab Title | Description |
|:-:|:-----------|:-------------|
| **[Lab 00 – Setup](lab00.md)** | Create a 3-node KIND cluster with ImageVolume feature gates and namespaces for ML, app, and monitoring workloads. |
| **[Lab 01 – Synthetic Data + RAG System](lab01.md)** | Generate synthetic clinic data, build a FAISS (or TF-IDF) index, and deploy the Retriever API on Kubernetes. |
| **[Lab 02 – CPU LoRA Fine-tuning (SmolLM2-135M)](lab02.md)** | Fine-tune a small open LLM on CPU using LoRA and merge adapters into a single model folder for serving. |
| **[Lab 03 – Packaging Model as OCI Image](lab03.md)** | Package the merged model into an OCI image and mount it using Kubernetes ImageVolumes. |
| **[Lab 04 – Serving with KServe and vLLM](lab04.md)** | Serve the model with vLLM (OpenAI-compatible) through KServe RawDeployment and connect it to the RAG retriever and Chat API. |
| **[Lab 04.0 – Testing vLLM on macOS](lab040.md)** | Optional: run the vLLM container locally on macOS to validate CPU-only inference before deploying to Kubernetes. |
| **[Lab 05 – Observability with Prometheus + Grafana](lab05.md)** | Instrument vLLM, Retriever, and Chat API with Prometheus metrics; build a Grafana dashboard for model and system insights. |
| **[Lab 06 – Autoscaling vLLM + RAG + Chat API](lab06.md)** | Implement CPU-based HPA, Prometheus-driven scaling via KEDA, and VPA recommendations for resource optimization. |
| **[Lab 07 – GitOps for GenAI with ArgoCD](lab07.md)** | Manage continuous delivery for all GenAI components using ArgoCD App-of-Apps and GitOps workflows. |

---

## Tech Stack

- **Kubernetes 1.34+ (KIND)** – local multi-node cluster  
- **KServe RawDeployment + vLLM** – model serving  
- **ImageVolumes** – mount OCI model artifacts  
- **PEFT LoRA / SmolLM2-135M** – CPU-friendly fine-tuning  
- **FAISS / TF-IDF Retriever** – semantic search backbone  
- **Prometheus + Grafana** – observability and metrics  
- **KEDA / HPA / VPA** – intelligent autoscaling  
- **ArgoCD** – GitOps-based continuous delivery  

---

## How to Use This Guide

1. Follow the labs **in order** — each lab builds on artifacts from previous ones.  
2. All commands are copy-paste-ready for macOS / Linux (KIND + kubectl + helm).  
3. Expect small CPU-only workloads suitable for laptops with ≥ 16 GB RAM.  
4. Each lab ends with a **“Lab Summary”** to recap what you achieved.  

---

> **Author:** [Gourav Shah](https://www.linkedin.com/in/gouravshah)  
> **Project:** [School of DevOps – LLMOps Bootcamp: Kubernetes for GenAI Workloads](https://github.com/schoolofdevops/llmops-labuide)  
> **License:** CC BY-NC-SA © 2025 Gourav Shah · School of DevOps

---
