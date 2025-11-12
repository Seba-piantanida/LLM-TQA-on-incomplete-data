# Evaluation of LLMs Capabilities for Table Question Answering on Incomplete Structured Data

**Master’s Thesis in Computer Science Engineering**

**Author:** Sebastiano Piantanida  
**Student ID:** 10658432  
**Advisor:** Davide Martinenghi  
**Co-advisors:** Paolo Papotti, Emilia Lenzi  
**Academic Year:** 2024–2025  

---

## 📘 Overview
This thesis investigates the ability of **Large Language Models (LLMs)** to perform **Table Question Answering (TQA)** on **incomplete or partially corrupted structured data**.  
While traditional benchmarks such as WikiSQL, Spider, and WikiTableQuestions assume fully complete and accurate datasets, this research explores a more realistic setting—where missing or inconsistent information is common.

---

## 🎯 Objectives
The study aims to evaluate whether and under what conditions LLMs can:
- Reconstruct missing information through **logical inference** or **parametric knowledge**.  
- Perform **ordering and ranking** operations on incomplete data.  
- Maintain **logical coherence** and **structural correctness** despite data loss.

---

## 🧪 Methodology
A fully automated **experimental pipeline** was developed to:
1. Execute traditional SQL-like queries on complete datasets.  
2. Gradually hide or remove key attributes to simulate incompleteness.  
3. Assess the ability of different LLMs to infer missing data and preserve ranking accuracy.

The evaluation spans:
- Real and synthetic datasets of varying scales.  
- Query types ranging from **simple sorting** to **multi-table JOIN operations** and **User Defined Functions (UDFs)**.  
- **Semantic similarity** tasks using movie recommendation scenarios.

---

## 📊 Findings
The experiments reveal significant limitations in the current application of LLMs to incomplete tabular contexts.  
Key influencing factors include:
- **Query complexity**  
- **Dataset size and attribute composition**  
- **Domain specificity and language** (with better results in English)

Despite their reasoning capabilities, current LLMs remain **unreliable for operational deployment**, emphasizing the need for improvements in **robustness, scalability**, and **uncertainty management**.

---

## 📁 Repository Contents
This repository includes:
- The **datasets** (real, synthetic, and relational) used for experiments  
- The **testing pipeline** for ordering, UDF, and similarity-based evaluations  
- The **results** and analysis scripts  

> All resources are provided to support **reproducibility** and **further research** on LLM-based reasoning over incomplete structured data.

---

## 🧩 Keywords
`Large Language Models` · `Table Question Answering` · `Incomplete Data` · `Semantic Inference` · `Ranking` · `Parametric Memory`