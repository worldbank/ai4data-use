# Introduction

Monitoring how datasets are mentioned and used in research papers is essential for improving transparency, data reuse, and informed decision-making. However, dataset mentions are often hidden in unstructured text and inconsistently referenced, making them difficult to track at scale.

In this project, we introduce an automated framework for identifying dataset mentions in scientific literature. By combining large language models (LLMs), synthetic data generation, and a two-stage fine-tuning approach, the framework is able to detect and classify dataset references with high accuracy.

The core goals of this work are:

* To make dataset usage more discoverable and accessible across research domains.

* To enable scalable and accurate dataset mention extraction.

* To support the research ecosystem—researchers, funders, and policymakers—by identifying data trends, gaps, and reuse patterns.

## Project Description

This project focuses on developing a system to **automatically detect and classify dataset mentions in scientific research papers**. Unlike traditional citation tracking, which relies on structured references, dataset mentions often appear informally within the text, making them difficult to extract reliably.

To solve this, the project proposes a novel pipeline that leverages **large language models (LLMs)** and **synthetic data generation** to train a high-performing dataset mention extractor. The core components of the project include:

* **Synthetic Data Creation**: Using LLMs to generate high-quality, weakly supervised training data based on minimal seed examples.
* **Two-Stage Model Training**:
  * **Pre-Fine-Tuning**: Training a compact instruction-tuned model (Phi-3.5-mini) on synthetic examples.
  * **Fine-Tuning**: Refining the model on a smaller, manually annotated dataset to improve precision.
* **Efficient Inference**: Employing a ModernBERT classifier to identify candidate sentences likely to contain dataset mentions, improving inference efficiency.
* **Evaluation and Comparison**: Demonstrating state-of-the-art performance compared to existing tools like NuExtract and GLiNER on real annotated samples.

---
