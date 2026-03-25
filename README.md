# MrM-Core-Idea-Validation

Validating the core idea of [MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems]([paper link](https://arxiv.org/abs/2506.07399)) (AAAI 2026).

## Core Idea

By masking objects in images and querying a multimodal model:
- **Member images** (in knowledge base) → model correctly identifies the masked object
- **Non-member images** (not in knowledge base) → model fails to identify

## Results

| Image Type | Accuracy (3 trials) |
|------------|---------------------|
| Member | 100% (3/3) |
| Non-member | 0% (0/3) |

## How to Run

```bash
python mrm_simple.py
