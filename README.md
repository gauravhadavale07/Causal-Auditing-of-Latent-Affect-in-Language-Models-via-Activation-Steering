# Causal-Auditing-of-Latent-Affect-in-Language-Models-via-Activation-Steering

This repository contains code and experiments for **causal auditing of affective behavior in large language models** using internal activation-level interventions.

We identify an anger-related direction in the residual stream of a transformer language model and perform **activation steering** during inference to generate *latent counterfactuals*—behavioral changes induced without modifying the input or retraining the model.

## Key Contributions
- Causal intervention on internal representations (activation steering)
- Negative control vectors to validate specificity
- Structural discourse evaluation beyond surface emotion classifiers
- Layer-wise localization of affective representations

## Method Overview
1. Extract residual stream activations using TransformerLens
2. Compute affect-aligned directions via mean-difference analysis
3. Inject steering vectors during inference
4. Evaluate behavioral changes using structural metrics

## Repository Structure
- `src/` – Core implementation
- `notebooks/` – Reproducible experiments
- `data/` – Prompts and evaluation metrics
- `paper/` – Research paper and references

## Disclaimer
This project is intended for **mechanistic analysis and safety auditing**, not deployment-time behavior control.

## Author
Gaurav Hadavale
