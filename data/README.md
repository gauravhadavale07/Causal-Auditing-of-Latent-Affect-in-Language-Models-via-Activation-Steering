# Data Description

This folder contains evaluation prompts and derived structural metrics used in the analysis.

## Files

- `prompts.txt`  
  Evaluation prompts used across all experimental conditions.

- `prompt_metadata.csv`  
  Mapping between prompts and experimental conditions (baseline, anger-suppressed, control).

- `structural_metrics.csv`  
  Per-prompt structural discourse metrics computed from generated outputs.
  These values are aggregated to produce the main results table in the paper.

## Notes

- No training data or model weights are included.
- All data was generated through controlled inference-time interventions.
- This data is intended for reproducibility and analysis only.
