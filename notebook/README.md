# Latent Counterfactuals: Activation Steering Notebook

**File:** `Project_Notebook_Annotated.ipynb`
**Model:** GPT-2 Small
**Methodology:** Mechanistic Interpretability (Activation Steering)
**Library:** TransformerLens

## Overview
This notebook implements a standalone pipeline for auditing the internal emotional reasoning of GPT-2. It utilizes the `TransformerLens` library to extract latent concept vectors from the residual stream and performs causal interventions ("steering") during inference to modulate text generation behavior.

The code demonstrates three core phenomena:
1.  **Functional Localization:** Identifying Layer 8 as the primary locus of affective semantic formation.
2.  **Causality:** Establishing a direct causal link between internal activations and output sentiment.
3.  **Control Asymmetry:** Highlighting the difficulty of suppressing toxicity via linear vector subtraction compared to inducing it.

## Dependencies
The notebook is self-contained and installs necessary libraries in the first execution cell.

* **Python 3.8+**
* **TransformerLens:** For hooking into model internals.
* **PyTorch:** For tensor operations.
* **Transformers (HuggingFace):** For model loading and tokenization.
* **NumPy & Matplotlib:** For data manipulation and visualization.

## Notebook Structure

### 1. Environment Setup
Installs `transformer_lens` and imports required modules. Configures the device (CUDA/CPU).

### 2. Data Definition
Defines the contrastive datasets used to isolate the steering vector:
* `angry_sentences`: A list of prompts exhibiting hostility or frustration.
* `neutral_sentences`: A list of prompts exhibiting factual, mundane descriptions.

### 3. Activation Extraction
* **Function:** `get_residual_activations()`
* **Logic:** Runs a forward pass on the input data and caches the residual stream activation at the final token position.
* **Target:** Defaults to Layer 8, based on preliminary layer-wise sweeps.

### 4. Vector Computation
Calculates the steering direction `theta` using the Difference-in-Means method:
`theta = mean(angry_activations) - mean(neutral_activations)`

The vector is normalized to unit length to ensure that the steering coefficient (`alpha`) represents a consistent magnitude of intervention.

### 5. Intervention Hook
* **Function:** `steering_hook()`
* **Logic:** Injects the steering vector into the residual stream during the forward pass.
    `activation += alpha * steering_vector`

### 6. Generation & Evaluation
Iterates through a range of `alpha` values to demonstrate the dose-response relationship on a fixed neutral prompt.
* **alpha > 0:** Induces anger (Demonstrates causality).
* **alpha < 0:** Attempts suppression (Demonstrates asymmetry/safety limitations).

## Usage Instructions

### Running on Google Colab (Recommended)
1.  Upload `Project_Notebook_Annotated.ipynb` to Google Drive.
2.  Open the file with Google Colab.
3.  Change Runtime Type to **T4 GPU** for faster execution.
4.  Execute all cells sequentially.

### Running Locally
1.  Ensure you have a virtual environment with PyTorch installed.
2.  Install dependencies: `pip install transformer_lens transformers numpy matplotlib`
3.  Launch Jupyter Lab/Notebook: `jupyter notebook Project_Notebook_Annotated.ipynb`

## Configuration
Key variables available for modification:

* `layer_idx`: The Transformer block index where the hook is applied (Default: 8). Changing this allows for layer-wise sensitivity analysis.
* `alpha`: The steering strength coefficient.
    * Range `[0.0, 5.0]`: Standard induction range.
    * Range `[-5.0, 0.0]`: Standard suppression range.
    * Range `> 50.0`: Tests for semantic collapse/entanglement.

## License
This code is provided for research and educational purposes.
