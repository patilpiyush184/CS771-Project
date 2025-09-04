# CS771 Course Project..
## Mini Project 1 – Binary Classification with Multiple Feature Representations

This mini-project involves training and evaluating binary classification models on **three different datasets**, all derived from the same raw data but represented using different feature sets.  

The project is divided into two tasks:  
1. **Task 1:** Identify the best classification model for each dataset individually.  
2. **Task 2:** Explore whether combining all three datasets can yield a better-performing model.  

---

### **Task 1: Individual Dataset Models**
- Train binary classification models of your choice on each of the 3 datasets:
  1. **Emoticons as Features Dataset**
  2. **Deep Features Dataset**
  3. **Text Sequence Dataset**
- Evaluate models based on:
  - **Accuracy** on validation/test sets
  - **Data efficiency** (performance with smaller training subsets)
- Experiment with different training set sizes:
  - {20%, 40%, 60%, 80%, 100%}
- Plot **validation accuracy vs % of training data** for each dataset.
- Select the **best model per dataset** and generate predictions on the corresponding **test sets**.

---

### **Task 2: Combined Dataset Model**
- Investigate whether combining the 3 datasets improves model performance.  
- Explore strategies for combining feature representations (e.g., concatenation, embedding-based transformations, ensemble learning, etc.).  
- Again, vary training sizes {20%, 40%, 60%, 80%, 100%} and plot **validation accuracy vs training set size**.  
- Submit predictions on the combined dataset **test set**.

---

## Dataset Description

### 1. **Emoticons as Features Dataset**
- Format: CSV  
- Features: 13 categorical emoticons  
- Label: Binary {0, 1}  
- Example:  input_emoticons: 😀😂😢..., label: 0
- Shape: `13 features`

---

### 2. **Deep Features Dataset**
- Format: `.npz`  
- Features: `13 × 786` matrix (each emoticon represented as a 786-dim embedding)  
- Label: Binary {0, 1}  
- Example:  features: [[0.1, -0.3, 0.5, ...], ...], label: 0

- Shape: `13 × 786`

---

### 3. **Text Sequence Dataset**
- Format: CSV  
- Features: string of 50 digits  
- Label: Binary {0, 1}  
- Example:  input_str: "0241812141226549266461...", label: 0


## Data Preprocessing

### Emoticons Dataset
- Extracted emoticons from input strings.  
- Mapped each unique emoticon to an integer index.  

### Deep Features Dataset
- Flattened each `13 × 768` embedding matrix into a vector of size `9984`.  
- Applied **StandardScaler** for normalization.  

### Text Sequence Dataset
- Converted 50-character strings into sequences of integers.  
- Tokenized into numerical arrays suitable for neural networks.  

### Combined Dataset
- **Emoticons:** One-hot encoded.  
- **Deep Features:** Flattened + scaled.  
- **Text Sequences:** Tokenized + one-hot encoded.  
- Concatenated all features → applied **StandardScaler** for uniform scaling.  

---

## Model Development & Evaluation

### 1. Emoticons Dataset – Custom RNN Model
- **Embedding Layer:** size 8  
- **Simple RNN Layer:** 16 units  
- **LSTM Layer:** 32 units  
- **Dense Output Layer:** sigmoid activation  
- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Binary Cross-Entropy  
- **Trainable Parameters:** 8,425  
- **Training:** 20 epochs, batch size = 32  

**Validation Accuracy vs Training Size**  
- 20% → 86.91%  
- 40% → 93.05%  
- 60% → 94.27%  
- 80% → 95.91%  
- 100% → **96.73%**  

---

### 2. Deep Features Dataset – Support Vector Classifier (SVC)
- **Input:** Flattened + scaled `13 × 768` embeddings  
- **Model:** SVC with RBF kernel  
- **Evaluation Metric:** Accuracy  

**Validation Accuracy vs Training Size**  
- 20% → 96.93%  
- 40% → 97.55%  
- 60% → 97.96%  
- 80% → 98.98%  
- 100% → **98.77%**  

---

### 3. Text Sequence Dataset – CNN + GRU Model
- **Embedding Layer:** size 8  
- **Conv1D Layer:** 32 filters, kernel size = 5  
- **GRU Layer:** 32 units  
- **Dense Layer:** 16 units  
- **Output Layer:** sigmoid activation  
- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Binary Cross-Entropy  
- **Trainable Parameters:** 8,273  
- **Training:** 20 epochs, batch size = 16  

**Validation Accuracy vs Training Size**  
- 20% → 68.71%  
- 40% → 79.35%  
- 60% → 79.35%  
- 80% → 81.60%  
- 100% → **85.28%**  

---

### 4. Combined Dataset – Unified Model
- **Feature Combination:** Concatenation of one-hot emoticons, flattened/scaled deep features, and one-hot text sequences.  
- **Scaling:** Applied StandardScaler after concatenation.  
- **Model:** Best-performing classifier trained on combined features.  

**Validation Accuracy vs Training Size**  
- 20% → 96.32%  
- 40% → 97.75%  
- 60% → 98.57%  
- 80% → 98.16%  
- 100% → **98.98%**  

---

## Results Summary

| Dataset                | Best Model            | Peak Validation Accuracy |
|------------------------|----------------------|--------------------------|
| Emoticons Dataset      | RNN + LSTM           | **96.73%**              |
| Deep Features Dataset  | SVC (RBF kernel)     | **98.77%**              |
| Text Sequence Dataset  | CNN + GRU            | **85.28%**              |
| Combined Dataset       | Unified Classifier   | **98.98%**              |

---

## Mini Project 2 – Continual Learning on CIFAR-10 Subsets

## Problem Statement
We are provided with **20 training datasets** `D1, D2, …, D20`, each of which is a subset of the CIFAR-10 image classification dataset.

- **Datasets D1–D10**: Inputs come from the **same distribution** `p(x)`.
- **Datasets D11–D20**: Inputs come from **different distributions**, though they are still related to the first distribution.

We are also provided with **20 held-out labeled datasets** `Ď1, Ď2, …, Ď20` that are **only for evaluation**.

Only **D1 is labeled**, while the rest (`D2–D20`) are **unlabeled**.  
The task is to **sequentially train models f1, f2, …, f20** using an **LwP classifier** (Learning with Pseudo-labels), ensuring that performance does not degrade drastically on previous datasets.

---
### **Task 1: D1–D10 (same distribution)**
1. Train `f1` on **D1 (labeled)**.  
2. Use `f1` to pseudo-label **D2**, then update to `f2`.  
3. Repeat sequentially until `f10`.  
4. Evaluate each model `fi` (1 ≤ i ≤ 10) on:
   - its corresponding held-out dataset `Di`
   - all previous held-out datasets `Dj, j < i`.  
   - Results reported in a **10×10 accuracy matrix**.

Goal: Maintain accuracy on earlier datasets while adapting to new ones.

---

### **Task 2: D11–D20 (different distributions)**
1. Start with `f10` from Task 1.  
2. Sequentially adapt to datasets `D11, D12, …, D20` → models `f11 … f20`.  
3. Evaluate each model `fi (11 ≤ i ≤ 20)` on:
   - its corresponding held-out dataset `Di`
   - all previous held-out datasets `Dj, j < i`.  
   - Results reported in a **10×20 accuracy matrix**.

Challenge: Prevent **catastrophic forgetting** while adapting across **shifting distributions**.

---
## Task 1.1: Semi-Supervised Continual Learning on CIFAR-10
### Approach
1. **Feature Extraction**
   - Images resized from `32×32` → `224×224` for ViT.
   - Pretrained model: `ViT-base-patch16-224-in21k` (from Hugging Face).
   - Extract features from the `[CLS]` token.

2. **Prototype Computation**
   - Prototypes initialized as the **mean feature vector per class** (from labeled D1).
   - For later datasets, pseudo-labels are used for prototype updates.

3. **Pseudo-Labeling**
   - Compute distances between features and prototypes.
   - Assign pseudo-labels based on nearest prototype.
   - Keep samples with **confidence ≥ 0.8**.

4. **Prototype Updates**
    ```
   Prototypes_new = α * Prototypes_old + (1 - α) * Prototypes_pseudo
   ```

- Momentum factor: `α = 0.9`.

5. **Evaluation**
- Classify test samples using nearest prototype.
- Compute accuracy against ground-truth labels.

---

## Task 1.2: Continual Learning with Distributional Shifts (D11–D20)

### Approach
1. **Prototype Initialization**
- Start with prototypes from **f10** (Task 1.1 final model).

2. **Feature Extraction**
- Continue using ViT (`ViT-base-patch16-224-in21k`).
- Extract `[CLS]` token features.

3. **Adjusted Pseudo-Labeling**
- Assign pseudo-labels based on nearest prototype.
- Use conservative confidence threshold (≥ 0.8).

4. **Momentum-Based Updates**
- Same update rule:
  ```
  Prototypes_new = α * Prototypes_old + (1 - α) * Prototypes_pseudo
  ```
- Momentum factor `α = 0.9` ensures stability against distributional shifts.

5. **Evaluation**
- Each model fi (i = 11...20) is evaluated on all **D1–Di** held-out datasets.
- Accuracy computed via nearest-prototype classification.

---
