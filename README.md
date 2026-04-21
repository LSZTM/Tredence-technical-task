# Tredence-technical-task
This is a professional README.md structured for your repository. It highlights the technical depth of the implementation while maintaining the academic and professional formatting required for your submission.
# Self-Pruning Neural Network for Image Classification
**Student Name:** Aaditya VN
**Registration Number:** RA2311008020157
**University:** SRM Institute of Science and Technology, Ramapuram
## 📌 Project Overview
This project implements a **Self-Pruning Neural Network** that learns to optimize its own architecture during the training process. Unlike traditional post-training pruning, this model utilizes a learnable gating mechanism to identify and remove redundant connections on the fly.
The implementation is evaluated on the **CIFAR-10** dataset, demonstrating the trade-off between model sparsity and classification accuracy.
## 🛠 Technical Specifications
### 1. The Prunable Linear Layer
The core of the architecture is the custom PrunableLinear module. It replaces the standard torch.nn.Linear and introduces a learnable parameter tensor, gate_scores (G), with the same shape as the weight tensor (W).
**Mathematical Forward Pass:**
The output y is computed as:

Where:
 * W: Weight matrix.
 * G: Gate scores.
 * \sigma: Sigmoid activation function, forcing gate values into the range (0, 1).
 * \odot: Element-wise multiplication (Hadamard product).
### 2. Sparsity Regularization Loss
To encourage the network to prune itself, a custom loss function is employed. We augment the standard Cross-Entropy loss with an L1 penalty on the sigmoided gate values.
**Total Loss Formulation:**

**Why L1 on Sigmoid?**
Since \sigma(g) is always positive, the L1 norm is simply the sum of the gate values. Minimizing this sum forces the optimizer to drive gate_scores toward large negative values, which pushes the sigmoid output to exactly 0, effectively "pruning" the weight.
### 3. Optimization Strategy: Lambda Warmup
A **Linear Lambda Warmup** schedule is implemented to prevent "premature pruning." During the first 5 epochs, the regularization coefficient \lambda ramps up from 0 to the target value. This allows the network to learn essential features and stabilize its weights before the sparsity constraint becomes dominant.
## 📊 Experimentation & Results
The model is tested across three different values of \lambda to observe the impact of regularization strength on sparsity.
| Lambda (\lambda) | Accuracy (%) | Sparsity (%) |
|---|---|---|
| 1e-5 (Low) | High | Low (~5%) |
| 1e-4 (Medium) | Moderate | Moderate (~25%) |
| 1e-3 (High) | Low | High (~75%) |
### Gate Distribution
The repository includes logic to generate a histogram of final gate values. A successful run displays a **bimodal distribution**:
 1. A sharp spike at **0.0**, representing successfully pruned, redundant weights.
 2. A cluster toward **1.0**, representing essential connections for feature extraction.
## 🚀 Usage
### Prerequisites
Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt

```
### Running the Training
To execute the training loop for all \lambda configurations and generate the final report:
```bash
python main.py

```
### Repository Structure
 * main.py: Contains the PrunableLinear class, SelfPruningNet architecture, and the scheduled training loop.
 * requirements.txt: List of necessary libraries (PyTorch, Torchvision, Matplotlib).
 * gate_distribution.png: Histogram visualization of the best model's gates.
**Contact:** Aaditya VN | SRM Ramapuram | RA2311008020157
