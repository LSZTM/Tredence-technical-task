import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import time
import os

# ==========================================
# PART 1: THE "PRUNABLE" LINEAR LAYER
# ==========================================
class PrunableLinear(nn.Module):
    """
    Custom linear layer that learns to prune itself.
    Uses a gate_scores parameter to mask weights during the forward pass.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Pruning gate parameters (same shape as weights)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Bias initialization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate_scores to 0.0, making sigmoid(0) = 0.5 initially
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, input):
        # Apply Sigmoid to turn scores into gates [0, 1]
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication to prune weights
        pruned_weights = self.weight * gates
        
        # Standard linear operation with pruned weights
        return F.linear(input, pruned_weights, self.bias)

# ==========================================
# PART 2: THE NETWORK ARCHITECTURE
# ==========================================
class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        # Input: 32x32x3 = 3072 features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# PART 3: UTILITY FUNCTIONS
# ==========================================
def calculate_sparsity_loss(model):
    """Calculates L1 penalty on the sigmoided gates across all layers."""
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(torch.abs(gates))
    return sparsity_loss

def get_sparsity_level(model, threshold=1e-2):
    """Calculates % of weights effectively pruned (gate < threshold)."""
    pruned_count = 0
    total_count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned_count += (gates < threshold).sum().item()
                total_count += gates.numel()
    return (pruned_count / total_count * 100) if total_count > 0 else 0.0

def plot_gate_histogram(model, lambda_val, filepath="gate_distribution.png"):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates.tolist())
                
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(f'Gate Value Distribution (λ = {lambda_val})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filepath)
    print(f"Histogram saved to {filepath}")
    plt.close()

# ==========================================
# PART 4: MAIN TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # 1. Data Prep
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    os.makedirs('./data', exist_ok=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    lambdas = [1e-5, 1e-4, 1e-3]
    warmup_epochs = 5
    total_epochs = 10
    results = []
    
    best_overall_acc = 0
    best_lambda = 0
    best_model_ref = None

    # 2. Experimentation
    for lambda_val in lambdas:
        print(f"\n--- Testing Lambda: {lambda_val} ---")
        model = SelfPruningNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(total_epochs):
            model.train()
            
            # --- Linear Lambda Warmup ---
            current_lambda = lambda_val * min(1.0, (epoch + 1) / warmup_epochs)
            
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Combined Loss
                cls_loss = criterion(outputs, labels)
                sparse_loss = calculate_sparsity_loss(model)
                loss = cls_loss + current_lambda * sparse_loss
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{total_epochs} | λ: {current_lambda:.2e} | Loss: {running_loss/len(trainloader):.4f}")

        # 3. Final Evaluation for this Lambda
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        sparsity = get_sparsity_level(model)
        results.append({"lambda": lambda_val, "accuracy": acc, "sparsity": sparsity})
        print(f"Final -> Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_lambda = lambda_val
            best_model_ref = model

    # 4. Summary & Plotting
    print("\n" + "="*35)
    print(f"{'Lambda':<10} | {'Accuracy':<10} | {'Sparsity (%)':<12}")
    print("-" * 35)
    for res in results:
        print(f"{res['lambda']:<10.0e} | {res['accuracy']:<10.2f} | {res['sparsity']:<12.2f}")
    
    if best_model_ref:
        plot_gate_histogram(best_model_ref, best_lambda)

if __name__ == '__main__':
    main()
  
