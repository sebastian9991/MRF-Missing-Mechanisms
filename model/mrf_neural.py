import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -- Configuration --
batch_size = 1000
num_epochs = 1000
num_features = 5  # multiple coin tosses per sample


# Simulate multivariate coin toss data (X: 0=heads, 1=tails)
def generate_coin_data(batch_size, num_features, p_heads=0.5, mnar=True):
    X = torch.bernoulli(
        torch.full((batch_size, num_features), 1 - p_heads)
    ).long()  # 0 = H, 1 = T
    if mnar:
        prob_observe = torch.where(X == 0, torch.tensor(0.95), torch.tensor(0.20))
    else:
        prob_observe = torch.full_like(X, 0.7, dtype=torch.float)
    O = torch.bernoulli(prob_observe).long()  # 1 = observed, 0 = missing
    Y = torch.where(O == 1, X, -1)  # -1 indicates missing
    return X, O, Y


# Two-layer neural network factor
class NeuralFactor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return torch.exp(self.net(x)).squeeze()


# Define multivariate factor-based model with NN factors
class CoinMRF(nn.Module):
    def __init__(self, model_type="MNAR", num_features=5):
        super().__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.value_factor = NeuralFactor(input_dim=num_features)
        self.obs_factors = nn.ModuleList(
            [NeuralFactor(input_dim=2) for _ in range(num_features)]
        )

    def forward(self, Y):
        log_probs = []
        for y in Y:
            # Value factor input (treat missing as 0)
            x_fill = torch.where(y == -1, torch.tensor(0, dtype=torch.float), y.float())
            log_phi1 = torch.log(self.value_factor(x_fill))
            log_phi2 = 0.0
            for i in range(self.num_features):
                if y[i] != -1:
                    input_pair = torch.tensor([float(y[i]), 1.0])
                else:
                    input_pair = torch.tensor([0.0, 0.0])
                log_phi2 += torch.log(self.obs_factors[i](input_pair))
            log_probs.append(log_phi1 + log_phi2)
        return -torch.stack(log_probs).mean()


# Train both models and track losses
def train_models(Y, epochs, num_features):
    models = {
        "MCAR": CoinMRF(model_type="MCAR", num_features=num_features),
        "MNAR": CoinMRF(model_type="MNAR", num_features=num_features),
    }
    losses = {"MCAR": [], "MNAR": []}

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            loss = model(Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[name].append(loss.item())
    return models, losses


# Plot loss curves
def plot_losses(losses_dict):
    plt.figure(figsize=(10, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MCAR vs MNAR Loss with Nonlinear Factor Networks")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    torch.manual_seed(42)
    losses_all = {}
    models_all = {}

    for data_type in ["MCAR", "MNAR"]:
        print(f"\n=== Training on {data_type} Data ===")
        mnar_flag = True if data_type == "MNAR" else False
        X, O, Y = generate_coin_data(
            batch_size=batch_size, num_features=num_features, mnar=mnar_flag
        )

        models, losses = train_models(Y, epochs=num_epochs, num_features=num_features)
        for model_type in models:
            label = f"{model_type} on {data_type} data"
            losses_all[label] = losses[model_type]
            models_all[label] = models[model_type]

    plot_losses(losses_all)


# Main
if __name__ == "__main__":
    main()
