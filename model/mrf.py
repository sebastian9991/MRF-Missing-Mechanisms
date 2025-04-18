import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# -- Configuration --
batch_size = 100
num_epochs = 500


# Simulate coin toss data (X: 0=heads, 1=tails)
def generate_coin_data(batch_size, p_heads=0.5, mnar=True):
    X = torch.bernoulli(torch.full((batch_size,), 1 - p_heads)).long()  # 0 = H, 1 = T
    if mnar:
        prob_observe = torch.where(X == 0, 0.9, 0.4)  # heads more likely to be observed
    else:
        prob_observe = torch.full_like(X, 0.7)
    O = torch.bernoulli(prob_observe).long()  # 1 = observed, 0 = missing
    Y = torch.where(O == 1, X, -1)  # -1 indicates missing
    return X, O, Y


# Define generic factor-based model
class CoinMRF(nn.Module):
    def __init__(self, model_type="MNAR"):
        super().__init__()
        self.model_type = model_type
        self.theta = nn.Parameter(torch.tensor(0.0))  # coin bias
        self.psi = nn.Parameter(torch.tensor(0.0))  # missingness factor

    def forward(self, Y):
        log_probs = []
        for y in Y:
            if y != -1:
                log_p_x = [self._log_joint(x, 1) for x in [0, 1]]
                log_p = log_p_x[y]
            else:
                log_p_x = torch.tensor([self._log_joint(x, 0) for x in [0, 1]])
                log_p = torch.logsumexp(log_p_x, dim=0)
            log_probs.append(log_p)
        return -torch.stack(log_probs).mean()

    def _log_joint(self, x, o):
        log_phi1 = self.theta * x
        if self.model_type == "MNAR":
            log_phi2 = self.psi * x * o  # ψ * x * o → MNAR
        elif self.model_type == "MCAR":
            log_phi2 = self.psi * o  # ψ * o → MCAR
        else:
            log_phi2 = 0.0
        return log_phi1 + log_phi2


# Train both models and track losses
def train_models(Y, epochs):
    models = {"MCAR": CoinMRF(model_type="MCAR"), "MNAR": CoinMRF(model_type="MNAR")}
    losses = {"MCAR": [], "MNAR": []}

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for epoch in range(epochs):
            loss = model(Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[name].append(loss.item())
    return models, losses


# Plot loss curves
def plot_losses(losses):
    for key, values in losses.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MCAR vs MNAR Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main
if __name__ == "__main__":
    torch.manual_seed(42)
    X, O, Y = generate_coin_data(
        batch_size=batch_size, mnar=True
    )  # Change mnar=True/False for test
    models, losses = train_models(Y, epochs=num_epochs)

    print("\nLearned Parameters:")
    for name, model in models.items():
        print(f"{name}: θ = {model.theta.item():.4f}, ψ = {model.psi.item():.4f}")

    plot_losses(losses)
