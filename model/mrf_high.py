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
        # Strong MNAR signal: heads are 95% observed, tails 20%
        prob_observe = torch.where(X == 0, torch.tensor(0.95), torch.tensor(0.20))
    else:
        prob_observe = torch.full_like(X, 0.7, dtype=torch.float)
    O = torch.bernoulli(prob_observe).long()  # 1 = observed, 0 = missing
    Y = torch.where(O == 1, X, -1)  # -1 indicates missing
    return X, O, Y


# Define multivariate factor-based model
class CoinMRF(nn.Module):
    def __init__(self, model_type="MNAR", num_features=5):
        super().__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.theta = nn.Parameter(torch.zeros(num_features))  # per-feature coin bias
        self.psi = nn.Parameter(
            torch.zeros(num_features)
        )  # per-feature missingness factor

    def forward(self, Y):
        log_probs = []
        for y in Y:
            log_sample = 0.0
            for i in range(self.num_features):
                if y[i] != -1:
                    log_p_x = [self._log_joint(i, x, 1) for x in [0, 1]]
                    log_p = log_p_x[y[i]]
                else:
                    log_p_x = torch.tensor([self._log_joint(i, x, 0) for x in [0, 1]])
                    log_p = torch.logsumexp(log_p_x, dim=0)
                log_sample += log_p
            log_probs.append(log_sample)
        return -torch.stack(log_probs).mean()

    def _log_joint(self, i, x, o):
        log_phi1 = self.theta[i] * x
        if self.model_type == "MNAR":
            log_phi2 = self.psi[i] * x * o
        elif self.model_type == "MCAR":
            log_phi2 = self.psi[i] * o
        else:
            log_phi2 = 0.0
        return log_phi1 + log_phi2


# Train both models and track losses
def train_models(Y, epochs, num_features):
    models = {
        "MCAR": CoinMRF(model_type="MCAR", num_features=num_features),
        "MNAR": CoinMRF(model_type="MNAR", num_features=num_features),
    }
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
def plot_losses(losses_dict):
    plt.figure(figsize=(10, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MCAR vs MNAR Loss on MCAR and MNAR Data (Multivariate)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    import torch

import matplotlib.pyplot as plt
import torch.nn as nn

# -- Configuration --
batch_size = 1000
num_epochs = 1000
num_features = 5  # multiple coin tosses per sample

# Simulate multivariate coin toss data (X: 0=heads, 1=tails)
def generate_coin_data(batch_size, num_features, p_heads=0.5, mnar=True):
    X = torch.bernoulli(torch.full((batch_size, num_features), 1 - p_heads)).long()  # 0 = H, 1 = T
    if mnar:
        # Strong MNAR signal: heads are 95% observed, tails 20%
        prob_observe = torch.where(X == 0, torch.tensor(0.95), torch.tensor(0.20))
    else:
        prob_observe = torch.full_like(X, 0.7, dtype=torch.float)
    O = torch.bernoulli(prob_observe).long()  # 1 = observed, 0 = missing
    Y = torch.where(O == 1, X, -1)  # -1 indicates missing
    return X, O, Y

# Define multivariate factor-based model
class CoinMRF(nn.Module):
    def __init__(self, model_type='MNAR', num_features=5):
        super().__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.theta = nn.Parameter(torch.zeros(num_features))  # per-feature coin bias
        self.psi = nn.Parameter(torch.zeros(num_features))    # per-feature missingness factor

    def forward(self, Y):
        log_probs = []
        for y in Y:
            log_sample = 0.0
            for i in range(self.num_features):
                if y[i] != -1:
                    log_p_x = [self._log_joint(i, x, 1) for x in [0, 1]]
                    log_p = log_p_x[y[i]]
                else:
                    log_p_x = torch.tensor([self._log_joint(i, x, 0) for x in [0, 1]])
                    log_p = torch.logsumexp(log_p_x, dim=0)
                log_sample += log_p
            log_probs.append(log_sample)
        return -torch.stack(log_probs).mean()

    def _log_joint(self, i, x, o):
        log_phi1 = self.theta[i] * x
        if self.model_type == 'MNAR':
            log_phi2 = self.psi[i] * x * o
        elif self.model_type == 'MCAR':
            log_phi2 = self.psi[i] * o
        else:
            log_phi2 = 0.0
        return log_phi1 + log_phi2

# Train both models and track losses
def train_models(Y, epochs, num_features):
    models = {
        'MCAR': CoinMRF(model_type='MCAR', num_features=num_features),
        'MNAR': CoinMRF(model_type='MNAR', num_features=num_features)
    }
    losses = {'MCAR': [], 'MNAR': []}

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
def plot_losses(losses_dict):
    plt.figure(figsize=(10, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("MCAR vs MNAR Loss on MCAR and MNAR Data (Multivariate)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main
if __name__ == "__main__":
    torch.manual_seed(42)
    losses_all = {}
    models_all = {}

    for data_type in ['MCAR', 'MNAR']:
        print(f"\n=== Training on {data_type} Data ===")
        mnar_flag = True if data_type == 'MNAR' else False
        X, O, Y = generate_coin_data(batch_size=batch_size, num_features=num_features, mnar=mnar_flag)

        models, losses = train_models(Y, epochs=num_epochs, num_features=num_features)
        for model_type in models:
            label = f"{model_type} on {data_type} data"
            losses_all[label] = losses[model_type]
            models_all[label] = models[model_type]

    for label, model in models_all.items():
        print(f"{label}: θ = {model.theta.data.tolist()}, ψ = {model.psi.data.tolist()}")

    plot_losses(losses_all)

# Main
if __name__ == "__main__":
    main()
