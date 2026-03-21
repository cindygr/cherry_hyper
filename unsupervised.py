# pytorch_unsupervised_200.py
import math
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------------------
# 2) Simple Autoencoder to learn embeddings
# ------------------------------------------------------------------------------
class AE(nn.Module):
    def __init__(self, in_dim=200, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ------------------------------------------------------------------------------
# 4) K-means in PyTorch (no sklearn dependency)
# ------------------------------------------------------------------------------
def kmeans_torch(X, k, num_iters=50, verbose=False):
    """
    X: [N, d] float tensor
    Returns: (assignments [N], centers [k, d])
    """
    N, d = X.shape
    # Initialize centers with k random points
    indices = torch.randperm(N, device=X.device)[:k]
    centers = X[indices].clone()

    for it in range(num_iters):
        # Compute distances to centers: [N, k]
        # Use (x - c)^2 = x^2 + c^2 - 2 x·c to avoid big expansions
        x2 = (X**2).sum(dim=1, keepdim=True)          # [N, 1]
        c2 = (centers**2).sum(dim=1).unsqueeze(0)     # [1, k]
        dist = x2 + c2 - 2 * (X @ centers.T)          # [N, k]

        # Assign to nearest center
        assignments = dist.argmin(dim=1)              # [N]

        # Recompute centers
        new_centers = torch.zeros_like(centers)
        for j in range(k):
            mask = (assignments == j)
            if mask.any():
                new_centers[j] = X[mask].mean(dim=0)
            else:
                # If a cluster gets no points, reinit to a random point
                new_centers[j] = X[torch.randint(0, N, (1,), device=X.device)]

        shift = (centers - new_centers).pow(2).sum().sqrt().item()
        centers = new_centers
        if verbose:
            print(f"iter {it+1:02d}/{num_iters}  center_shift={shift:.4f}")
        # Optional: early stop
        if shift < 1e-4:
            break

    return assignments, centers


# Rebuild the same encoder architecture
class Encoder(nn.Module):
    def __init__(self, in_dim=200, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def assign_to_clusters(new_data: torch.Tensor,
                       artifacts_path: str = "ae_kmeans_artifacts.pt",
                       device: torch.device = None,
                       batch_size: int = 1024) -> torch.Tensor:
    """
    Assign new samples to learned clusters.

    Args:
        new_data: Float tensor of shape [N_new, D] with the SAME feature layout used during training.
        artifacts_path: Path to saved encoder/centers.
        device: torch.device, if None chooses CUDA if available else CPU.
        batch_size: encode in batches to avoid OOM.

    Returns:
        assignments: Long tensor of shape [N_new] with cluster IDs in [0, K-1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(artifacts_path, map_location=device)
    D = ckpt["in_dim"]
    latent_dim = ckpt["latent_dim"]
    centers = ckpt["centers"].to(device)  # [K, latent_dim]
    K = centers.shape[0]

    # Rebuild and load encoder
    encoder = Encoder(in_dim=D, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    # Ensure data dtype/shape
    assert new_data.dim() == 2 and new_data.shape[1] == D, \
        f"Expected shape [N_new, {D}], got {tuple(new_data.shape)}"
    new_data = new_data.to(device).float()

    # Encode in batches
    Z_list = []
    for i in range(0, new_data.size(0), batch_size):
        z = encoder(new_data[i:i+batch_size])  # [b, latent_dim]
        Z_list.append(z)
    Z = torch.cat(Z_list, dim=0)               # [N_new, latent_dim]

    # Compute distances to centers: [N_new, K]
    x2 = (Z**2).sum(dim=1, keepdim=True)            # [N_new, 1]
    c2 = (centers**2).sum(dim=1).unsqueeze(0)       # [1, K]
    dist = x2 + c2 - 2 * (Z @ centers.T)            # [N_new, K]

    assignments = dist.argmin(dim=1)                # [N_new]
    return assignments


@torch.no_grad()
def assign_with_scores(new_data, artifacts_path="ae_kmeans_artifacts.pt", temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(artifacts_path, map_location=device)
    D = ckpt["in_dim"]; latent_dim = ckpt["latent_dim"]
    centers = ckpt["centers"].to(device)

    # Rebuild encoder
    enc = Encoder(in_dim=D, latent_dim=latent_dim).to(device)
    enc.load_state_dict(ckpt["encoder_state"]); enc.eval()

    X = new_data.to(device).float()
    Z = enc(X)

    # distances [N, K]
    x2 = (Z**2).sum(1, keepdim=True)
    c2 = (centers**2).sum(1).unsqueeze(0)
    dist2 = x2 + c2 - 2 * (Z @ centers.T)          # squared distance
    logits = -dist2 / max(1e-8, temperature)       # larger => closer
    probs = torch.softmax(logits, dim=1)           # [N, K]
    hard = probs.argmax(dim=1)
    return hard, probs


if __name__ == '__main__':

    # ------------------------------------------------------------------------------
    # 1) Dummy data: N samples, each with 200 features (replace with your data)
    # ------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: synthetic data (3 clusters) in 200-D
    torch.manual_seed(42)
    N = 1200        # number of samples (can be whatever you have)
    D = 150         # feature dimension (your "size 200")
    K = 3           # number of clusters you expect

    centers = torch.stack([
        torch.randn(D) * 0.5 + 2.0,
        torch.randn(D) * 0.5 - 2.0,
        torch.randn(D) * 0.5
    ])
    data = torch.cat([
        centers[0] + 0.6 * torch.randn(N // 3, D),
        centers[1] + 0.6 * torch.randn(N // 3, D),
        centers[2] + 0.6 * torch.randn(N - 2 * (N // 3), D),
    ], dim=0).to(device)

    # If you have real data:
    # data = torch.tensor(your_numpy_array, dtype=torch.float32).to(device)  # shape [N, 200]


    latent_dim = 16
    model = AE(in_dim=D, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Tiny DataLoader for simplicity (full-batch works too if it fits)
    batch_size = 256
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Train AE (self-supervised)
    epochs = 20
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:02d}/{epochs}  recon_loss={total/len(dataset):.4f}")

    # ------------------------------------------------------------------------------
    # 3) Get embeddings (latent vectors) for clustering
    # ------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        _, Z = model(data)  # [N, latent_dim]

    assignments, centers = kmeans_torch(Z, K, num_iters=100, verbose=True)

    print("\nCluster counts:")
    for j in range(K):
        print(f"  Cluster {j}: {(assignments == j).sum().item()} samples")

    # ------------------------------------------------------------------------------
    # 5) (Optional) Map clusters to “classes”
    # ------------------------------------------------------------------------------
    # In unsupervised settings, labels are arbitrary. If you later obtain
    # a few labeled examples, you can post-hoc map cluster IDs to classes.


    # --- After k-means finishes in the training script ---

    # Save only the encoder (we don't need decoder for clustering)
    encoder_state = model.encoder.state_dict()
    torch.save({
        "encoder_state": encoder_state,
        "latent_dim": latent_dim,
        "in_dim": D,
        "centers": centers.detach().cpu(),   # [K, latent_dim]
    }, "ae_kmeans_artifacts.pt")

    print("Saved encoder and centers to ae_kmeans_artifacts.pt")


    # Just another example Suppose you have 5 new samples, 200-D each
    X_new = torch.randn(5, 200)  # Replace with your real data

    assignments = assign_to_clusters(X_new, "ae_kmeans_artifacts.pt")
    print("Assigned cluster IDs:", assignments.tolist())
