import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from torchdr import TNERV as TorchdrTNERV
from torchdr import NERV as TorchdrNERV

'''
def prepare_mnist(max_samples=20_000):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data.astype(np.float32) / 255.0
    target = mnist.target.astype(np.int64)
    
    if max_samples:
        data = data[:max_samples]
        target = target[:max_samples]
    
    data = StandardScaler().fit_transform(data)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)
'''

def prepare_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data.astype(np.float32) / 255.0
    target = mnist.target.astype(np.int64)
    data = StandardScaler().fit_transform(data)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare data
    #data, labels = prepare_mnist(max_samples=20_000)
    data, labels = prepare_mnist()
    data = data.to(device)
    labels_np = labels.cpu().numpy()
    print(f"Data shape: {tuple(data.shape)}")

    # Common parameters for both models
    base_kwargs = {
        "perplexity": 30.0,
        "n_components": 2,
        "lr": "auto",
        "max_iter": 1000,
        "early_exaggeration_coeff": 12.0,
        "early_exaggeration_iter": 250,
        "init": "pca",
        "device": device,
        "backend": "keops",
        "random_state": 0,
        "sparsity": True,
        "metric": "sqeuclidean",
        "verbose": True,
        "compile": False,
    }
    
    lambda_param = 0.5
    
    # Create models
    models = {
        "NeRV": TorchdrNERV(lambda_param=lambda_param, **base_kwargs),
        "t-NeRV": TorchdrTNERV(lambda_param=lambda_param, degrees_of_freedom=1, **base_kwargs),
    }

    # Compute embeddings
    embeddings = {}
    for name, model in models.items():
        t0 = time.time()
        emb = model.fit_transform(data).detach().cpu().numpy()
        elapsed = time.time() - t0
        print(f"{name} completed in {elapsed:.2f}s")
        embeddings[name] = emb

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, (name, emb) in zip(axes, embeddings.items()):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels_np, cmap="tab10", s=1, alpha=0.7)
        ax.set_title(f"{name} Embedding (λ={lambda_param})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    fig.colorbar(sc, ax=axes, label="Digit")
    fig.suptitle("NeRV vs t-NeRV Embeddings of MNIST")
    plt.show()

if __name__ == "__main__":
    main()