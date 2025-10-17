import time
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from torchdr import TNERV as TorchdrTNERV
from torchdr.distance import FaissConfig
import numpy as np

def _prepare_mnist(max_samples: int | None = None) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data.astype(np.float32) / 255.0
    target = mnist.target.astype(np.int64)
    if max_samples is not None:
        data = data[:max_samples]
        target = target[:max_samples]
    data = StandardScaler().fit_transform(data)
    features = torch.tensor(data, dtype=torch.float32)
    labels   = torch.tensor(target, dtype=torch.long)
    return features, labels, data

def _compute_embedding(model, data: torch.Tensor) -> np.ndarray:
    emb = model.fit_transform(data)
    return emb.detach().cpu().numpy()

def main() -> None:
    torch.manual_seed(0)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Diag] torch.cuda.is_available(): {torch.cuda.is_available()}, device: {device_str}")

    # ---- data ----
    data, labels, _ = _prepare_mnist(max_samples=20_000)  # bump to 70_000 once KeOps GPU confirmed
    data = data.to(device_str, non_blocking=True)        # << ensure data is on GPU
    print(f"[Diag] data.device={data.device}, shape={tuple(data.shape)}")

    # ---- backends ----
    # Use KeOps for the EMBEDDING (repulsive term) -> massive speedup on GPU.
    # If your TNERV implementation forwards the same backend to EntropicAffinity,
    # that will run kNN via KeOps (fine). If you specifically want FAISS-GPU for kNN,
    # some TorchDR variants let you pass a FaissConfig instead of "keops". If yours
    # does not, prefer speed on repulsion and keep backend="keops".
    # If your TNERV *does* accept the FaissConfig directly and still uses KeOps for
    # pairwise, switch the backend below to `FaissConfig(...)`.
    backend_for_embedding = "keops"

    # (Optional) If your TNERV supports FaissConfig directly:
    # backend_for_embedding = FaissConfig(index_type="IVF", nlist=4096, nprobe=32, use_float16=True)

    common_kwargs = dict(
        perplexity=50.0,
        n_components=2,
        lr="auto",
        max_iter=1000,                   # start smaller; raise to 1000 once fast path is confirmed
        early_exaggeration_coeff=8.0,
        early_exaggeration_iter=250,
        init="pca",
        device=device_str,
        backend=backend_for_embedding,     
        degrees_of_freedom=1,
        random_state=0,
        sparsity=True,
        metric="sqeuclidean",
        verbose=True,
        compile=False,
    )

    model = TorchdrTNERV(lambda_param=0.5, **common_kwargs)

    t0 = time.time()
    emb = _compute_embedding(model, data)
    print(f"[Timing] fit_transform: {time.time() - t0:.2f}s")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=1, alpha=0.7)
    plt.colorbar(sc, label="Digit")
    plt.title("t-NeRV Embedding of MNIST (KeOps repulsion)")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()