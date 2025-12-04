import time

from datasetloader.datasetloader import DatasetLoader
from torchdr.neighbor_embedding.t_nerv import TNERV


def main():
    X, _ = DatasetLoader.load_mnist()

    model = TNERV(n_components=2, perplexity=50.0, device="cuda", backend="keops")

    t0 = time.perf_counter()
    embedding = model.fit_transform(X)
    runtime = time.perf_counter() - t0

    print(f"X shape: {X.shape}, embedding shape: {embedding.shape}, runtime: {runtime:.3f}s")
    return X.shape, embedding.shape, runtime


if __name__ == "__main__":
    main()