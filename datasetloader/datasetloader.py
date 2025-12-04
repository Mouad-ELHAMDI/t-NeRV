import gzip
import pickle
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

try:
    import requests
except ImportError:
    requests = None


def _resolve(p: Optional[str | Path]) -> Optional[Path]:
    return Path(p).expanduser().resolve() if p is not None else None


def _existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _maybe_find_file(folder: Path, patterns: list[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    for entry in folder.iterdir():
        if entry.is_file() and any(re.fullmatch(pat, entry.name) for pat in patterns):
            return entry
    return None


def _ensure_str_array(a: np.ndarray) -> np.ndarray:
    if a.ndim > 1:
        a = a.ravel()
    if a.dtype.kind in ("U", "S", "O"):
        return a.astype(str)
    return np.array([str(x) for x in a], dtype=str)


def load_tasic_preprocessed(
    data_dir: str | Path,
    pca_path: Optional[str | Path] = None,
    ttypes_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    data_dir = _resolve(data_dir)

    if pca_path is None:
        pca_path = _maybe_find_file(
            data_dir, [r"tasic[_-]?pca50\.npy"]
        ) or (data_dir / "tasic-pca50.npy")

    if ttypes_path is None:
        ttypes_path = _maybe_find_file(
            data_dir, [r"tasic[_-]?t?types\.npy", r"tasic[_-]?ttypes\.npy"]
        ) or (data_dir / "tasic-ttypes.npy")

    pca_path = _existing(_resolve(pca_path))
    ttypes_path = _existing(_resolve(ttypes_path))

    X = np.load(pca_path).astype(np.float32, copy=False)
    cell_types = _ensure_str_array(np.load(ttypes_path, allow_pickle=True))

    if X.ndim != 2:
        raise ValueError(f"Expected 2D PCA matrix, got shape {X.shape}")
    if cell_types.shape[0] != X.shape[0]:
        raise ValueError(
            f"Length mismatch: ttypes has {cell_types.shape[0]} entries but X has {X.shape[0]} rows"
        )

    obs = pd.DataFrame(
        {"cell_id": np.arange(X.shape[0], dtype=int), "cell_type": cell_types}
    ).set_index("cell_id")

    return {"X": X, "obs": obs}


class DatasetLoader:
    @staticmethod
    def load_openml_dataset(
        dataset_name: str, version: Optional[int] = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a dataset from OpenML."""
        data = fetch_openml(name=dataset_name, version=version, as_frame=False)
        X = data["data"].astype("float32")
        y_raw = data["target"]
        if y_raw.dtype.kind in ("U", "S", "O"):
            y = LabelEncoder().fit_transform(y_raw.astype(str))
        else:
            y = y_raw.astype("int64", copy=False)
        return X, y

    @staticmethod
    def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
        """Load MNIST (first 60,000 samples)."""
        X, y = DatasetLoader.load_openml_dataset("mnist_784")
        return X[:60000], y[:60000]

    @staticmethod
    def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
        """Load Fashion-MNIST (first 60,000 samples)."""
        X, y = DatasetLoader.load_openml_dataset("Fashion-MNIST")
        return X[:60000], y[:60000]

    @staticmethod
    def load_scanpy_benchmarks() -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load Macosko 2015 and 10x mouse Zheng benchmarks."""
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for this function. "
                "Install with: pip install requests"
            )

        url_macosko = "http://file.biolab.si/opentsne/benchmark/macosko_2015.pkl.gz"
        url_10x = "http://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz"

        def download_and_load(url: str) -> dict[str, Any]:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with gzip.open(BytesIO(resp.content), "rb") as f:
                return pickle.load(f)

        data_macosko = download_and_load(url_macosko)
        x_macosko = data_macosko["pca_50"].astype("float32")
        y_macosko = LabelEncoder().fit_transform(data_macosko["CellType1"].astype(str))

        data_10x = download_and_load(url_10x)
        x_10x = data_10x["pca_50"].astype("float32")
        y_10x = LabelEncoder().fit_transform(data_10x["CellType1"].astype(str))

        return {
            "macosko": (x_macosko, y_macosko),
            "10x_zheng": (x_10x, y_10x),
        }

    @staticmethod
    def load_macosko() -> Tuple[np.ndarray, np.ndarray]:
        """Load Macosko 2015 single-cell dataset."""
        benchmarks = DatasetLoader.load_scanpy_benchmarks()
        return benchmarks["macosko"]

    @staticmethod
    def load_10x_zheng() -> Tuple[np.ndarray, np.ndarray]:
        """Load 10x mouse Zheng single-cell dataset."""
        benchmarks = DatasetLoader.load_scanpy_benchmarks()
        return benchmarks["10x_zheng"]

    @staticmethod
    def load_tasic(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed Tasic dataset (PCA50 + cell types)."""
        data = load_tasic_preprocessed(data_dir=data_dir)
        X = data["X"].astype("float32", copy=False)
        y = LabelEncoder().fit_transform(data["obs"]["cell_type"].astype(str))
        return X, y

    @staticmethod
    def load_coil100(
        data_dir: str | Path,
        coil_path: Optional[str | Path] = None,
        download_if_missing: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        flatten: bool = True,
        as_float: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the COIL-100 dataset."""
        data_dir = Path(data_dir).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        if coil_path is None:
            coil_dir = data_dir / "coil-100"
        else:
            coil_dir = Path(coil_path).expanduser().resolve()

        zip_path = data_dir / "coil-100.zip"

        if download_if_missing and not coil_dir.exists():
            if not zip_path.exists():
                url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-100/coil-100.zip"
                try:
                    urlretrieve(url, zip_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to download COIL-100: {e}") from e

            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)

        if not coil_dir.exists():
            raise FileNotFoundError(
                "COIL-100 dataset not found at "
                f"{coil_dir}. Place the extracted 'coil-100' directory there "
                "or call with download_if_missing=True."
            )

        images = []
        labels = []

        for img_file in sorted(coil_dir.glob("*.png")):
            if img_file.name.startswith("obj"):
                try:
                    parts = img_file.stem.split("__")
                    obj_id = int(parts[0][3:]) - 1
                except (ValueError, IndexError):
                    continue

                img = Image.open(img_file)
                if resize is not None:
                    img = img.resize(resize, Image.Resampling.LANCZOS)
                img_array = np.array(img)

                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                images.append(img_array)
                labels.append(obj_id)

        if len(images) == 0:
            raise ValueError(
                f"No valid COIL-100 images found in {coil_dir}. "
                f"Check that the dataset was extracted correctly."
            )

        X = np.array(images, dtype=np.uint8)
        y = np.array(labels, dtype=np.int64)

        sort_idx = np.argsort(y)
        X = X[sort_idx]
        y = y[sort_idx]

        if flatten:
            n = X.shape[0]
            X = X.reshape(n, -1)
            if as_float:
                X = X.astype(np.float32, copy=False)

        return X, y