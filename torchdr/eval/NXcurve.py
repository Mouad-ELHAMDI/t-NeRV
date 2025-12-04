# NXcurve: exact + fast (fastqa) R-NX with large-N safe init
# -----------------------------------------------------------------------------
# Single-file, no imports inside methods; logic matches your fastqa & NXcurve docs.

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict, Any
from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as ssd
import numba


class NXcurve:
    """
    Unified R-NX quality curves with exact and fast (fastqa) evaluators.

    Changes vs prior version (logic preserved):
      • Avoid O(N^2) HD distance at __init__: computed lazily ONLY if exact() is called.
      • Exact path computes LD distances with Euclidean (as in your tests).
      • Cosine distance avoids mutating inputs.
      • Same fastqa sampling/pow2K/ranking pipeline as provided in fastqa.txt.
    """

    # =========================== NUMBA KERNELS (fastqa) =========================
    # Kept inside the class (static) to satisfy the "single class" requirement.

    def _eucl_dist(X: np.ndarray, x: np.ndarray) -> np.ndarray:
        M = X - x
        return np.sqrt((M * M).sum(axis=1))

    def _cos_dist(X: np.ndarray, x: np.ndarray) -> np.ndarray:
        # 1 - cosine similarity, without mutating inputs
        xnorm = np.sqrt((x * x).sum())
        if xnorm == 0.0:
            return np.ones(X.shape[0])
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            s = 0.0
            xn = 0.0
            for j in range(X.shape[1]):
                s += X[i, j] * x[j]
                xn += X[i, j] * X[i, j]
            xn = np.sqrt(xn)
            if xn == 0.0:
                out[i] = 1.0
            else:
                out[i] = 1.0 - (s / (xn * xnorm))
        return out

    def _randomized_partition(a, ind, p, r, tmp):
        r_1 = r + 1
        i = np.random.randint(p, r_1)
        x = a[ind[i]]
        for j in range(r, i, -1):
            if a[ind[j]] == x:
                i = j
                break
        rv = ind[i]
        i_tmp = 0
        i_ind = p
        for lb, ub in [(p, i), (i+1, r_1)]:
            for j in range(lb, ub, 1):
                if a[ind[j]] > x:
                    tmp[i_tmp] = ind[j]
                    i_tmp += 1
                else:
                    ind[i_ind] = ind[j]
                    i_ind += 1
        ind[i_ind] = rv
        ind[i_ind+1:r_1] = tmp[:i_tmp]
        return i_ind

    def _randomized_partition_unstable(a, ind, p, r):
        i = np.random.randint(p, r+1)
        x = a[ind[i]]
        t = ind[r]; ind[r] = ind[i]; ind[i] = t
        i = p
        for j in range(p, r, 1):
            if a[ind[j]] <= x:
                t = ind[i]; ind[i] = ind[j]; ind[j] = t
                i += 1
        t = ind[r]; ind[r] = ind[i]; ind[i] = t
        return i

    def _pow2_sort(a, ind, p, r, tmp):
        if p != r:
            q = NXcurve._randomized_partition(a, ind, p, r, tmp) - 1
            if (q > p) and ((p == 0) or (int(np.log2(p)) != int(np.log2(q)))):
                NXcurve._pow2_sort(a, ind, p, q, tmp)
            q += 2
            if (q < r) and (int(np.log2(q)) != int(np.log2(r))):
                NXcurve._pow2_sort(a, ind, q, r, tmp)

    def _pow2_rank(d, tmp):
        N = d.size
        ind = np.arange(N)
        NXcurve._pow2_sort(d, ind, 0, N-1, tmp)
        r = np.empty(shape=N, dtype=np.int64)
        r[ind[0]] = 0
        lb, ub, cr = 1, 2, 1
        while lb < N:
            r[ind[lb:ub]] = cr
            cr += 1
            lb = ub
            ub = min(N, ub * 2)
        return r

    def _rank_from_dist(d):
        nd = d.size
        v = d.argsort(kind='mergesort')
        r = np.empty(shape=nd, dtype=np.int64)
        for i in range(nd):
            r[v[i]] = i
        return r

    def _lower_median_partition(a, ind):
        p = 0
        r = ind.size - 1
        imed = r // 2
        while p < r:
            q = NXcurve._randomized_partition_unstable(a, ind, p, r)
            if q == imed:
                break
            elif q > imed:
                r = q - 1
            else:
                p = q + 1
        return imed

    def _vp_repr(X_hd, dist_hd, n):
        N, M = X_hd.shape
        ind = np.arange(N)
        depth = int(np.log2(n))
        nleafs = 2**depth
        leafs = np.empty(shape=nleafs+1, dtype=np.int64)
        leafs[0] = 0
        leafs[nleafs] = N
        depth_cur = 0
        jump = nleafs
        mean_cur = np.empty(shape=M, dtype=np.float64)
        d_vp = np.empty(shape=N, dtype=np.float64)
        while depth_cur < depth:
            jump_2 = jump // 2
            leaf_start = 0
            while leaf_start < nleafs:
                leaf_stop = leaf_start + jump
                for j in range(M):
                    mean_cur[j] = X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]], j].mean()
                blk = X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]], :]
                vp_idx = dist_hd(blk, blk[dist_hd(blk, mean_cur).argmin(), :]).argmax()
                d_vp[ind[leafs[leaf_start]:leafs[leaf_stop]]] = dist_hd(
                    blk, blk[vp_idx, :]
                )
                leafs[leaf_start + jump_2] = NXcurve._lower_median_partition(
                    d_vp, ind[leafs[leaf_start]:leafs[leaf_stop]]
                ) + leafs[leaf_start] + 1
                leaf_start = leaf_stop
            depth_cur += 1
            jump = jump_2
        reprs = np.empty(shape=n, dtype=np.int64)
        n_cur = 0
        id_leafs = np.arange(nleafs)
        iid = 0
        leafs_stop = leafs[1:].copy()
        leafs = leafs[:nleafs]
        while n_cur < n:
            if iid == nleafs:
                np.random.shuffle(id_leafs)
                iid = 0
            cL = id_leafs[iid]
            if leafs_stop[cL] - leafs[cL] > 0:
                id_samp = np.random.randint(leafs[cL], leafs_stop[cL])
                reprs[n_cur] = ind[id_samp]
                ind[id_samp] = ind[leafs[cL]]
                leafs[cL] += 1
                n_cur += 1
            iid += 1
        return reprs

    def _fast_eval_qnx(X_hd, X_ld, dist_hd, dist_ld, n=-1, seed=-1, pow2K=False, vp_samp=False):
        # Mirrors fastqa.txt behavior and outputs (Q_NX, denom grid).  (No logic change.)
        if seed >= 0:
            np.random.seed(seed)
        N = X_hd.shape[0]
        N_1 = N - 1
        if n < 1:
            n = int(round(10 * np.log(N)))
        if n > N:
            n = N
        if pow2K:
            size_qnx = int(np.log2(N_1))
            size_qnx = size_qnx + 2 if 2**size_qnx < N_1 else size_qnx + 1
            den_qnx = np.empty(shape=size_qnx, dtype=np.float64)
            den_qnx[0] = 1.0
            for i in range(1, size_qnx-1, 1):
                den_qnx[i] = den_qnx[i-1] * 2
            den_qnx[size_qnx-1] = float(N_1)
        else:
            size_qnx = N_1
            den_qnx = np.arange(N_1) + 1.0
        qnx = np.zeros(shape=size_qnx, dtype=np.int64)
        tmp = np.empty(shape=N_1, dtype=np.int64)
        if vp_samp:
            samp = NXcurve._vp_repr(X_hd, dist_hd, n)
        else:
            samp = np.random.choice(N, size=n, replace=False)
        for i in samp:
            i_1 = i + 1
            d_hd = dist_hd(np.vstack((X_hd[:i, :], X_hd[i_1:, :])), X_hd[i, :])
            d_ld = dist_ld(np.vstack((X_ld[:i, :], X_ld[i_1:, :])), X_ld[i, :])
            if pow2K:
                r_hd = NXcurve._pow2_rank(d_hd, tmp)
                r_ld = NXcurve._pow2_rank(d_ld, tmp)
            else:
                r_hd = NXcurve._rank_from_dist(d_hd)
                r_ld = NXcurve._rank_from_dist(d_ld)
            for j, rj_hd in enumerate(r_hd):
                qnx[max(rj_hd, r_ld[j])] += 1
        return qnx.cumsum().astype(np.float64) / (n * den_qnx), den_qnx  # Q_NX(K)

    def _eval_auc(arr: np.ndarray) -> np.float64:
        i_all_k = 1.0 / (np.arange(arr.size) + 1.0)
        return np.float64(arr.dot(i_all_k)) / (i_all_k.sum())

    def _eval_auc_lin(arr: np.ndarray) -> np.float64:
        return np.float64(arr.sum()) / arr.size

    def _eval_rnx_from_qnx(qnx: np.ndarray) -> np.ndarray:
        N_1 = qnx.size
        arr_K = np.arange(N_1)[1:].astype(np.float64)
        return (N_1 * qnx[:N_1-1] - arr_K) / (N_1 - arr_K)

    # ============================ EXACT (co-ranking) ============================

    @staticmethod
    def _coranking_from_dist(d_hd: np.ndarray, d_ld: np.ndarray) -> np.ndarray:
        # Stable mergesort ranks; identical coranking logic to your NXcurve.
        perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
        perm_ld = d_ld.argsort(axis=-1, kind='mergesort')
        N = d_hd.shape[0]
        i = np.arange(N, dtype=np.int64)
        R = np.empty(shape=(N, N), dtype=np.int64)
        for j in range(N):
            R[perm_ld[j, i], j] = i
        Q = np.zeros(shape=(N, N), dtype=np.int64)
        for j in range(N):
            Q[i, R[perm_hd[j, i], j]] += 1
        return Q[1:, 1:]

    @staticmethod
    def _eval_qnx_rnx_from_Q(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N_1 = Q.shape[0]
        N = N_1 + 1
        qnxk = np.empty(shape=N_1, dtype=np.float64)
        acc_q = 0.0
        for K in range(N_1):
            acc_q += (Q[K, K] + Q[K, :K].sum() + Q[:K, K].sum())
            qnxk[K] = acc_q / ((K + 1) * N)
        arr_K = np.arange(N_1)[1:].astype(np.float64)
        rnxk = (N_1 * qnxk[:N_1-1] - arr_K) / (N_1 - arr_K)
        return qnxk, rnxk

    # =============================== FINGERPRINT ===============================

    def _fnv1a64(u8: np.ndarray) -> np.uint64:
        h = np.uint64(1469598103934665603)
        p = np.uint64(1099511628211)
        for b in u8:
            h ^= np.uint64(b)
            h *= p
        return h

    @staticmethod
    def _fingerprint_array(arr: np.ndarray) -> Tuple:
        arr = np.asarray(arr)
        # ensure last axis contiguous so .view(np.uint8) is legal
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)   # copy only if needed
        u8 = arr.view(np.uint8).ravel()       # <-- flatten to 1-D for the numba kernel
        h = NXcurve._fnv1a64(u8).item()
        return (arr.shape, str(arr.dtype), arr.strides, int(h))

    # ================================ CACHING =================================

    def _ensure_cache(self):
        if not hasattr(self, "_cache"):
            self._cache: "OrderedDict[Tuple, Dict[str, Any]]" = OrderedDict()
        if not hasattr(self, "_cache_budget"):
            self._cache_budget = self.max_cache_bytes

    def _estimate_bytes(self, obj: Any) -> int:
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        if isinstance(obj, (list, tuple)):
            return int(sum(self._estimate_bytes(x) for x in obj))
        if isinstance(obj, dict):
            return int(sum(self._estimate_bytes(v) for v in obj.values()))
        return 0

    def _cache_put(self, key: Tuple, payload: Dict[str, Any]) -> None:
        self._ensure_cache()
        item_bytes = self._estimate_bytes(payload)
        if item_bytes > self.max_cache_bytes:
            return
        while self._cache and (
            self._estimate_bytes(self._cache) + item_bytes > self.max_cache_bytes
            or len(self._cache) >= self.max_cache_items
        ):
            self._cache.popitem(last=False)
        self._cache[key] = payload
        self._cache.move_to_end(key, last=True)

    def _cache_get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        self._ensure_cache()
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key, last=True)
        return val

    def clear_cache(self) -> None:
        self._cache = OrderedDict()

    # ================================ INIT ====================================

    def __init__(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metric_hd: str = "euclidean",
        cache_mode: str = "rnx",   # {'rnx','qnx','coranking','none'}
        max_cache_bytes: int = 256 * 1024 * 1024,
        max_cache_items: int = 3,
        precompute_exact_hd: bool = False,  # NEW: default False to support big N
    ):
        """
        Args mirror your prior class; only change is precompute_exact_hd=False so
        the constructor does NOT allocate O(N^2) memory by default.
        """
        self.X = np.asarray(X)
        self.labels = labels
        self.metric_hd = metric_hd
        self.cache_mode = cache_mode
        self.max_cache_bytes = int(max_cache_bytes)
        self.max_cache_items = int(max_cache_items)
        self._cache = OrderedDict()

        # Big-N safe by default: don't precompute HD distances unless requested
        self._Dx = None
        if precompute_exact_hd:
            self._Dx = pairwise_distances(self.X, metric=("euclidean" if metric_hd == "euclidean" else "cosine"))

    # ============================= EXACT INTERFACE =============================

    def exact_qrnx_auc(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Exact Q_NX(K), R_NX(K), and AUC (linear K-weighting).
        This allocates O(N^2) memory/time, per the exact metric definition.
        """
        Y = np.asarray(Y)
        key = ("exact", self._fingerprint_array(Y), self.metric_hd, self.cache_mode)
        cached = self._cache_get(key)
        if cached is not None and "rnx" in cached:
            return cached.get("qnx"), cached.get("rnx"), cached.get("auc_lin")

        # Lazily compute HD distances if needed
        if self._Dx is None:
            self._Dx = pairwise_distances(self.X, metric=("euclidean" if self.metric_hd == "euclidean" else "cosine"))

        d_hd = self._Dx
        # LD distances follow your reference tests: Euclidean in embedding space
        d_ld = pairwise_distances(Y, metric="euclidean")  # matches fastqa test exact path :contentReference[oaicite:3]{index=3}

        Q = NXcurve._coranking_from_dist(d_hd, d_ld)
        qnx, rnx = NXcurve._eval_qnx_rnx_from_Q(Q)
        auc_lin = NXcurve._eval_auc_lin(rnx)

        payload: Dict[str, Any] = {"rnx": rnx, "qnx": qnx, "auc_lin": auc_lin}
        if self.cache_mode == "coranking":
            payload["Q"] = Q
        self._cache_put(key, payload)
        return qnx, rnx, auc_lin

    # ============================== FAST INTERFACE =============================

    def fast_qrnx_auc(
        self,
        Y: np.ndarray,
        n: int = -1,
        seed: int = -1,
        pow2K: bool = False,
        vp_samp: bool = False,
        metric_ld: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Fast (fastqa) Q_NX, R_NX, AUC_linear, AUC_log (harmonic weights).
        Logic mirrors fastqa.txt exactly (sampling, pow2 grid, ranking). :contentReference[oaicite:4]{index=4}
        """
        Y = np.asarray(Y)
        key = ("fast", self._fingerprint_array(Y), n, seed, pow2K, vp_samp, self.metric_hd, metric_ld, self.cache_mode)
        cached = self._cache_get(key)
        if cached is not None and "rnx" in cached:
            return cached.get("qnx"), cached.get("rnx"), cached.get("auc_lin"), cached.get("auc_log")

        dist_hd = NXcurve._eucl_dist if self.metric_hd == "euclidean" else NXcurve._cos_dist
        dist_ld = NXcurve._eucl_dist if metric_ld == "euclidean" else NXcurve._cos_dist

        qnx, den_qnx = NXcurve._fast_eval_qnx(self.X, Y, dist_hd, dist_ld, n=n, seed=seed, pow2K=pow2K, vp_samp=vp_samp)
        if pow2K:
            srnx = qnx.size - 1
            N_1 = self.X.shape[0] - 1
            rnx = (N_1 * qnx[:srnx] - den_qnx[:srnx]) / (N_1 - den_qnx[:srnx])
            auc_lin = NXcurve._eval_auc_lin(rnx)
            auc_log = NXcurve._eval_auc(rnx)
        else:
            rnx = NXcurve._eval_rnx_from_qnx(qnx)
            auc_lin = NXcurve._eval_auc_lin(rnx)
            auc_log = NXcurve._eval_auc(rnx)

        payload: Dict[str, Any] = {"rnx": rnx, "qnx": qnx, "auc_lin": float(auc_lin), "auc_log": float(auc_log)}
        self._cache_put(key, payload)
        return qnx, rnx, float(auc_lin), float(auc_log)

    # ============================== HIGH-LEVEL API =============================

    def compute_rnx(
        self,
        Y: np.ndarray,
        mode: str = "fast",     # {'fast','exact'}
        **fast_kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Returns (R_NX, AUC_linear) for plotting.
        """
        if mode == "exact":
            _, rnx, auc_lin = self.exact_qrnx_auc(Y)
            return rnx, float(auc_lin)
        else:
            _, rnx, auc_lin, _ = self.fast_qrnx_auc(Y, **fast_kwargs)
            return rnx, float(auc_lin)

    def compare_embeddings(
        self,
        embeddings: List[np.ndarray],
        method_names: Optional[List[str]] = None,
        mode: str = "fast",
        log_scale: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        colors: Optional[List[str]] = None,
        fast_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[float], plt.Figure]:
        """
        Plot 100·R_NX(K) for a list of embeddings (same style as your NXcurve).  :contentReference[oaicite:5]{index=5}
        """
        if fast_kwargs is None:
            fast_kwargs = {}
        if method_names is None:
            method_names = [f"Embedding {i+1}" for i in range(len(embeddings))]
        if len(embeddings) != len(method_names):
            raise ValueError("Number of embeddings must match number of method names")

        fig, ax = plt.subplots(figsize=figsize)
        aucs: List[float] = []

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i, (Y, name) in enumerate(zip(embeddings, method_names)):
            rnx, auc = self.compute_rnx(Y, mode=mode, **fast_kwargs)
            aucs.append(auc)
            ks = np.arange(1, len(rnx) + 1)
            ax.plot(ks, 100.0 * rnx, label=f"{name} (AUC: {auc:.3f})", color=colors[i % len(colors)])

        ax.set_ylim(0, 100)
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(r"$K$", fontsize=12)
        ax.set_ylabel(r"$100\,R_{NX}(K)$", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()
        return aucs, fig