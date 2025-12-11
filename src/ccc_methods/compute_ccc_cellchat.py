# ============================================================
# compute_ccc_cellchat.py
# Single-cell CellChat formula -> BI Frobenius distance + Laplacian
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import re


# ============================================================
# Device / dtype helpers
# ============================================================

def _select_device(device):
    if isinstance(device, torch.device):
        return device
    dev_str = str(device).lower()
    if dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype_for_device(device, dtype=None):
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            return dtype
        s = str(dtype).lower()
        if s in ("float64", "double"):
            return torch.float64
        return torch.float32

    if device.type == "cuda":
        return torch.float64
    elif device.type == "mps":
        return torch.float32
    else:
        return torch.float32


# ============================================================
# Utility parsing
# ============================================================

def _parse_genes(s):
    if pd.isna(s) or s == "":
        return []
    parts = re.split(r"[;,|+]+", str(s))
    return [x.strip() for x in parts if x.strip()]


def _geometric_mean(vectors):
    X = torch.stack(vectors, dim=0)      # (k, n_cells)
    return torch.exp(torch.mean(torch.log(X + 1e-9), dim=0))


def _geo_mean_for_list(
    gene_list,
    gene_expr,
    n_cells,
    device,
    dtype,
    allow_missing=False,
):
    if len(gene_list) == 0:
        return torch.zeros(n_cells, device=device, dtype=dtype)

    vecs = []
    for g in gene_list:
        if g in gene_expr:
            vecs.append(gene_expr[g])
        else:
            if allow_missing:
                return torch.zeros(n_cells, device=device, dtype=dtype)
            else:
                return None

    return _geometric_mean(vecs)


# ============================================================
# 1. 从 expr + CellChatDB 构建按 CellChat 公式需要的量
# ============================================================

def _default_lrdb_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "CellChatDB.human.v1.csv")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"Default CellChatDB.human.v1.csv not found at: {candidate}\n"
            f"Please put CellChatDB.human.v1.csv in the same folder as compute_ccc_cellchat.py."
        )
    return candidate


def _prepare_LR_items_cellchat(
    expr,
    genes,
    lrdb_path=None,
    top_k=None,
    cutoff=0.1,
    device=None,
    dtype=None,
    Kh=1.0,
    verbose=True,
):
    device = _select_device(device)
    dtype = _select_dtype_for_device(device, dtype)

    # expr -> torch
    if isinstance(expr, np.ndarray):
        expr_t = torch.tensor(expr, device=device, dtype=dtype)
    else:
        expr_t = expr.to(device=device, dtype=dtype)

    n_cells, n_genes = expr_t.shape

    gene_expr = {g: expr_t[:, i] for i, g in enumerate(genes)}

    # 读取 LR DB
    if lrdb_path is None:
        lrdb_path = _default_lrdb_path()
    df = pd.read_csv(lrdb_path)

    LR_items = []

    for _, row in df.iterrows():
        L_genes  = _parse_genes(row.get("ligand", ""))
        R_genes  = _parse_genes(row.get("receptor", ""))
        AG_genes = _parse_genes(row.get("agonist", ""))
        AN_genes = _parse_genes(row.get("antagonist", ""))
        RA_genes = _parse_genes(row.get("co_A_receptor", ""))
        RI_genes = _parse_genes(row.get("co_I_receptor", ""))

        # L 和 R 必须存在
        L_vals = _geo_mean_for_list(L_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if L_vals is None:
            continue

        R_core = _geo_mean_for_list(R_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if R_core is None:
            continue

        # co-stimulatory / co-inhibitory receptor
        RA_vals = _geo_mean_for_list(RA_genes, gene_expr, n_cells, device, dtype, allow_missing=True)
        RI_vals = _geo_mean_for_list(RI_genes, gene_expr, n_cells, device, dtype, allow_missing=True)

        R_vals = R_core * (1.0 + RA_vals) / (1.0 + RI_vals)

        # agonist / antagonist
        AG_vals = _geo_mean_for_list(AG_genes, gene_expr, n_cells, device, dtype, allow_missing=True)
        AN_vals = _geo_mean_for_list(AN_genes, gene_expr, n_cells, device, dtype, allow_missing=True)

        L_mean = L_vals.mean().item()
        R_mean = R_vals.mean().item()
        score = min(L_mean, R_mean)

        if score < cutoff:
            continue

        LR_items.append(
            {
                "L_vals":  L_vals,
                "R_vals":  R_vals,
                "AG_vals": AG_vals,
                "AN_vals": AN_vals,
                "score":   score,
                "name":    row.get("interaction_name", ""),
            }
        )

    if len(LR_items) == 0:
        raise ValueError(
            f"No valid LR pairs after cutoff={cutoff}. "
            f"Check LR DB and gene names."
        )

    LR_items = sorted(LR_items, key=lambda x: x["score"], reverse=True)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive or None.")
        if top_k < len(LR_items):
            LR_items = LR_items[:top_k]

    return LR_items, n_cells, device, dtype


# ============================================================
# 2. Strict CellChat formula -> BI Frobenius distance
# ============================================================

def compute_ccc_bi_distance_from_expr(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    verbose=True,
):
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_cellchat(
        expr=expr,
        genes=genes,
        lrdb_path=None,
        top_k=top_k,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        Kh=Kh,
        verbose=verbose,
    )

    D_out2 = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)
    D_in2  = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)

    K = len(LR_items)

    for start in range(0, K, batch_size_pairs):
        end = min(start + batch_size_pairs, K)
        batch = LR_items[start:end]
        B = len(batch)
        if B == 0:
            continue

        L_batch  = torch.stack([item["L_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        R_batch  = torch.stack([item["R_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        AG_batch = torch.stack([item["AG_vals"] for item in batch], dim=0).to(device_t, dtype_t)
        AN_batch = torch.stack([item["AN_vals"] for item in batch], dim=0).to(device_t, dtype_t)

        # CellChat 单细胞公式
        Li = L_batch[:, :, None]        # (B, n, 1)
        Rj = R_batch[:, None, :]        # (B, 1, n)
        LR = Li * Rj                    # (B, n, n)
        LR_term = LR / (Kh + LR)

        AG_i = AG_batch[:, :, None]
        AG_j = AG_batch[:, None, :]
        AG_term = (1.0 + AG_i / (Kh + AG_i)) * (1.0 + AG_j / (Kh + AG_j))

        AN_i = AN_batch[:, :, None]
        AN_j = AN_batch[:, None, :]
        AN_term = (Kh / (Kh + AN_i)) * (Kh / (Kh + AN_j))

        P = LR_term * AG_term * AN_term   # (B, n, n)

        # OUT profile
        X_out = P.reshape(B * n_cells, n_cells)
        Gram_out = X_out.T @ X_out
        diag_out = torch.diag(Gram_out)
        contrib_out = diag_out[:, None] + diag_out[None, :] - 2.0 * Gram_out
        D_out2 += contrib_out

        # IN profile
        X_in = P.permute(1, 0, 2).reshape(n_cells, B * n_cells)
        Gram_in = X_in @ X_in.T
        diag_in = torch.diag(Gram_in)
        contrib_in = diag_in[:, None] + diag_in[None, :] - 2.0 * Gram_in
        D_in2 += contrib_in

        del L_batch, R_batch, AG_batch, AN_batch
        del Li, Rj, LR, LR_term, AG_i, AG_j, AG_term, AN_i, AN_j, AN_term, P
        del X_out, Gram_out, diag_out, contrib_out, X_in, Gram_in, diag_in, contrib_in
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    eps = 1e-12
    D_out2 = torch.clamp(D_out2, min=0.0)
    D_in2  = torch.clamp(D_in2,  min=0.0)

    D_bi2 = alpha * D_out2 + (1.0 - alpha) * D_in2
    D_bi2 = torch.clamp(D_bi2, min=0.0)
    D_bi  = torch.sqrt(D_bi2 + eps)

    D_bi_np = D_bi.detach().cpu().numpy().astype(np.float64)

    D_bi_np = 0.5 * (D_bi_np + D_bi_np.T)
    np.fill_diagonal(D_bi_np, 0.0)

    D_bi_np[D_bi_np < 0] = 0.0
    finite_mask = np.isfinite(D_bi_np)
    if not finite_mask.all():
        finite_max = np.max(D_bi_np[finite_mask])
        D_bi_np[~finite_mask] = finite_max

    return D_bi_np


# ============================================================
# 3. 从 D_bi 构建对称 normalized Laplacian
# ============================================================

def build_laplacian_from_distance(D_bi):
    D = np.asarray(D_bi, dtype=np.float64)

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    finite_mask = np.isfinite(D)
    if not finite_mask.all():
        finite_max = np.max(D[finite_mask])
        D[~finite_mask] = finite_max

    pos = D[D > 0]
    if pos.size == 0:
        n = D.shape[0]
        return np.zeros((n, n), dtype=np.float64)

    sigma = np.median(pos)
    if sigma <= 0:
        sigma = np.mean(pos) if np.mean(pos) > 0 else 1.0

    S = np.exp(- (D ** 2) / (2.0 * sigma ** 2))
    S = 0.5 * (S + S.T)
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)

    deg = S.sum(axis=1)
    deg[deg <= 1e-12] = 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(deg)

    n = S.shape[0]
    I = np.eye(n, dtype=np.float64)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L = I - D_inv_sqrt_mat @ S @ D_inv_sqrt_mat

    L = 0.5 * (L + L.T)
    L[L < 0] = 0.0

    return L


# ============================================================
# 4. 对外接口：cccVAE 用的 Laplacian 构建函数
# ============================================================

def build_ccc_laplacian_from_expr(
    expr,
    genes,
    cutoff=0.1,
    top_k=None,
    alpha=0.5,
    Kh=1.0,
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    verbose=True,
):
    D_bi = compute_ccc_bi_distance_from_expr(
        expr=expr,
        genes=genes,
        top_k=top_k,
        cutoff=cutoff,
        alpha=alpha,
        Kh=Kh,
        device=device,
        dtype=dtype,
        batch_size_pairs=batch_size_pairs,
        verbose=verbose,
    )

    L_np = build_laplacian_from_distance(D_bi)
    return L_np

