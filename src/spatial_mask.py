import numpy as np

# ------------------------------------------------------------
# 1. Compute Euclidean distance matrix from spatial coords
# ------------------------------------------------------------
def compute_spatial_dist_matrix(coords):
    """
    coords: numpy array (n_cells, 2) or (n_cells, 3)

    Returns:
        D_spatial: (n_cells, n_cells) Euclidean distance
    """
    coords = np.asarray(coords)
    n = coords.shape[0]

    # (x-y)^2 trick
    sq = np.sum(coords**2, axis=1)
    D2 = (
        sq[:, None] + sq[None, :] - 2 * coords @ coords.T
    )
    D2[D2 < 0] = 0
    return np.sqrt(D2)
    

# ------------------------------------------------------------
# 2. Build spatial mask
# ------------------------------------------------------------
def build_spatial_mask(
    coords,
    radius_cutoff=100.0,
    kernel_scale=50.0,
    mode="exp",     # "exp" or "linear" or "binary"
):
    """
    Returns:
        M: (n_cells, n_cells) spatial mask in [0,1]
    """
    D_spatial = compute_spatial_dist_matrix(coords)
    n = D_spatial.shape[0]

    # initialize
    M = np.zeros((n, n), dtype=np.float32)

    # 1) Hard cutoff
    inside = D_spatial <= radius_cutoff

    # 2) Scaling scheme
    if mode == "exp":
        # exponential decay: near = 1, far -> 0
        scale = np.exp(- D_spatial / kernel_scale)
        scale[~inside] = 0.0
        M = scale

    elif mode == "linear":
        # linear scaling: 1 at d=0, 0 at d=cutoff
        scale = 1.0 - (D_spatial / radius_cutoff)
        scale[scale < 0] = 0.0
        M = scale

    elif mode == "binary":
        # inside cutoff => 1, else 0
        M = inside.astype(np.float32)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    # symmetry
    M = 0.5 * (M + M.T)

    # no self-suppression
    np.fill_diagonal(M, 1.0)

    return M


# ------------------------------------------------------------
# 3. Apply spatial mask to CCC matrix
# ------------------------------------------------------------
def apply_spatial_mask_to_ccc(
    D_ccc,
    coords,
    radius_cutoff=100.0,
    kernel_scale=50.0,
    mode="exp",
):
    """
    Multiply CCC matrix with a spatial mask.

    Parameters
    ----------
    D_ccc : np.ndarray
        (n_cells, n_cells) CCC distance or Laplacian or similarity
    coords : np.ndarray
        (n_cells, 2/3) spatial coordinates
    radius_cutoff : float
        max spatial distance to allow CCC
    kernel_scale : float
        scaling factor for distance decay
    mode : str
        "exp", "linear", or "binary"

    Returns
    -------
    D_new : np.ndarray
        spatial-aware CCC matrix
    """
    M = build_spatial_mask(
        coords,
        radius_cutoff=radius_cutoff,
        kernel_scale=kernel_scale,
        mode=mode,
    )

    # IMPORTANT:
    # For Laplacian: apply mask on similarity graph BEFORE recomputing L.
    # For distance: apply mask directly.
    return D_ccc * M

