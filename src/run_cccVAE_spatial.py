# ============================================================
# run_cccVAE_spatial.py
# CCCVAE_Spatial with spatial Laplacian on Normal latent
# For Visium data directory
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import json
import torch

from cccVAE_spatial import CCCVAE_Spatial
from preprocess import normalize, geneSelection

from ccc_methods.compute_ccc_cellchat import build_ccc_laplacian_from_expr as build_ccc_cellchat
# from ccc_methods.compute_ccc_cellphonedb import build_ccc_laplacian_from_expr as build_ccc_cellphonedb


# ============================================================
# Auto-detect Visium files
# ============================================================

def find_visium_files(data_dir):

    data_dir = os.path.abspath(data_dir)

    matrix_h5 = None
    for fn in ["filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5.gz"]:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            matrix_h5 = fp
            break

    if matrix_h5 is None:
        raise FileNotFoundError("Cannot find filtered_feature_bc_matrix.h5 or .h5.gz")

    spatial_dir = os.path.join(data_dir, "spatial")
    if not os.path.exists(spatial_dir):
        raise FileNotFoundError("Cannot find spatial/ directory")

    # positions
    pos_file = None
    if os.path.exists(os.path.join(spatial_dir, "tissue_positions_list.txt")):
        pos_file = os.path.join(spatial_dir, "tissue_positions_list.txt")
    elif os.path.exists(os.path.join(spatial_dir, "tissue_positions.parquet")):
        pos_file = os.path.join(spatial_dir, "tissue_positions.parquet")
    else:
        raise FileNotFoundError("Cannot find tissue_positions_list.txt or tissue_positions.parquet")

    # scalefactors
    scale_file = os.path.join(spatial_dir, "scalefactors_json.json")
    if not os.path.exists(scale_file):
        print("WARNING: scalefactors_json.json not found. Using raw pixel coordinates.")
        scale_file = None

    return matrix_h5, pos_file, scale_file


# ============================================================
# argparse
# ============================================================

parser = argparse.ArgumentParser(
    description="CCC + Spatial VAE (spatial loss on Normal latent)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--data_dir", required=True, help="Visium output folder")
parser.add_argument("--output_prefix", default="cccvae_spatial")

# latent dims
parser.add_argument("--CCC_dim", type=int, default=4)
parser.add_argument("--Normal_dim", type=int, default=16)

# architecture
parser.add_argument("--encoder_layers", nargs="+", type=int, default=[256, 128])
parser.add_argument("--decoder_layers", nargs="+", type=int, default=[256])

# CCC hyperparams
parser.add_argument("--ccc_method", type=str, default="cellchat",
                    choices=["cellchat", "cellphonodb"])
parser.add_argument("--ccc_cutoff", type=float, default=0.1)
parser.add_argument("--ccc_top_k", type=int, default=100)
parser.add_argument("--ccc_alpha", type=float, default=0.5)
parser.add_argument("--ccc_Kh", type=float, default=1.0)
parser.add_argument("--ccc_batch_size_pairs", type=int, default=4)

# Warmup & CCC schedule
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--ccc_update_interval", type=int, default=10)
parser.add_argument("--lambda_start", type=float, default=0.2)
parser.add_argument("--lambda_max",   type=float, default=1.0)
parser.add_argument("--lambda_step",  type=float, default=0.2)
parser.add_argument("--lambda_update_interval", type=int, default=10)

# Spatial graph hyperparams
parser.add_argument("--radius_cutoff", type=float, default=100)
parser.add_argument("--spatial_scale", type=float, default=50.0)
parser.add_argument("--spatial_mode", type=str, default="exp")
parser.add_argument("--spatial_lambda_coef", type=float, default=0.001)

# training hyperparams
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--maxiter", type=int, default=500)
parser.add_argument("--train_size", type=float, default=0.95)
parser.add_argument("--patience", type=int, default=30)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-6)

# VAE hyperparams
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--dropoutE", type=float, default=0.0)
parser.add_argument("--dropoutD", type=float, default=0.0)
parser.add_argument("--KL_loss", type=float, default=0.5)
parser.add_argument("--init_beta", type=float, default=1.0)
parser.add_argument("--min_beta", type=float, default=0.01)
parser.add_argument("--max_beta", type=float, default=10.0)

parser.add_argument("--device", default="mps")
parser.add_argument("--select_genes", type=int, default=0)
parser.add_argument("--latent_interval", type=int, default=None)

args = parser.parse_args()


# ============================================================
# 1. Find Visium files
# ============================================================

matrix_h5, pos_file, scale_file = find_visium_files(args.data_dir)

print("Matrix:", matrix_h5)
print("Positions:", pos_file)
print("Scalefactors:", scale_file)


# ============================================================
# 2. Load counts
# ============================================================

adata = sc.read_10x_h5(matrix_h5)
adata.var_names_make_unique()

n_cells, n_genes = adata.shape
print("Loaded matrix:", adata.shape)


# ============================================================
# 3. Load spatial coordinates
# ============================================================

if pos_file.endswith(".txt"):
    pos_df = pd.read_csv(pos_file, sep=",", header=None)
    pos_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
else:
    pos_df = pd.read_parquet(pos_file)
    pos_df.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]

pos_df = pos_df.set_index("barcode")
pos_df = pos_df.loc[adata.obs_names]  # align

if scale_file is not None:
    with open(scale_file, "r") as f:
        scale_json = json.load(f)
    scale = scale_json.get("tissue_lowres_scalef", 1.0)
    print("Using scalefactor:", scale)
else:
    scale = 1.0
    print("No scalefactor found, using raw pixel coords.")

adata.obs["x"] = pos_df["pxl_col"].values * scale
adata.obs["y"] = pos_df["pxl_row"].values * scale
coords = adata.obs[["x", "y"]].values.astype(np.float32)


# ============================================================
# 4. Normalize (log-normalized counts)
# ============================================================

adata = normalize(
    adata,
    filter_min_counts=False,
    size_factors=True,
    normalize_input=True,
    logtrans_input=True
)

ncounts = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
raw_counts = adata.raw.X.toarray() if scipy.sparse.issparse(adata.raw.X) else adata.raw.X
size_factors = adata.obs.size_factors.values.astype(np.float32)


# ============================================================
# 5. HVG selection
# ============================================================

if args.select_genes > 0:
    selected_mask = geneSelection(raw_counts, n=args.select_genes, plot=False, verbose=1)
    hvg_idx = np.where(selected_mask)[0]
else:
    hvg_idx = np.arange(n_genes, dtype=int)


# ============================================================
# 6. CCC builder
# ============================================================

ccc_builder = build_ccc_cellchat if args.ccc_method == "cellchat" else build_ccc_cellphonedb


# ============================================================
# 7. Build model
# ============================================================

pos_index = np.arange(n_cells).astype(int)

model = CCCVAE_Spatial(
    adata=adata,
    input_dim=n_genes,
    CCC_dim=args.CCC_dim,
    Normal_dim=args.Normal_dim,
    encoder_layers=args.encoder_layers,
    decoder_layers=args.decoder_layers,
    noise=args.noise,
    encoder_dropout=args.dropoutE,
    decoder_dropout=args.dropoutD,
    KL_loss=args.KL_loss,
    dynamicVAE=True,
    init_beta=args.init_beta,
    min_beta=args.min_beta,
    max_beta=args.max_beta,
    dtype=torch.float32,
    device=args.device,
    hvg_idx=hvg_idx,
    ccc_builder=ccc_builder,
    ccc_cutoff=args.ccc_cutoff,
    ccc_top_k=args.ccc_top_k,
    ccc_alpha=args.ccc_alpha,
    ccc_Kh=args.ccc_Kh,
    ccc_batch_size_pairs=args.ccc_batch_size_pairs,
    warmup_epochs=args.warmup_epochs,
    ccc_update_interval=args.ccc_update_interval,
    laplacian_lambda_start=args.lambda_start,
    laplacian_lambda_max=args.lambda_max,
    laplacian_lambda_step=args.lambda_step,
    laplacian_lambda_update_interval=args.lambda_update_interval,
    spatial_coords=coords,
    spatial_radius_cutoff=args.radius_cutoff,
    spatial_scale=args.spatial_scale,
    spatial_mode=args.spatial_mode,
    spatial_lambda_coef=args.spatial_lambda_coef,
)


# ============================================================
# 8. Train
# ============================================================

model.train_model(
    pos=pos_index,
    ncounts=ncounts,
    raw_counts=raw_counts,
    size_factors=size_factors,
    lr=args.lr,
    weight_decay=args.weight_decay,
    batch_size=args.batch_size,
    num_samples=1,
    train_size=args.train_size,
    maxiter=args.maxiter,
    patience=args.patience,
    model_weights=args.output_prefix + "_model.pt",
    save_latent_interval=args.latent_interval,
    latent_prefix="latent",
    output_prefix=args.output_prefix
)


# ============================================================
# 9. Save latent & denoised
# ============================================================

latent = model.batching_latent_samples(
    X=pos_index,
    Y=ncounts,
    num_samples=1,
    batch_size=args.batch_size
)
np.savetxt(args.output_prefix + "_latent.txt", latent, delimiter=",")

denoised = model.batching_denoise_counts(
    X=pos_index,
    Y=ncounts,
    n_samples=25,
    batch_size=args.batch_size
)
np.savetxt(args.output_prefix + "_denoised.txt", denoised, delimiter=",")

print("\nAll done.\n")

