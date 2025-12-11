import os
from collections import deque

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal, kl_divergence
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from I_PID import PIDControl
from VAE_utils import DenseEncoder, buildNetwork, MeanAct, NBLoss


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
            return

        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


# ============================================================
# Spatial utilities
# ============================================================

def pairwise_dist(x: np.ndarray) -> np.ndarray:
    """
    Euclidean distance matrix for coords (n, d).
    Works for 2D / 3D / any d.
    """
    x = np.asarray(x, dtype=float)
    sq = np.sum(x**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (x @ x.T)
    d2[d2 < 0] = 0.0
    return np.sqrt(d2)


def build_spatial_kernel(coords: np.ndarray,
                         radius_cutoff: float = None,
                         spatial_scale: float = 50.0,
                         mode: str = "exp"):
    """
    Build spatial similarity kernel W and graph Laplacian L from coords.

    Parameters
    ----------
    coords : (n_cells, d)
    radius_cutoff : float or None
        If None, will use median pairwise distance.
    spatial_scale : float
    mode : "exp" | "linear" | "binary"

    Returns
    -------
    W : (n, n) spatial kernel
    L : (n, n) graph Laplacian
    median_dist : float
    """
    n = coords.shape[0]
    D = pairwise_dist(coords)

    # median of off-diagonal distances
    nonzero = D[np.triu_indices(n, k=1)]
    median_dist = float(np.median(nonzero))

    if radius_cutoff is None:
        radius_cutoff = median_dist

    if mode == "exp":
        W = np.exp(-D / spatial_scale)
        W[D > radius_cutoff] = 0.0
    elif mode == "linear":
        W = 1.0 - D / radius_cutoff
        W[W < 0.0] = 0.0
    elif mode == "binary":
        W = (D <= radius_cutoff).astype(float)
    else:
        raise ValueError(f"Unknown spatial mode: {mode}")

    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)

    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    return W, L, median_dist


# ============================================================
# CCC + Spatial VAE (spatial loss on Normal latent)
# ============================================================

class CCCVAE_Spatial(nn.Module):
    """
    Latent = [CCC_dim | Normal_dim]

    - CCC latent: graph Laplacian from CCC builder (CellChat / CellPhoneDB)
    - Normal latent: KL + spatial Laplacian (直接把空间约束加在表达 latent 上)
    """

    def __init__(
        self,
        adata,
        input_dim,
        CCC_dim,
        Normal_dim,
        encoder_layers,
        decoder_layers,
        noise,
        encoder_dropout,
        decoder_dropout,
        KL_loss,
        dynamicVAE,
        init_beta,
        min_beta,
        max_beta,
        dtype,
        device,

        # HVG mask
        hvg_idx=None,

        # CCC methods
        ccc_builder=None,
        ccc_cutoff=0.1,
        ccc_top_k=None,
        ccc_alpha=0.5,
        ccc_Kh=1.0,
        ccc_batch_size_pairs=4,

        # warmup schedule for CCC
        warmup_epochs=20,
        ccc_update_interval=5,

        # CCC Laplacian λ schedule
        laplacian_lambda_start=0.1,
        laplacian_lambda_max=1.0,
        laplacian_lambda_step=0.1,
        laplacian_lambda_update_interval=5,

        # Spatial info (applied on Normal latent)
        spatial_coords=None,            # np.array (n_cells, d)
        spatial_radius_cutoff=None,
        spatial_scale=50.0,
        spatial_mode="exp",
        spatial_lambda_coef=0.3,       # lambda_spatial = median_dist * coef
    ):
        super().__init__()

        torch.set_default_dtype(dtype)
        self.dtype = dtype
        self.device = device

        # ---------------- Latent dims ----------------
        if CCC_dim < 0 or Normal_dim <= 0:
            raise ValueError("CCC_dim>=0, Normal_dim>0 required.")
        self.CCC_dim = CCC_dim
        self.Normal_dim = Normal_dim
        self.latent_dim = CCC_dim + Normal_dim

        # ---------------- Encoder / Decoder ----------------
        self.encoder = DenseEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            output_dim=self.latent_dim,
            activation="elu",
            dropout=encoder_dropout,
        )

        self.decoder = buildNetwork(
            [self.latent_dim] + decoder_layers,
            activation="elu",
            dropout=decoder_dropout,
        )

        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(
                nn.Linear(decoder_layers[-1], input_dim),
                MeanAct()
            )
        else:
            self.dec_mean = nn.Sequential(
                nn.Linear(self.latent_dim, input_dim),
                MeanAct()
            )

        self.dec_disp = nn.Parameter(torch.randn(input_dim))
        self.NB_loss = NBLoss().to(device)

        # ---------------- Dynamic KL on Normal latent ----------------
        self.PID = PIDControl(
            Kp=0.01,
            Ki=-0.005,
            init_beta=init_beta,
            min_beta=min_beta,
            max_beta=max_beta,
        )
        self.KL_loss = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta

        # ---------------- CCC builder & params ----------------
        self.ccc_builder = ccc_builder
        self.ccc_cutoff = ccc_cutoff
        self.ccc_top_k = ccc_top_k
        self.ccc_alpha = ccc_alpha
        self.ccc_Kh = ccc_Kh
        self.ccc_batch_size_pairs = ccc_batch_size_pairs

        self.warmup_epochs = warmup_epochs
        self.ccc_update_interval = ccc_update_interval

        self.laplacian_lambda_start = laplacian_lambda_start
        self.laplacian_lambda_max = laplacian_lambda_max
        self.laplacian_lambda_step = laplacian_lambda_step
        self.laplacian_lambda_update_interval = laplacian_lambda_update_interval

        self.ccc_L = None
        self.lambda_graph_ccc = 0.0

        # gene names
        self.genes = list(adata.var_names)

        # HVG mask
        if hvg_idx is None:
            self.hvg_idx = None
        else:
            self.hvg_idx = torch.tensor(hvg_idx, dtype=torch.long, device=device)

        # noise (占位，目前没用)
        self.noise = noise

        # ---------------- Spatial on Normal latent ----------------
        self.spatial_coords_np = None
        self.spatial_L = None
        self.lambda_graph_spatial = 0.0

        if spatial_coords is not None:
            coords = np.asarray(spatial_coords, dtype=float)
            self.spatial_coords_np = coords.copy()

            _, L_spatial, median_dist = build_spatial_kernel(
                coords,
                radius_cutoff=spatial_radius_cutoff,
                spatial_scale=spatial_scale,
                mode=spatial_mode,
            )
            L_sp = torch.tensor(L_spatial, dtype=self.dtype, device=self.device)
            L_sp = 0.5 * (L_sp + L_sp.T)
            self.spatial_L = L_sp

            self.lambda_graph_spatial = float(median_dist * spatial_lambda_coef)
        else:
            self.spatial_coords_np = None
            self.spatial_L = None
            self.lambda_graph_spatial = 0.0

        self.to(device)

    # ------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        d = torch.load(path, map_location="cpu")
        model_dict = self.state_dict()
        d = {k: v for k, v in d.items() if k in model_dict}
        model_dict.update(d)
        self.load_state_dict(model_dict)

    # ============================================================
    # Update CCC Laplacian
    # ============================================================

    def update_ccc_laplacian(self, L_np):
        if L_np is None:
            self.ccc_L = None
            return
        L = torch.tensor(L_np, dtype=self.dtype, device=self.device)
        L = 0.5 * (L + L.T)
        L = torch.clamp(L, min=0.0)
        self.ccc_L = L

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x, y, raw_y, size_factors, num_samples=1):
        """
        x: cell indices (batch)
        y: normalized log counts (batch, genes)
        raw_y: raw counts
        size_factors: per-cell size factor
        """
        b = y.shape[0]

        # Encode -> q_mu, q_var
        q_mu, q_var = self.encoder(y)

        # split latent
        if self.CCC_dim > 0:
            ccc_mu = q_mu[:, :self.CCC_dim]
            ccc_var = q_var[:, :self.CCC_dim]
        else:
            ccc_mu = None
            ccc_var = None

        norm_mu = q_mu[:, self.CCC_dim:]
        norm_var = q_var[:, self.CCC_dim:]

        # KL on Normal part
        prior_norm = Normal(torch.zeros_like(norm_mu), torch.ones_like(norm_var))
        post_norm = Normal(norm_mu, torch.sqrt(norm_var + 1e-8))
        kl_norm = kl_divergence(post_norm, prior_norm).sum()

        # reconstruction (full latent)
        post_all = Normal(q_mu, torch.sqrt(q_var + 1e-8))
        recon_loss = 0.0

        for _ in range(num_samples):
            z = post_all.rsample()
            h = self.decoder(z)
            mean = self.dec_mean(h)
            disp = torch.exp(torch.clamp(self.dec_disp, -15, 15)).unsqueeze(0)

            if self.hvg_idx is not None:
                idx = self.hvg_idx
                recon_loss += self.NB_loss(
                    x=raw_y[:, idx],
                    mean=mean[:, idx],
                    disp=disp[:, idx],
                    scale_factor=size_factors,
                )
            else:
                recon_loss += self.NB_loss(
                    x=raw_y,
                    mean=mean,
                    disp=disp,
                    scale_factor=size_factors,
                )

        recon_loss /= num_samples

        # ---------------- CCC graph loss ----------------
        graph_loss_ccc = 0.0
        if self.CCC_dim > 0 and self.ccc_L is not None and self.lambda_graph_ccc > 0:
            idx_cells = x.long()
            L_bb = self.ccc_L.index_select(0, idx_cells).index_select(1, idx_cells)
            graph_loss_ccc = torch.trace(ccc_mu.t() @ L_bb @ ccc_mu) / float(b)

        # ---------------- Spatial Laplacian on Normal latent ----------------
        graph_loss_spatial = 0.0
        if self.spatial_L is not None and self.lambda_graph_spatial > 0:
            idx_cells = x.long()
            Zn = norm_mu  # (B, Normal_dim)

            L_bb_sp = self.spatial_L.index_select(0, idx_cells).index_select(1, idx_cells)
            lap_loss = torch.trace(Zn.t() @ L_bb_sp @ Zn) / float(b)

            graph_loss_spatial = self.lambda_graph_spatial * lap_loss

        # total loss
        loss = recon_loss + self.beta * kl_norm
        if self.CCC_dim > 0 and self.ccc_L is not None:
            loss += self.lambda_graph_ccc * graph_loss_ccc
        loss += graph_loss_spatial

        return loss, recon_loss, kl_norm, graph_loss_ccc, graph_loss_spatial

    # ============================================================
    # Latent / denoise
    # ============================================================

    def batching_latent_samples(self, X, Y, num_samples=1, batch_size=512):
        self.eval()
        Y = torch.tensor(Y, dtype=self.dtype)
        latents = []
        for i in range(0, Y.shape[0], batch_size):
            yb = Y[i:i+batch_size].to(self.device)
            q_mu, q_var = self.encoder(yb)
            dist = Normal(q_mu, torch.sqrt(q_var + 1e-8))
            for _ in range(num_samples):
                latents.append(dist.rsample().detach().cpu())
        return torch.cat(latents, dim=0).numpy()

    def batching_denoise_counts(self, X, Y, n_samples=5, batch_size=512):
        self.eval()
        Y = torch.tensor(Y, dtype=self.dtype)
        outs = []

        for i in range(0, Y.shape[0], batch_size):
            yb = Y[i:i+batch_size].to(self.device)
            q_mu, q_var = self.encoder(yb)
            dist = Normal(q_mu, torch.sqrt(q_var + 1e-8))

            items = []
            for _ in range(n_samples):
                z = dist.rsample()
                h = self.decoder(z)
                m = self.dec_mean(h)
                items.append(m)

            outs.append(torch.stack(items, dim=0).mean(0).detach().cpu())

        return torch.cat(outs, dim=0).numpy()

    # ============================================================
    # Training
    # ============================================================

    def train_model(
        self,
        pos,
        ncounts,
        raw_counts,
        size_factors,
        lr=0.001,
        weight_decay=1e-6,
        batch_size=512,
        num_samples=1,
        train_size=0.95,
        maxiter=500,
        patience=50,
        model_weights="model.pt",
        save_latent_interval=None,
        latent_prefix="latent",
        output_prefix="cccvae_spatial",
        use_scheduler=False,
        scheduler_type="cosine",
        scheduler_step=10,
        scheduler_gamma=0.5,
    ):
        if sp.issparse(ncounts):
            ncounts = ncounts.toarray()
        if sp.issparse(raw_counts):
            raw_counts = raw_counts.toarray()

        pos = torch.tensor(pos, dtype=torch.int64)

        dataset = TensorDataset(
            pos,
            torch.tensor(ncounts, dtype=self.dtype),
            torch.tensor(raw_counts, dtype=self.dtype),
            torch.tensor(size_factors, dtype=self.dtype),
        )

        # train / val split
        if train_size < 1:
            n_total = len(dataset)
            n_train = int(train_size * n_total)
            n_val = n_total - n_train
            train_ds, valid_ds = random_split(dataset, [n_train, n_val])
            valid_loader = DataLoader(valid_ds, batch_size=batch_size)
        else:
            train_ds = dataset
            valid_loader = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        early_stop = EarlyStopping(patience=patience, modelfile=model_weights)

        opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None
        if use_scheduler:
            if scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(opt, T_max=maxiter)
            elif scheduler_type == "step":
                scheduler = StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma)
            elif scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(opt, factor=scheduler_gamma, patience=scheduler_step)

        log_dir = "logs"
        latent_dir = os.path.join("latent", output_prefix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)

        loss_f = open(os.path.join(log_dir, f"{output_prefix}_loss_log.txt"), "w")
        loss_f.write("epoch\tELBO\trecon\tKL\tgraph_ccc\tgraph_spatial\tbeta\tlambda_ccc\tlambda_spatial\n")

        n_cells = ncounts.shape[0]
        queue = deque(maxlen=10)

        print("Training...")

        for epoch in range(maxiter):
            self.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_graph_ccc = 0.0
            total_graph_spatial = 0.0
            N = 0

            for xb, yb, yrb, sf in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yrb = yrb.to(self.device)
                sf = sf.to(self.device)

                loss, rec, kl_val, graph_ccc_val, graph_sp_val = self.forward(
                    xb, yb, yrb, sf, num_samples=num_samples
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                bs = xb.shape[0]
                N += bs
                total_loss += loss.item()
                total_recon += rec.item()
                total_kl += kl_val.item()
                total_graph_ccc += float(graph_ccc_val)
                total_graph_spatial += float(graph_sp_val)

                # PID β
                if self.dynamicVAE:
                    KL_val = kl_val.item() / bs
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    target_KL = self.KL_loss * self.Normal_dim
                    self.beta, _ = self.PID.pid(target_KL, avg_KL)

            avg_loss = total_loss / N
            avg_recon = total_recon / N
            avg_kl = total_kl / N
            avg_graph_ccc = total_graph_ccc / N
            avg_graph_spatial = total_graph_spatial / N
            ELBO = -avg_loss

            print(
                f"Epoch {epoch+1}: "
                f"ELBO={ELBO:.4f}, "
                f"Recon={avg_recon:.4f}, "
                f"KL={avg_kl:.4f}, "
                f"Graph_CCC={avg_graph_ccc:.4f}, "
                f"Graph_Spatial={avg_graph_spatial:.4f}, "
                f"beta={self.beta:.4f}, "
                f"lambda_ccc={self.lambda_graph_ccc:.4f}, "
                f"lambda_spatial={self.lambda_graph_spatial:.4f}"
            )

            loss_f.write(
                f"{epoch+1}\t{ELBO}\t{avg_recon}\t{avg_kl}\t{avg_graph_ccc}\t"
                f"{avg_graph_spatial}\t{self.beta}\t{self.lambda_graph_ccc}\t{self.lambda_graph_spatial}\n"
            )
            loss_f.flush()

            # scheduler
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

            # CCC update (denoised → CCC Laplacian)
            if self.CCC_dim > 0 and self.ccc_builder is not None:
                if (epoch + 1) >= self.warmup_epochs and ((epoch + 1 - self.warmup_epochs) % self.ccc_update_interval == 0):
                    print(f"[CCC] Updating CCC graph at epoch {epoch+1}...")

                    denoised = self.batching_denoise_counts(
                        X=np.arange(n_cells),
                        Y=ncounts,
                        n_samples=5,
                        batch_size=batch_size,
                    )

                    L_new = self.ccc_builder(
                        expr=denoised,
                        genes=self.genes,
                        cutoff=self.ccc_cutoff,
                        top_k=self.ccc_top_k,
                        alpha=self.ccc_alpha,
                        Kh=self.ccc_Kh,
                        batch_size_pairs=self.ccc_batch_size_pairs,
                        device=str(self.device),
                        dtype="float32",
                        verbose=False,
                    )

                    self.update_ccc_laplacian(L_new)

                    # CCC λ schedule
                    if self.lambda_graph_ccc == 0.0:
                        self.lambda_graph_ccc = self.laplacian_lambda_start
                    else:
                        if ((epoch + 1 - self.warmup_epochs) % self.laplacian_lambda_update_interval) == 0:
                            self.lambda_graph_ccc = min(
                                self.lambda_graph_ccc + self.laplacian_lambda_step,
                                self.laplacian_lambda_max
                            )

            # validation
            if valid_loader is not None:
                self.eval()
                val_loss = 0.0
                N_val = 0
                with torch.no_grad():
                    for xb, yb, yrb, sf in valid_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        yrb = yrb.to(self.device)
                        sf = sf.to(self.device)

                        loss, _, _, _, _ = self.forward(xb, yb, yrb, sf, num_samples=1)
                        val_loss += loss.item()
                        N_val += xb.shape[0]

                val_loss /= N_val
                val_ELBO = -val_loss
                print(f"  Validation ELBO = {val_ELBO:.4f}")

                early_stop(val_loss, self)
                if early_stop.early_stop:
                    print("Early stopping triggered.")
                    break

            # save latent snapshots
            if (save_latent_interval is not None) and ((epoch + 1) % save_latent_interval == 0):
                latent_np = self.batching_latent_samples(
                    X=pos.cpu().numpy(),
                    Y=ncounts,
                    num_samples=1,
                    batch_size=batch_size,
                )
                np.savetxt(
                    os.path.join(latent_dir, f"{latent_prefix}_epoch_{epoch+1}.txt"),
                    latent_np,
                    delimiter=",",
                )

        loss_f.close()
        torch.save(self.state_dict(), model_weights)

