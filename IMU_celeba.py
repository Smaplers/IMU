import os
import random
from copy import deepcopy
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import clip



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 20
EPOCHS = 70
LR = 1e-5


# =====================================================================================
LAMBDA_RS = 1.5
LAMBDA_REPEL = 2.2
LAMBDA_MIX = 0.6
LAMBDA_RT = 0.03
K_CLUSTERS = 2

# Paths
FORGET_DIR = "../data_CelebA/forget_identity"
ATTR_DIR = "../data_CelebA/attribute_matched"
RETAIN_DIR = "../data_CelebA/retain"

FT_VISUAL_PATH = r"./outputs_celeba/clip_celeba__visual.pth"
OUT_PATH = r"./MU_celeba\IMU\IMU.pth"


# -----------------------------
DISTILL_LAYERS = [3, 7]
RT_FINAL_RATIO = 0.3

MIX_K = 16
MIX_TAU = 0.3
MIX_DETACH_CENTER = True

TOP_M_ATTR = 64

BANK_MOMENTUM = 0.8
BANK_NEG_SAMPLES = 512
BANK_TOPK_HARD = 256
BANK_MARGIN = 0.25

KMEANS_ITERS = 25
KMEANS_RESTARTS = 2
ID_END_SCALE = 0.5

NUM_WORKERS = 0



# -----------------------------

def repel_total_scale(step, total_steps):
    warmup = int(0 * total_steps)
    ramp = int(0.2 * total_steps)
    return ramp01(step, warmup, ramp)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(v, eps=1e-6):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def cosine_preserve_loss(v, v_ref):
    v = normalize(v.float())
    v_ref = normalize(v_ref.float())
    return (1 - (v * v_ref).sum(dim=1)).mean()


def cosine_push_loss(v, r):
    v = normalize(v.float())
    r = normalize(r.float()).to(v.device)
    if r.shape[0] == 1:
        return torch.abs(v @ r.T).mean()
    return torch.abs((v * r).sum(dim=1)).mean()


def identity_repulsion_loss(v, margin=0.5):
    v = normalize(v.float())
    sim = v @ v.T
    mask = ~torch.eye(len(v), device=v.device).bool()
    sim = sim[mask]
    return torch.relu(sim - margin).mean()


def ramp01(step, start_step, ramp_steps):
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return float(max(0.0, min(1.0, (step - start_step) / float(ramp_steps))))


def id_mix_scales(step, total_steps, id_end_scale=0.5):
    warmup = int(0.20 * total_steps)
    ramp = int(0.50 * total_steps)
    t = ramp01(step, warmup, ramp)
    mix_scale = t
    rs_scale = 1.0 - (1.0 - float(id_end_scale)) * t
    return rs_scale, mix_scale


def repulsion_scales(step, total_steps):
    warmup = int(0.05 * total_steps)
    ramp = int(0.25 * total_steps)
    t = ramp01(step, warmup, ramp)
    return 1.0 - t, t


class ImageFolderWithIndex(ImageFolder):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index


@torch.no_grad()
def kmeans_cosine(X, K, iters=20, restarts=1):
    X = normalize(X.float())
    N, D = X.shape
    K = int(max(1, min(K, N)))

    best_labels = None
    best_centroids = None
    best_obj = -1e9

    for _ in range(restarts):
        perm = torch.randperm(N, device=X.device)
        centroids = X[perm[:K]].clone()

        for _ in range(iters):
            sim = X @ centroids.T
            labels = sim.argmax(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    new_centroids[k] = X[torch.randint(0, N, (1,), device=X.device)]
            centroids = normalize(new_centroids)

        obj = (X * centroids[labels]).sum(dim=1).mean().item()
        if obj > best_obj:
            best_obj = obj
            best_labels = labels.clone()
            best_centroids = centroids.clone()

    return best_labels, best_centroids


@torch.no_grad()
def update_bank(bank, bank_filled, idx, vec, momentum=0.8):
    idx = idx.long()
    old = bank[idx]
    new = normalize(old * momentum + vec * (1 - momentum))
    bank[idx] = new
    bank_filled[idx] = True


def global_bank_repulsion_loss(v_f, idx_f, bank, bank_filled,
                               neg_samples=512, topk_hard=128,
                               margin=0.5, eps=1e-6):
    vf = normalize(v_f.float(), eps=eps)

    filled_idx = torch.nonzero(bank_filled, as_tuple=False).squeeze(1)
    if filled_idx.numel() < 2:
        return vf.new_tensor(0.0)

    n = filled_idx.numel()
    m = int(max(1, min(neg_samples, n)))
    sample = filled_idx[torch.randint(0, n, (m,), device=filled_idx.device)]
    neg = bank[sample]

    idx_f = idx_f.to(sample.device).long()
    same = idx_f[:, None] == sample[None, :]

    sims = vf @ neg.T
    sims = sims.masked_fill(same, -1.0)

    k = int(max(1, min(topk_hard, sims.shape[1])))
    hard_sims = torch.topk(sims, k=k, dim=1, largest=True).values
    return torch.relu(hard_sims - margin).mean()


def neighbor_center_loss(v_f, v_a, k=16, tau=0.35, detach_center=True, eps=1e-6):
    vf = normalize(v_f.float(), eps=eps)
    va = normalize(v_a.float(), eps=eps)

    sims = vf @ va.T
    k = int(max(1, min(k, sims.shape[1])))
    topk_idx = torch.topk(sims, k=k, dim=1, largest=True).indices
    center = va[topk_idx].mean(dim=1)
    center = normalize(center, eps=eps)

    if detach_center:
        center = center.detach()

    cos = (vf * center).sum(dim=1)
    return torch.relu(tau - cos).mean()


def load_celeba_adapted_visual(model, path, device):
    if not path or not os.path.isfile(path):
        print("[WARN] FT_VISUAL_PATH not found. Using original pretrained CLIP visual encoder.")
        return

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[Load] full CLIP checkpoint loaded from: {path}")
    else:
        model.visual.load_state_dict(ckpt, strict=True)
        print(f"[Load] visual-only checkpoint loaded from: {path}")


@torch.no_grad()
def encode_dataset(encoder, dataset, batch_size, device, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    embeds = []
    indices = []
    for batch in tqdm(loader, desc=desc):
        if len(batch) == 3:
            x, _, idx = batch
            indices.append(idx)
        else:
            x, _ = batch
        embeds.append(encoder(x.to(device)).float())
    embeds = torch.cat(embeds, dim=0)
    if indices:
        indices = torch.cat(indices, dim=0).long()
        return embeds, indices
    return embeds, None


def build_cluster_residuals(image_encoder_orig, forget_ds, attr_ds):
    print("Computing cluster residual directions for RS...")

    A, _ = encode_dataset(image_encoder_orig, attr_ds, BATCH_SIZE, DEVICE, "Attr embed (teacher)")
    A_norm = normalize(A)
    Na = A.shape[0]
    mu_attr_global = A.mean(0, keepdim=True)
    print(f"[Attr] Na={Na}")

    F, IDX = encode_dataset(image_encoder_orig, forget_ds, BATCH_SIZE, DEVICE, "Forget embed (teacher)")
    Nf = F.shape[0]
    K = int(max(1, min(K_CLUSTERS, Nf)))
    print(f"[RS] N_forget={Nf}, K={K}")

    labels, _ = kmeans_cosine(F, K=K, iters=KMEANS_ITERS, restarts=KMEANS_RESTARTS)
    sizes = [(labels == k).sum().item() for k in range(K)]
    print("[RS] cluster sizes:", sizes)

    r_multi = []
    for k in range(K):
        mask = labels == k
        mu_k = F[mask].mean(0, keepdim=True) if mask.any() else F.mean(0, keepdim=True)

        sims = (normalize(mu_k) @ A_norm.T).squeeze(0)
        m = int(max(1, min(TOP_M_ATTR, Na)))
        top_idx = torch.topk(sims, k=m, largest=True).indices
        mu_attr_k = A[top_idx].mean(0, keepdim=True)

        r_k = normalize(mu_k - mu_attr_k)
        r_multi.append(r_k)

    r_multi = torch.cat(r_multi, dim=0).to(DEVICE)

    cluster_id_map = torch.empty(len(forget_ds), dtype=torch.long)
    cluster_id_map[IDX.cpu()] = labels.cpu()

    r_global = normalize(F.mean(0, keepdim=True) - mu_attr_global.to(F.device)).to(DEVICE)
    return r_multi, cluster_id_map, r_global



# -----------------------------
def main():
    set_seed(SEED)

    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA版本: {torch.version.cuda}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")

    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.float()
    load_celeba_adapted_visual(model, FT_VISUAL_PATH, DEVICE)
    model.train()

    image_encoder = model.visual

    # Teacher is the CelebA-adapted encoder before MU.
    image_encoder_orig = deepcopy(image_encoder).eval()
    for p in image_encoder_orig.parameters():
        p.requires_grad = False

    # Data
    forget_ds = ImageFolderWithIndex(FORGET_DIR, transform=preprocess)
    attr_ds = ImageFolder(ATTR_DIR, transform=preprocess)
    retain_ds = ImageFolder(RETAIN_DIR, transform=preprocess)

    # Mix 使用的非目标邻域：attribute-matched + retain
    # 注意：forget 不放进去，避免同化回目标实例本身
    non_target_ds = ConcatDataset([attr_ds, retain_ds])

    forget_loader = DataLoader(
        forget_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # attr_loader 仍然保留，RS 构建 residual direction 时使用 attr_ds
    attr_loader = DataLoader(
        attr_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    retain_loader = DataLoader(
        retain_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    non_target_loader = DataLoader(
        non_target_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    total_steps = EPOCHS * len(forget_loader)
    print(f"[Schedule] total_steps={total_steps}")

    # Memory bank
    with torch.no_grad():
        tmp_x, _, _ = next(iter(DataLoader(forget_ds, batch_size=1, shuffle=False)))
        tmp_v = image_encoder_orig(tmp_x.to(DEVICE)).float()
        embed_dim = tmp_v.shape[-1]
    print(f"[Bank] N_forget={len(forget_ds)}, D={embed_dim}")
    forget_bank = torch.zeros((len(forget_ds), embed_dim), device=DEVICE)
    forget_bank_filled = torch.zeros((len(forget_ds),), dtype=torch.bool, device=DEVICE)

    # Residual directions
    r_id_multi, cluster_id_map, r_id_global = build_cluster_residuals(
        image_encoder_orig, forget_ds, attr_ds
    )

    # Layer distillation hooks
    student_acts = {}
    teacher_acts = {}

    def make_hook(store, key):
        def _hook(module, inp, out):
            store[key] = out
        return _hook

    hooks = []
    for li in DISTILL_LAYERS:
        hooks.append(image_encoder.transformer.resblocks[li].register_forward_hook(make_hook(student_acts, li)))
        hooks.append(image_encoder_orig.transformer.resblocks[li].register_forward_hook(make_hook(teacher_acts, li)))

    def layer_distill_loss():
        losses = []
        for li in DISTILL_LAYERS:
            cls_s = student_acts[li][0].float()
            cls_t = teacher_acts[li][0].float()
            losses.append(cosine_preserve_loss(cls_s, cls_t))
        return torch.stack(losses).mean()

    optimizer = optim.Adam(image_encoder.parameters(), lr=LR)

    use_amp = DEVICE == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("Starting IMU training...")
    retain_iter = iter(retain_loader)
    non_target_iter = iter(non_target_loader)
    global_step = 0

    for epoch in range(EPOCHS):
        for x_f, _, idx_f in tqdm(forget_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            try:
                x_r, _ = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x_r, _ = next(retain_iter)

            try:
                x_n, _ = next(non_target_iter)
            except StopIteration:
                non_target_iter = iter(non_target_loader)
                x_n, _ = next(non_target_iter)

            x_f = x_f.to(DEVICE)
            x_r = x_r.to(DEVICE)
            x_n = x_n.to(DEVICE)
            idx_f_dev = idx_f.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            rs_scale, mix_scale = id_mix_scales(global_step, total_steps, ID_END_SCALE)
            lambda_rs_eff = LAMBDA_RS * rs_scale
            lambda_mix_eff = LAMBDA_MIX * mix_scale

            with autocast_ctx():
                # Forget forward
                v_f = image_encoder(x_f)

                # Retain forward with hooks
                student_acts.clear()
                teacher_acts.clear()
                v_r = image_encoder(x_r)
                with torch.no_grad():
                    v_r_orig = image_encoder_orig(x_r)

                loss_retain_final = cosine_preserve_loss(v_r, v_r_orig)
                loss_retain_layers = layer_distill_loss()
                loss_rt = RT_FINAL_RATIO * loss_retain_final + loss_retain_layers

                # RS: suppress cluster-specific residual direction
                cids = cluster_id_map[idx_f.cpu()].to(DEVICE)
                r_batch = r_id_multi[cids] if r_id_multi is not None else r_id_global
                loss_rs = cosine_push_loss(v_f, r_batch)

                # CR-1: forget internal repulsion
                loss_repel_batch = identity_repulsion_loss(v_f, margin=BANK_MARGIN)

                with torch.no_grad():
                    update_bank(
                        forget_bank,
                        forget_bank_filled,
                        idx_f_dev,
                        normalize(v_f.detach().float()),
                        momentum=BANK_MOMENTUM,
                    )

                loss_repel_bank = global_bank_repulsion_loss(
                    v_f,
                    idx_f_dev,
                    forget_bank,
                    forget_bank_filled,
                    neg_samples=BANK_NEG_SAMPLES,
                    topk_hard=BANK_TOPK_HARD,
                    margin=BANK_MARGIN,
                )

                w_batch, w_bank = repulsion_scales(global_step, total_steps)
                loss_repel = w_batch * loss_repel_batch + w_bank * loss_repel_bank

                # CR-2: local neighborhood assimilation
                # Mix 到当前训练过程中最近的非目标邻域
                v_n = image_encoder(x_n)
                loss_mix = neighbor_center_loss(
                    v_f,
                    v_n,
                    k=MIX_K,
                    tau=MIX_TAU,
                    detach_center=MIX_DETACH_CENTER,
                )

                repel_scale = repel_total_scale(global_step, total_steps)

                loss = (
                        lambda_rs_eff * loss_rs
                        + LAMBDA_REPEL * repel_scale * loss_repel
                        + lambda_mix_eff * loss_mix
                        + LAMBDA_RT * loss_rt
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        filled_ratio = forget_bank_filled.float().mean().item()
        w_batch, w_bank = repulsion_scales(global_step, total_steps)
        rs_scale, mix_scale = id_mix_scales(global_step, total_steps, ID_END_SCALE)
        print(
            f"Epoch {epoch + 1} | "
            f"Loss={loss.item():.6f} | "
            f"RS={loss_rs.item():.4f} (lambda={LAMBDA_RS * rs_scale:.3f}) | "
            f"Repel={loss_repel.item():.4f} (wb={w_batch:.2f}, wk={w_bank:.2f}) | "
            f"Mix={loss_mix.item():.4f} (lambda={LAMBDA_MIX * mix_scale:.3f}) | "
            f"RT={loss_rt.item():.4f} | "
            f"bank={filled_ratio:.2%}"
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(image_encoder.state_dict(), OUT_PATH)
    print(f"MU model saved to {OUT_PATH}")

    for h in hooks:
        h.remove()

    print("Done.")


if __name__ == "__main__":
    main()
