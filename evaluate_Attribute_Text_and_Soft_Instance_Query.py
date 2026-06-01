

import os
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip


# -----------------------------
# Utils
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")



def load_visual_encoder_into_clip(model, visual_path: str, device: str, name: str = "MODEL"):
    """
    Load a CelebA-adapted visual encoder into a CLIP model.

    Supported formats:
      1) pure visual state_dict: torch.save(model.visual.state_dict(), path)
      2) full CLIP checkpoint: {"model_state_dict": model.state_dict(), ...}
      3) wrapper checkpoint: {"state_dict": model.state_dict(), ...}

    If visual_path is empty, the model remains vanilla pretrained CLIP.
    """
    if visual_path is None or str(visual_path).strip() == "":
        print(f"[{name}-LOAD] No visual checkpoint provided. Use vanilla pretrained CLIP.")
        return

    if not os.path.exists(visual_path):
        raise FileNotFoundError(f"[{name}-LOAD] checkpoint not found: {visual_path}")

    ckpt = torch.load(visual_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[{name}-LOAD] full CLIP checkpoint loaded from ckpt['model_state_dict']: {visual_path}")
        return

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(
            f"[{name}-LOAD] wrapper checkpoint loaded from ckpt['state_dict'] "
            f"(missing={len(missing)}, unexpected={len(unexpected)}): {visual_path}"
        )
        return

    if isinstance(ckpt, dict):
        try:
            model.visual.load_state_dict(ckpt, strict=True)
            print(f"[{name}-LOAD] visual-only checkpoint loaded into model.visual: {visual_path}")
            return
        except RuntimeError as e_visual:
            # Fallback: maybe it is a full model state_dict without wrapper.
            try:
                missing, unexpected = model.load_state_dict(ckpt, strict=False)
                print(
                    f"[{name}-LOAD] raw dict loaded into full CLIP model "
                    f"(missing={len(missing)}, unexpected={len(unexpected)}): {visual_path}"
                )
                return
            except RuntimeError as e_full:
                raise RuntimeError(
                    f"[{name}-LOAD] failed to load checkpoint as visual-only or full model.\n"
                    f"visual error: {e_visual}\nfull error: {e_full}"
                )

    raise RuntimeError(f"[{name}-LOAD] unsupported checkpoint type: {type(ckpt)} from {visual_path}")


def load_mu_model_into_clip(model_mu, mu_path: str, device: str):
    """
    Load an MU checkpoint into CLIP.

    Supported formats:
      1) RUCLIP/full wrapper: {"state_dict": student.state_dict(), ...}
      2) full CLIP checkpoint: {"model_state_dict": model.state_dict(), ...}
      3) visual-only state_dict: torch.save(model.visual.state_dict(), path)
      4) raw full model state_dict
    """
    if not os.path.exists(mu_path):
        raise FileNotFoundError(f"[MU-LOAD] checkpoint not found: {mu_path}")

    ckpt = torch.load(mu_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_mu.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[MU-LOAD] full CLIP checkpoint loaded from ckpt['model_state_dict']: {mu_path}")
        return

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        missing, unexpected = model_mu.load_state_dict(sd, strict=False)
        print(
            f"[MU-LOAD] wrapper/full checkpoint loaded from ckpt['state_dict'] "
            f"(missing={len(missing)}, unexpected={len(unexpected)}): {mu_path}"
        )
        return

    if isinstance(ckpt, dict):
        try:
            model_mu.visual.load_state_dict(ckpt, strict=True)
            print(f"[MU-LOAD] visual-only checkpoint loaded into model_mu.visual: {mu_path}")
            return
        except RuntimeError:
            missing, unexpected = model_mu.load_state_dict(ckpt, strict=False)
            print(
                f"[MU-LOAD] raw dict loaded into full CLIP model "
                f"(missing={len(missing)}, unexpected={len(unexpected)}): {mu_path}"
            )
            return

    raise RuntimeError(f"[MU-LOAD] unsupported checkpoint type: {type(ckpt)} from {mu_path}")


def list_images(root: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    paths = sorted(list(set(paths)))
    return paths


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def read_lines(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def sample_without_bias(paths: List[str], max_n: int, rng: np.random.Generator) -> List[str]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    idx = rng.choice(len(paths), size=max_n, replace=False)
    return [paths[i] for i in idx]



# -----------------------------
def load_celeba_attr_map(attr_file: str):
    with open(attr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    attr_names = lines[1].split()
    attr_idx = {a: i for i, a in enumerate(attr_names)}

    attr_map = {}
    for line in lines[2:]:
        parts = line.split()
        fn = parts[0]
        vals = np.array(list(map(int, parts[1:])), dtype=np.int8)
        attr_map[fn] = vals

    return attr_idx, attr_map


def make_attr_mask_for_index(
    index_paths: List[str],
    attr_name: str,
    attr_idx: Dict[str, int],
    attr_map: Dict[str, np.ndarray],
) -> np.ndarray:
    if attr_name not in attr_idx:
        raise ValueError(f"Attribute '{attr_name}' not found in list_attr_celeba.txt")
    col = attr_idx[attr_name]

    mask = np.zeros(len(index_paths), dtype=bool)
    missing = 0
    for i, p in enumerate(index_paths):
        fn = os.path.basename(p)
        if fn not in attr_map:
            missing += 1
            continue
        mask[i] = (attr_map[fn][col] == 1)

    if missing > 0:
        print(f"[WARN] Attr '{attr_name}': missing annotations for {missing}/{len(index_paths)} index images.")
    return mask


def make_attr_pair_mask_for_index(
    index_paths: List[str],
    a: str,
    b: str,
    attr_idx: Dict[str, int],
    attr_map: Dict[str, np.ndarray],
) -> np.ndarray:
    ma = make_attr_mask_for_index(index_paths, a, attr_idx, attr_map)
    mb = make_attr_mask_for_index(index_paths, b, attr_idx, attr_map)
    return ma & mb


def make_attr_intersection_mask_for_index(
    index_paths: List[str],
    attrs: List[str],
    attr_idx: Dict[str, int],
    attr_map: Dict[str, np.ndarray],
) -> np.ndarray:
    if len(attrs) == 0:
        return np.ones(len(index_paths), dtype=bool)
    mask = np.ones(len(index_paths), dtype=bool)
    for a in attrs:
        mask &= make_attr_mask_for_index(index_paths, a, attr_idx, attr_map)
    return mask



# -----------------------------
@dataclass
class Sample:
    path: str
    group: str  # "id:person_A", "id:others", "pool:retain", "pool:attr_matched", ...


class ImageIndexDataset(Dataset):
    def __init__(self, samples: List[Sample], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.preprocess(img)
        return x, s.group, s.path


@torch.no_grad()
def encode_images(model, loader, device: str) -> Tuple[torch.Tensor, List[str], List[str]]:
    feats = []
    groups, paths = [], []
    for x, g, p in tqdm(loader, desc="Encoding images", leave=False):
        x = x.to(device)
        f = model.encode_image(x).float()
        f = normalize(f).cpu()
        feats.append(f)
        groups.extend(list(g))
        paths.extend(list(p))
    feats = torch.cat(feats, dim=0)
    return feats, groups, paths


@torch.no_grad()
def encode_texts(model, texts: List[str], device: str) -> torch.Tensor:
    tokens = clip.tokenize(texts).to(device)
    t = model.encode_text(tokens).float()
    t = normalize(t).cpu()
    return t



# -----------------------------
def per_query_metrics(
    sims: np.ndarray,          # [N] float
    target_mask: np.ndarray,   # [N] bool
    ks: List[int],
) -> Dict[str, float]:
    """
    Compute hit@K, recall@K for multiple K, plus first-rank, MRR.
    """
    N = sims.shape[0]
    order = np.argsort(-sims)  # descending

    total_t = int(target_mask.sum())
    first_rank = float("inf")
    for r, idx in enumerate(order, start=1):
        if target_mask[idx]:
            first_rank = float(r)
            break

    mrr = 0.0 if not np.isfinite(first_rank) else 1.0 / first_rank

    out = {
        "num_targets": float(total_t),
        "first_rank": first_rank,
        "mrr": float(mrr),
    }

    for k in ks:
        kk = min(int(k), N)
        top_idx = order[:kk]
        hit = float(target_mask[top_idx].any())
        recall = 0.0 if total_t == 0 else float(target_mask[top_idx].sum() / total_t)
        out[f"hit@{k}"] = hit
        out[f"recall@{k}"] = recall

    return out


def aggregate_metrics(metrics_list: List[Dict[str, float]], ks: List[int]) -> Dict[str, float]:
    if len(metrics_list) == 0:
        return {}

    num_targets = int(metrics_list[0]["num_targets"])

    first_ranks = [m["first_rank"] for m in metrics_list]
    finite_ranks = [r for r in first_ranks if np.isfinite(r)]

    agg = {
        "num_targets": num_targets,
        "MRR": float(np.mean([m["mrr"] for m in metrics_list])),
        "median_rank": float(np.median(finite_ranks)) if len(finite_ranks) else float("inf"),
        "mean_rank": float(np.mean(finite_ranks)) if len(finite_ranks) else float("inf"),
    }

    for k in ks:
        agg[f"hit@{k}"] = float(np.mean([m[f"hit@{k}"] for m in metrics_list]))
        agg[f"recall@{k}"] = float(np.mean([m[f"recall@{k}"] for m in metrics_list]))

    return agg


def fmt(x: float, nd=6) -> str:
    if x is None:
        return ""
    if np.isinf(x):
        return "inf"
    if np.isnan(x):
        return "nan"
    return f"{float(x):.{nd}f}"


def print_side_by_side_table(
    title: str,
    rows: List[Tuple[str, Dict[str, float], Dict[str, float]]],
    ks: List[int],
):
    hit_cols = [f"hit@{k}" for k in ks]
    rec_cols = [f"recall@{k}" for k in ks]
    cols = ["Query", "#targets"] + \
           [f"O:{c}" for c in hit_cols] + [f"O:{c}" for c in rec_cols] + ["O:MRR", "O:median", "O:mean"] + \
           [f"M:{c}" for c in hit_cols] + [f"M:{c}" for c in rec_cols] + ["M:MRR", "M:median", "M:mean"]

    def col_val(q, mo, mm, col):
        if col == "Query":
            return q
        if col == "#targets":
            return str(int(mo.get("num_targets", mm.get("num_targets", 0))))
        if col.startswith("O:"):
            key = col[2:]
            if key == "MRR":
                return fmt(mo.get("MRR", float("nan")))
            if key == "median":
                return fmt(mo.get("median_rank", float("nan")), nd=1)
            if key == "mean":
                return fmt(mo.get("mean_rank", float("nan")), nd=1)
            return fmt(mo.get(key, float("nan")))
        if col.startswith("M:"):
            key = col[2:]
            if key == "MRR":
                return fmt(mm.get("MRR", float("nan")))
            if key == "median":
                return fmt(mm.get("median_rank", float("nan")), nd=1)
            if key == "mean":
                return fmt(mm.get("mean_rank", float("nan")), nd=1)
            return fmt(mm.get(key, float("nan")))
        return ""

    table = [cols]
    for (q, mo, mm) in rows:
        table.append([col_val(q, mo, mm, c) for c in cols])

    widths = [max(len(str(table[r][c])) for r in range(len(table))) for c in range(len(cols))]

    def render_row(vals):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(render_row(table[0]))
    print(sep)
    for r in range(1, len(table)):
        print(render_row(table[r]))
    print("")



# -----------------------------
def build_samples(
    retain_dir: str,
    forget_personA_dir: str,
    attr_matched_dir: str,
    eval_personA_dir: str,
    eval_others_dir: str,
    max_retain_images: int,
    max_attr_matched_images: int,
    seed: int,
) -> List[Sample]:
    rng = np.random.default_rng(seed)
    samples: List[Sample] = []

    # person_A (train/proto + unseen)
    for p in list_images(forget_personA_dir):
        samples.append(Sample(p, "id:person_A"))
    if eval_personA_dir and os.path.isdir(eval_personA_dir):
        for p in list_images(eval_personA_dir):
            samples.append(Sample(p, "id:person_A"))

    # others negatives
    for p in list_images(eval_others_dir):
        samples.append(Sample(p, "id:others"))

    # retain pool
    if retain_dir and os.path.isdir(retain_dir):
        imgs = list_images(retain_dir)
        imgs = sample_without_bias(imgs, max_retain_images, rng)
        for p in imgs:
            samples.append(Sample(p, "pool:retain"))

    # attr_matched pool
    if attr_matched_dir and os.path.isdir(attr_matched_dir):
        imgs = list_images(attr_matched_dir)
        imgs = sample_without_bias(imgs, max_attr_matched_images, rng)
        for p in imgs:
            samples.append(Sample(p, "pool:attr_matched"))

    if len(samples) == 0:
        raise RuntimeError("No images found. Please check your directory paths.")
    return samples



# -----------------------------
@torch.no_grad()
def encode_paths_as_feats(model, preprocess, device: str, paths: List[str], batch_size: int, num_workers: int) -> torch.Tensor:
    items = [Sample(p, "tmp") for p in paths]
    ds = ImageIndexDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    for x, _, _ in tqdm(dl, desc="Encoding images", leave=False):
        x = x.to(device)
        f = model.encode_image(x).float()
        f = normalize(f).cpu()
        feats.append(f)
    return torch.cat(feats, dim=0)


def learn_soft_identity_query(
    model_orig,
    preprocess,
    device: str,
    pos_paths: List[str],
    neg_paths: List[str],
    batch_size: int,
    num_workers: int,
    steps: int = 600,
    lr: float = 0.1,
    tau: float = 0.07,
    neg_per_step: int = 512,
    seed: int = 42,
) -> torch.Tensor:

    rng = np.random.default_rng(seed)

    print("[Exp2] Encoding positives (proto split person_A) for soft identity query...")
    pos_feats = encode_paths_as_feats(model_orig, preprocess, device, pos_paths, batch_size, num_workers)  # [P,D]
    print("[Exp2] Encoding negatives for soft identity query...")
    neg_feats = encode_paths_as_feats(model_orig, preprocess, device, neg_paths, batch_size, num_workers)  # [N,D]

    P, D = pos_feats.shape
    N = neg_feats.shape[0]
    print(f"[Exp2] Optimizing soft identity query: P={P}, N={N}, steps={steps}, neg_per_step={neg_per_step}")

    q0 = normalize(pos_feats.mean(dim=0, keepdim=True)).clone()  # [1,D]
    q = q0.to(device).requires_grad_(True)

    pos_feats_d = pos_feats.to(device)
    neg_feats_d = neg_feats.to(device)

    opt = torch.optim.Adam([q], lr=lr)

    for _ in tqdm(range(steps), desc="[Exp2] Learn soft ID query", leave=False):
        opt.zero_grad(set_to_none=True)

        if N > neg_per_step:
            idx = torch.from_numpy(rng.integers(0, N, size=(neg_per_step,), dtype=np.int64)).to(device)
            neg_batch = neg_feats_d[idx]
        else:
            neg_batch = neg_feats_d

        qn = normalize(q)

        pos_s = (qn @ pos_feats_d.T).squeeze(0) / tau
        neg_s = (qn @ neg_batch.T).squeeze(0) / tau

        loss_pos = F.softplus(-pos_s).mean()
        loss_neg = F.softplus(neg_s).mean()
        loss = loss_pos + loss_neg

        loss.backward()
        opt.step()

        with torch.no_grad():
            q[:] = normalize(q)

    q_cpu = normalize(q.detach().cpu())
    print("[Exp2] Done. Soft identity query embedding learned (fixed).")
    return q_cpu



# -----------------------------
PHRASE = {
    "Male": "male",
    "Young": "young",
    "Smiling": "smiling",
    "Eyeglasses": "wearing glasses",
    "Bald": "bald",
    "Bangs": "with bangs",
    "Black_Hair": "with black hair",
    "Blond_Hair": "with blond hair",
    "Brown_Hair": "with brown hair",
    "Gray_Hair": "with gray hair",
    "Wavy_Hair": "with wavy hair",
    "Straight_Hair": "with straight hair",
    "Receding_Hairline": "with a receding hairline",
    "Sideburns": "with sideburns",
    "Mustache": "with a mustache",
    "No_Beard": "clean-shaven",
    "Goatee": "with a goatee",
    "5_o_Clock_Shadow": "with stubble",
    "Wearing_Hat": "wearing a hat",
    "Wearing_Lipstick": "wearing lipstick",
    "Wearing_Necklace": "wearing a necklace",
    "Wearing_Earrings": "wearing earrings",
}

EXCLUDE = {
    "Attractive", "Blurry", "Pale_Skin", "Oval_Face", "High_Cheekbones",
    "Pointy_Nose", "Big_Lips", "Big_Nose", "Chubby", "Double_Chin",
    "Heavy_Makeup", "Rosy_Cheeks", "Mouth_Slightly_Open",
    "Narrow_Eyes", "Bags_Under_Eyes", "Arched_Eyebrows", "Bushy_Eyebrows",
}

PRIORITY_ATTRS = [
    "Male", "Young", "Eyeglasses", "Smiling",
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair",
    "Bangs", "Bald", "Mustache", "No_Beard", "Goatee", "5_o_Clock_Shadow",
    "Wearing_Hat", "Sideburns", "Receding_Hairline",
    "Wearing_Lipstick", "Wearing_Earrings", "Wearing_Necklace",
    "Wavy_Hair", "Straight_Hair",
]


def stable_attrs_from_personA(
    personA_paths: List[str],
    attr_idx: Dict[str, int],
    attr_map: Dict[str, np.ndarray],
    min_freq: float = 0.6,
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    cnt = {a: 0 for a in attr_idx.keys()}
    used = 0
    missing = 0

    for p in personA_paths:
        fn = os.path.basename(p)
        if fn not in attr_map:
            missing += 1
            continue
        used += 1
        vals = attr_map[fn]
        for a, col in attr_idx.items():
            if vals[col] == 1:
                cnt[a] += 1

    denom = max(1, used)
    freq = {a: cnt[a] / denom for a in cnt.keys()}

    stable = []
    for a, fr in sorted(freq.items(), key=lambda x: -x[1]):
        if a in EXCLUDE:
            continue
        if a not in PHRASE:
            continue
        if fr >= min_freq:
            stable.append((a, fr))

    if missing > 0:
        print(f"[Exp3] WARN: missing CelebA attr annotations for {missing}/{len(personA_paths)} person_A images.")
    return stable, freq


def pick_attrs_for_prompts(stable: List[Tuple[str, float]], freq: Dict[str, float], max_attrs: int = 10) -> List[str]:
    stable_set = {a for a, _ in stable}
    picked = []
    for a in PRIORITY_ATTRS:
        if a in stable_set and a in PHRASE and a not in EXCLUDE:
            picked.append(a)
        if len(picked) >= max_attrs:
            break

    if len(picked) < max_attrs:
        for a, fr in sorted(freq.items(), key=lambda x: -x[1]):
            if a in picked:
                continue
            if a in EXCLUDE or a not in PHRASE:
                continue
            picked.append(a)
            if len(picked) >= max_attrs:
                break
    return picked


def compose_prompt(attr_subset: List[str], rng: np.random.Generator) -> str:
    s = set(attr_subset)

    gender = "person"
    if "Male" in s:
        gender = "man"
    elif "Wearing_Lipstick" in s:
        gender = "woman"

    words = ["a photo of a"]

    if "Young" in s:
        words.append("young")
    words.append(gender)

    if "Eyeglasses" in s:
        words.append(PHRASE["Eyeglasses"])
    if "Smiling" in s:
        words.append(PHRASE["Smiling"])

    hair_colors = [a for a in s if a in {"Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"}]
    if len(hair_colors) > 1:
        keep = hair_colors[int(rng.integers(0, len(hair_colors)))]
        for a in hair_colors:
            if a != keep:
                s.remove(a)

    rest = [a for a in s if a not in {"Male", "Young", "Eyeglasses", "Smiling"}]
    rng.shuffle(rest)

    # Keep all selected stable attributes instead of truncating to 3.
    for a in rest:
        words.append(PHRASE[a])

    return " ".join(words)


def generate_combo_prompts_from_stable(
    personA_paths: List[str],
    attr_idx: Dict[str, int],
    attr_map: Dict[str, np.ndarray],
    n_prompts: int = 200,
    min_freq: float = 0.6,
    max_attrs: int = 10,
    seed: int = 42,
) -> Tuple[List[str], List[Tuple[str, float]], List[str]]:
    rng = np.random.default_rng(seed)

    stable, freq = stable_attrs_from_personA(personA_paths, attr_idx, attr_map, min_freq=min_freq)
    picked = pick_attrs_for_prompts(stable, freq, max_attrs=max_attrs)

    print("[Exp3] Top stable attrs (freq>=%.2f):" % min_freq)
    for a, fr in stable[:15]:
        print(f"  {a:20s}  freq={fr:.2f}")
    print("[Exp3] Picked attrs for prompt generation:", ", ".join(picked))

    prompts = set()
    base = [a for a in picked if a in {"Male", "Young", "Eyeglasses", "Smiling"}]

    for _ in range(5000):
        # Use up to 8 stable attributes for each composed query.
        # If fewer than 8 are available, use all available picked attributes.
        k = min(max_attrs, len(picked))
        subset = list(rng.choice(picked, size=k, replace=False))

        # Always keep basic demographic / expression attributes if available.
        subset = list(dict.fromkeys(subset + base))

        p = compose_prompt(subset, rng)
        prompts.add(p)
        if len(prompts) >= n_prompts:
            break

    prompts = sorted(list(prompts))
    print(f"[Exp3] Generated combo prompts: {len(prompts)}")
    return prompts, stable, picked


# -----------------------------
# Experiments helpers
# -----------------------------
def eval_text_queries_on_index(
    text_feats: torch.Tensor,        # [Q,D] CPU normalized
    image_feats: torch.Tensor,       # [N,D] CPU normalized
    target_mask: np.ndarray,         # [N] bool
    ks: List[int],
) -> Dict[str, float]:
    sims_all = (text_feats @ image_feats.T).numpy()  # [Q,N]
    ms = []
    for i in range(sims_all.shape[0]):
        ms.append(per_query_metrics(sims_all[i], target_mask, ks))
    return aggregate_metrics(ms, ks)


def eval_single_query_embed_on_index(
    q_embed: torch.Tensor,           # [1,D] CPU normalized
    image_feats: torch.Tensor,       # [N,D] CPU normalized
    target_mask: np.ndarray,
    ks: List[int],
) -> Dict[str, float]:
    sims = (q_embed @ image_feats.T).squeeze(0).numpy()  # [N]
    m = per_query_metrics(sims, target_mask, ks)
    return aggregate_metrics([m], ks)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        "Paper-style CLIP retrieval eval (Original vs MU) with hit/recall@{3,50,200,500} + MRR + median/mean rank. "
        "Exp3 includes control on attr_matched pool."
    )

    # Paths (your provided roots are consistent with these defaults)
    parser.add_argument("--retain_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\retain",
                        help="retain pool root (recursive)")
    parser.add_argument("--attr_matched_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\attribute_matched",
                        help="attribute matched pool root")
    parser.add_argument("--forget_personA_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\forget_identity\person_A",
                        help="person_A proto/train split (positives, also Exp2 training positives)")
    parser.add_argument("--eval_personA_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\eval\id_reid\person_A",
                        help="unseen person_A images (optional extra positives in index)")
    parser.add_argument("--eval_others_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\eval\id_reid\others",
                        help="others images (negatives)")

    # CelebA attr file
    parser.add_argument("--celeba_attr_file", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\CelebA\Anno\list_attr_celeba.txt",
                        help="Path to list_attr_celeba.txt (enables global masks + Exp3 stable attrs)")

    # Original encoder: CelebA fine-tuned CLIP visual encoder.
    # This makes the comparison fair:
    #   Original = CelebA-adapted CLIP
    #   MU       = CelebA-adapted CLIP + IMU
    parser.add_argument(
        "--orig_visual_path",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\outputs_celeba_pair_special_attr_ft\clip_celeba_pair_attr_member_plus_retain_visual_only_epoch5_visual.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\outputs_celeba_clean_ft\clip_celeba_pair_attr_retain_only_visual_only_epoch3_visual.pth",
        help="CelebA fine-tuned visual encoder used as the Original baseline. "
             "Leave empty to use vanilla pretrained CLIP."
    )

    # MU encoder
    parser.add_argument(
        "--mu_encoder_path",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\IMU\IMU.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba_knn\IMU\IMU_knn.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\IMU\IMU_noisy30_2.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ft\ft_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ga\ga_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\graddiff\graddiff_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\kl\kl_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\cliperase\cliperase_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ruclip\ruclip_celeba.pth",
        help="MU visual encoder checkpoint, usually obtained by applying IMU to the same CelebA-adapted CLIP."
    )

    # Index sampling
    parser.add_argument("--max_retain_images", type=int, default=18000)
    parser.add_argument("--max_attr_matched_images", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)

    # Runtime
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    # Ks
    parser.add_argument("--ks", type=str, default="3,50,100,500",
                        help="Comma-separated Ks for hit/recall (default: 3,50,100,500)")


    # Exp2 scheme-B
    parser.add_argument("--do_exp2", action="store_true",
                        help="Run Exp2 Scheme-B (learn soft identity query on proto split in Original; fixed eval on Original+MU).")
    parser.add_argument("--exp2_neg_max", type=int, default=20000,
                        help="Max negatives used to learn soft identity query (sample from non-personA index).")
    parser.add_argument("--exp2_steps", type=int, default=600)
    parser.add_argument("--exp2_lr", type=float, default=0.1)
    parser.add_argument("--exp2_tau", type=float, default=0.07)
    parser.add_argument("--exp2_neg_per_step", type=int, default=512)

    # Exp3 stable-attrs prompt gen
    parser.add_argument("--do_exp3", action="store_true",
                        help="Run Exp3 composed-attr attack with prompts generated from person_A stable attrs + control on attr_matched.")
    parser.add_argument("--exp3_n_prompts", type=int, default=200)
    parser.add_argument("--exp3_min_freq", type=float, default=0.6)
    parser.add_argument("--exp3_max_attrs", type=int, default=6)
    parser.add_argument("--exp3_prompts_out", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\combo_prompts_person_A.txt",
                        help="If set, save generated prompts (one per line).")

    # NEW: Exp3 control options
    parser.add_argument("--exp3_do_attr_matched_control", action="store_true",
                        help="If set, Exp3 also evaluates the SAME prompts on target=pool:attr_matched (control).")
    parser.add_argument("--exp3_do_attr_matched_core_control", action="store_true",
                        help="If set, Exp3 also evaluates control on target=pool:attr_matched ∩ stable_core_attrs (stricter control).")
    parser.add_argument("--exp3_core_top", type=int, default=5,
                        help="How many top stable attrs to define 'core' intersection mask for strict control (default=5).")

    # Exp1 toggle
    parser.add_argument("--do_exp1", action="store_true",
                        help="Run Exp1 attribute retrieval with global masks (singles+pairs).")

    args = parser.parse_args()

    # defaults: if user didn't specify any, run exp1+exp2+exp3 AND exp3 control
    if not (args.do_exp2 or args.do_exp3):
        args.do_exp2 = True
        args.do_exp3 = True
        args.exp3_do_attr_matched_control = True
        args.exp3_do_attr_matched_core_control = True

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE = {device}\n")

    # Load models
    print("Loading CLIP models...")

    model_orig, preprocess = clip.load("ViT-B/32", device=device)
    load_visual_encoder_into_clip(
        model_orig,
        visual_path=args.orig_visual_path,
        device=device,
        name="ORIGINAL",
    )
    model_orig.eval()

    model_mu, _ = clip.load("ViT-B/32", device=device)
    load_mu_model_into_clip(model_mu, args.mu_encoder_path, device=device)
    model_mu.eval()

    if not args.celeba_attr_file or not os.path.exists(args.celeba_attr_file):
        raise RuntimeError(f"CelebA attr file not found: {args.celeba_attr_file}")
    print(f"Loading CelebA attributes: {args.celeba_attr_file}\n")
    attr_idx, attr_map = load_celeba_attr_map(args.celeba_attr_file)

    # Build index
    samples = build_samples(
        retain_dir=args.retain_dir,
        forget_personA_dir=args.forget_personA_dir,
        attr_matched_dir=args.attr_matched_dir,
        eval_personA_dir=args.eval_personA_dir,
        eval_others_dir=args.eval_others_dir,
        max_retain_images=args.max_retain_images,
        max_attr_matched_images=args.max_attr_matched_images,
        seed=args.seed,
    )
    print(f"Collected {len(samples)} images for retrieval index.\n")

    ds = ImageIndexDataset(samples, preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Encode image index once per model
    print("Encoding image index with ORIGINAL...")
    img_feat_orig, groups, paths = encode_images(model_orig, dl, device)
    print("Encoding image index with MU...")
    img_feat_mu, groups2, paths2 = encode_images(model_mu, dl, device)
    assert groups == groups2 and paths == paths2

    groups_np = np.array(groups)
    mask_personA = (groups_np == "id:person_A")
    mask_attr_matched = (groups_np == "pool:attr_matched")




    if args.do_exp2:
        pos_paths = list_images(args.forget_personA_dir)
        if len(pos_paths) == 0:
            raise RuntimeError("Exp2: proto split person_A dir is empty.")

        # negatives: sample from index excluding person_A
        neg_all = [p for p, isA in zip(paths, mask_personA.tolist()) if not isA]
        rng = np.random.default_rng(args.seed)
        if len(neg_all) > args.exp2_neg_max:
            idx = rng.choice(len(neg_all), size=args.exp2_neg_max, replace=False)
            neg_paths = [neg_all[i] for i in idx]
        else:
            neg_paths = neg_all

        # Learn q on Original ONLY, then keep it fixed
        q_embed = learn_soft_identity_query(
            model_orig=model_orig,
            preprocess=preprocess,
            device=device,
            pos_paths=pos_paths,
            neg_paths=neg_paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.exp2_steps,
            lr=args.exp2_lr,
            tau=args.exp2_tau,
            neg_per_step=args.exp2_neg_per_step,
            seed=args.seed,
        )


        proto_set = {os.path.basename(p).lower() for p in pos_paths}
        unseen_paths = list_images(args.eval_personA_dir) if (
                    args.eval_personA_dir and os.path.isdir(args.eval_personA_dir)) else []
        unseen_set = {os.path.basename(p).lower() for p in unseen_paths}

        if len(unseen_set) == 0:
            print("[Exp2] WARN: eval_personA_dir is empty or missing. Unseen split will be empty.")

        mask_proto = np.zeros(len(paths), dtype=bool)
        mask_unseen = np.zeros(len(paths), dtype=bool)

        for i, p in enumerate(paths):
            if groups[i] != "id:person_A":
                continue
            bn = os.path.basename(p).lower()
            if bn in proto_set:
                mask_proto[i] = True
            if bn in unseen_set:
                mask_unseen[i] = True

        n_all = int(mask_personA.sum())
        n_proto = int(mask_proto.sum())
        n_unseen = int(mask_unseen.sum())
        print(f"[Exp2] Target split sizes: all={n_all}, proto/train={n_proto}, unseen={n_unseen}")

        rows = []

        # (1) proto/train targets (the same subset used to learn prompt)
        if n_proto > 0:
            m_o_proto = eval_single_query_embed_on_index(q_embed, img_feat_orig, mask_proto, ks)
            m_m_proto = eval_single_query_embed_on_index(q_embed, img_feat_mu, mask_proto, ks)
            rows.append(("SoftID_fixed(person_A) [proto/train]", m_o_proto, m_m_proto))
        else:
            print("[Exp2] WARN: proto/train target mask is empty in index (unexpected).")


        if n_unseen > 0:
            m_o_unseen = eval_single_query_embed_on_index(q_embed, img_feat_orig, mask_unseen, ks)
            m_m_unseen = eval_single_query_embed_on_index(q_embed, img_feat_mu, mask_unseen, ks)
            rows.append(("SoftID_fixed(person_A) [unseen]", m_o_unseen, m_m_unseen))
        else:
            rows.append(("SoftID_fixed(person_A) [unseen]", {"num_targets": 0}, {"num_targets": 0}))

        # (optional) still keep a "all person_A" row for convenience
        m_o_all = eval_single_query_embed_on_index(q_embed, img_feat_orig, mask_personA.astype(bool), ks)
        m_m_all = eval_single_query_embed_on_index(q_embed, img_feat_mu, mask_personA.astype(bool), ks)
        rows.append(("SoftID_fixed(person_A) [all]", m_o_all, m_m_all))

        print_side_by_side_table(
            title="Experiment 2: Fixed Scheme-B Soft Identity Query (learn on proto/train in Original; eval split proto vs unseen on Original vs MU)",
            rows=rows,
            ks=ks,
        )


    # -----------------------------
    if args.do_exp3:
        personA_proto_paths = list_images(args.forget_personA_dir)
        if len(personA_proto_paths) == 0:
            raise RuntimeError("Exp3: forget_personA_dir empty, cannot compute stable attrs.")

        prompts, stable_list, picked = generate_combo_prompts_from_stable(
            personA_paths=personA_proto_paths,
            attr_idx=attr_idx,
            attr_map=attr_map,
            n_prompts=args.exp3_n_prompts,
            min_freq=args.exp3_min_freq,
            max_attrs=args.exp3_max_attrs,
            seed=args.seed,
        )

        if args.exp3_prompts_out:
            with open(args.exp3_prompts_out, "w", encoding="utf-8") as f:
                for p in prompts:
                    f.write(p + "\n")
            print(f"[Exp3] Prompts saved to: {args.exp3_prompts_out}")

        # Encode prompts
        t_o = encode_texts(model_orig, prompts, device=device)
        t_m = encode_texts(model_mu, prompts, device=device)

        rows = []



        tmask_A = mask_personA.astype(bool)
        m_o_A = eval_text_queries_on_index(t_o, img_feat_orig, tmask_A, ks)
        m_m_A = eval_text_queries_on_index(t_m, img_feat_mu, tmask_A, ks)
        rows.append(("Combo->person_A(stable-attrs)", m_o_A, m_m_A))



        if args.exp3_do_attr_matched_control:
            tmask_AM = mask_attr_matched.astype(bool)
            m_o_AM = eval_text_queries_on_index(t_o, img_feat_orig, tmask_AM, ks)
            m_m_AM = eval_text_queries_on_index(t_m, img_feat_mu, tmask_AM, ks)
            rows.append(("Control: Combo->attr_matched(pool)", m_o_AM, m_m_AM))

        # (C) Stricter control: attr_matched ∩ stable_core_attrs
        if args.exp3_do_attr_matched_core_control:
            # pick top stable attrs as "core" (intersection makes it stricter)
            stable_core = [a for (a, fr) in stable_list if (a in PHRASE and a not in EXCLUDE)]
            stable_core = stable_core[:max(1, int(args.exp3_core_top))]
            print(f"[Exp3-Control] stable_core_attrs (top-{len(stable_core)}): {', '.join(stable_core)}")

            core_mask = make_attr_intersection_mask_for_index(paths, stable_core, attr_idx, attr_map)
            tmask_AM_core = (mask_attr_matched.astype(bool) & core_mask)

            m_o_AMc = eval_text_queries_on_index(t_o, img_feat_orig, tmask_AM_core, ks)
            m_m_AMc = eval_text_queries_on_index(t_m, img_feat_mu, tmask_AM_core, ks)
            rows.append((f"Control: attr_matched ∩ stable_core(top{len(stable_core)})", m_o_AMc, m_m_AMc))

        print_side_by_side_table(
            title=f"Experiment 3: Composed attributes attack + control (prompts from person_A stable attrs, n_prompts={len(prompts)})",
            rows=rows,
            ks=ks,
        )

    print("Done.\n")


if __name__ == "__main__":
    main()
