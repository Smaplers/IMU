import os
import glob
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import clip



IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(root: str) -> List[str]:
    if root is None or (not os.path.exists(root)):
        return []
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    paths = sorted(list(set(paths)))
    return paths


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # handle ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def load_visual_encoder_into_clip(model, visual_path: str, device: str, name: str = "ORIGINAL"):
    if visual_path is None or str(visual_path).strip() == "":
        print(f"[{name}-LOAD] No visual checkpoint provided. Use vanilla pretrained CLIP.")
        return

    if not os.path.exists(visual_path):
        raise FileNotFoundError(f"[{name}-LOAD] checkpoint not found: {visual_path}")

    ckpt = torch.load(visual_path, map_location=device)

    # Case 1: checkpoint saved by train_clip_celeba_finetune.py:
    # {"model_state_dict": model.state_dict(), ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[{name}-LOAD] loaded full CLIP checkpoint from ckpt['model_state_dict']: {visual_path}")
        return


    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(
            f"[{name}-LOAD] loaded full checkpoint from ckpt['state_dict'] "
            f"(missing={len(missing)}, unexpected={len(unexpected)}): {visual_path}"
        )
        return

    # Case 3: pure visual encoder state_dict, saved as model.visual.state_dict().
    if isinstance(ckpt, dict):
        try:
            model.visual.load_state_dict(ckpt, strict=True)
            print(f"[{name}-LOAD] loaded visual-only checkpoint into model.visual: {visual_path}")
            return
        except RuntimeError:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(
                f"[{name}-LOAD] loaded dict as full model state_dict with strict=False "
                f"(missing={len(missing)}, unexpected={len(unexpected)}): {visual_path}"
            )
            return

    raise RuntimeError(f"[{name}-LOAD] Unsupported checkpoint format: {type(ckpt)} from {visual_path}")


def load_mu_model_into_clip(model_mu, mu_path: str, device: str):
    if mu_path is None or str(mu_path).strip() == "":
        raise ValueError("[MU-LOAD] mu_path is empty.")
    if not os.path.exists(mu_path):
        raise FileNotFoundError(f"[MU-LOAD] checkpoint not found: {mu_path}")

    ckpt = torch.load(mu_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_mu.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[MU-LOAD] loaded full CLIP checkpoint from ckpt['model_state_dict']: {mu_path}")
        return

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        missing, unexpected = model_mu.load_state_dict(sd, strict=False)
        print(
            f"[MU-LOAD] loaded full CLIP state_dict from ckpt['state_dict'] "
            f"(missing={len(missing)}, unexpected={len(unexpected)}): {mu_path}"
        )
        if "cfg" in ckpt:
            print(f"[MU-LOAD] cfg keys: {list(ckpt['cfg'].keys())[:8]} ...")
        if "paths" in ckpt:
            print(f"[MU-LOAD] paths: {ckpt['paths']}")
        return

    if isinstance(ckpt, dict):
        try:
            model_mu.visual.load_state_dict(ckpt, strict=True)
            print(f"[MU-LOAD] loaded visual-only checkpoint into model_mu.visual: {mu_path}")
            return
        except RuntimeError:
            missing, unexpected = model_mu.load_state_dict(ckpt, strict=False)
            print(
                f"[MU-LOAD] loaded dict as full model state_dict with strict=False "
                f"(missing={len(missing)}, unexpected={len(unexpected)}): {mu_path}"
            )
            return

    raise RuntimeError(f"[MU-LOAD] Unsupported checkpoint type: {type(ckpt)} from {mu_path}")



# -------------------------
@dataclass
class ImgItem:
    path: str
    label: int  # 1 = person_A, 0 = others


class SimpleImageDataset(Dataset):
    def __init__(self, items: List[ImgItem], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it.path).convert("RGB")
        x = self.preprocess(img)
        return x, it.label, it.path


@torch.no_grad()
def encode_images(model, loader, device: str) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    feats = []
    labels = []
    paths = []
    for x, y, p in tqdm(loader, desc="Encoding", leave=False):
        x = x.to(device)
        f = model.encode_image(x).float()
        f = normalize(f).cpu()
        feats.append(f)
        labels.append(y.numpy())
        paths.extend(list(p))
    feats = torch.cat(feats, dim=0)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    return feats, labels, paths


@torch.no_grad()
def build_prototype(model, preprocess, device: str, proto_dir: str, batch_size: int) -> torch.Tensor:
    proto_imgs = list_images(proto_dir)
    if len(proto_imgs) == 0:
        raise RuntimeError(f"No images found for proto_dir: {proto_dir}")

    items = [ImgItem(p, 1) for p in proto_imgs]
    ds = SimpleImageDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats, _, _ = encode_images(model, dl, device)
    proto = feats.mean(dim=0, keepdim=True)
    proto = normalize(proto)
    return proto



# -------------------------
def alignment_metrics(
    feats: torch.Tensor,
    labels: np.ndarray,
    proto: torch.Tensor,
    fixed_proto_from_original: torch.Tensor = None,
) -> dict:
    """
    per_model_proto:  用 proto（通常是当前模型自己从 proto split 里算的 proto）
    fixed_proto:      用 fixed_proto_from_original（通常是 proto_orig），用于 fixed 口径
    """
    pos = (labels == 1)
    neg = (labels == 0)

    # ---- per-model proto metrics (sim_pos/sim_neg/gap + AUC) ----
    sims = (feats @ proto.T).squeeze(1).numpy()  # [N]
    sim_pos = float(sims[pos].mean()) if pos.any() else float("nan")
    sim_neg = float(sims[neg].mean()) if neg.any() else float("nan")
    gap = sim_pos - sim_neg
    auc_per_model = roc_auc_binary(labels, sims)

    # ---- fixed proto metrics (sim_pos/sim_neg/gap + AUC) ----
    sim_pos_fixed = float("nan")
    sim_neg_fixed = float("nan")
    gap_fixed = float("nan")
    auc_fixed = float("nan")
    if fixed_proto_from_original is not None:
        sims_fixed = (feats @ fixed_proto_from_original.T).squeeze(1).numpy()
        sim_pos_fixed = float(sims_fixed[pos].mean()) if pos.any() else float("nan")
        sim_neg_fixed = float(sims_fixed[neg].mean()) if neg.any() else float("nan")
        gap_fixed = sim_pos_fixed - sim_neg_fixed
        auc_fixed = roc_auc_binary(labels, sims_fixed)

    return {
        # per-model
        "sim_pos": sim_pos,
        "sim_neg": sim_neg,
        "gap": float(gap),
        "auc_per_model_proto": float(auc_per_model),

        # fixed
        "sim_pos_fixed": float(sim_pos_fixed),
        "sim_neg_fixed": float(sim_neg_fixed),
        "gap_fixed": float(gap_fixed),
        "auc_fixed_proto_from_original": float(auc_fixed),
    }



# -------------------------
def topk_reid_multiK(
    gallery_feats: torch.Tensor, gallery_labels: np.ndarray, gallery_paths: List[str],
    query_feats: torch.Tensor, query_paths: List[str],
    ks=(1, 15, 50, 200, 500)
) -> dict:
    """
    For each query:
      - exclude any gallery image whose basename equals query basename (defensive no-self-match)
      - compute first rank where label==1 appears
      - compute hit@K for multiple K
      - compute recall@K for multiple K (macro avg over queries):
          recall@K = (#positives retrieved in topK) / (#positives in gallery)
    Returns:
      hits: dict K->float
      recalls: dict K->float
      mrr: float
      mean_rank_first: float
      first_ranks: List[float] (inf if not found)
      per_query_recall: dict K->List[float]
    """
    sims = (query_feats @ gallery_feats.T).numpy()  # [Q, G]

    gallery_bn_to_indices = {}
    for idx, p in enumerate(gallery_paths):
        bn = os.path.basename(p).lower()
        gallery_bn_to_indices.setdefault(bn, []).append(idx)

    Q, G = sims.shape
    pos_total = int((gallery_labels == 1).sum())
    if pos_total == 0:
        raise RuntimeError("No positive (person_A) images in gallery_labels==1; cannot compute recall.")

    first_ranks = []
    top1_paths = []
    top1_is_pos = []
    first_pos_paths = []
    first_pos_sims = []
    mrrs = []

    hit_counts = {k: 0 for k in ks}
    recall_sums = {k: 0.0 for k in ks}
    per_query_recall = {k: [] for k in ks}

    for i in range(Q):
        row = sims[i].copy()

        # no self match by basename
        q_bn = os.path.basename(query_paths[i]).lower()
        if q_bn in gallery_bn_to_indices:
            for j in gallery_bn_to_indices[q_bn]:
                row[j] = -1e9

        order = np.argsort(-row)  # descending

        # top-1 info
        top1_j = int(order[0])
        top1_paths.append(gallery_paths[top1_j])
        top1_is_pos.append(int(gallery_labels[top1_j] == 1))

        # first positive info (rank + which image)
        first = None
        first_j = None
        for r, j in enumerate(order, start=1):
            if gallery_labels[j] == 1:
                first = r
                first_j = int(j)
                break

        if first_j is None:
            first_pos_paths.append(None)
            first_pos_sims.append(float("nan"))
        else:
            first_pos_paths.append(gallery_paths[first_j])
            first_pos_sims.append(float(row[first_j]))

        if first is None:
            first_ranks.append(np.inf)
            mrrs.append(0.0)
        else:
            first_ranks.append(float(first))
            mrrs.append(1.0 / float(first))

        # hit@K and recall@K
        for k in ks:
            kk = min(int(k), G)
            topk_idx = order[:kk]
            pos_in_topk = int((gallery_labels[topk_idx] == 1).sum())

            if pos_in_topk > 0:
                hit_counts[k] += 1

            rq = pos_in_topk / float(pos_total)  # per-query recall
            recall_sums[k] += rq
            per_query_recall[k].append(rq)

    hits = {k: hit_counts[k] / float(Q) for k in ks}
    recalls = {k: recall_sums[k] / float(Q) for k in ks}

    finite = [r for r in first_ranks if np.isfinite(r)]
    mean_rank = float(np.mean(finite)) if len(finite) else float("inf")

    return {
        "hits": hits,
        "recalls": recalls,
        "mrr": float(np.mean(mrrs)),
        "mean_rank_first": mean_rank,
        "first_ranks": first_ranks,
        "per_query_recall": per_query_recall,
        "pos_total": pos_total,
        "num_queries": Q,
        "top1_paths": top1_paths,
        "top1_is_pos": top1_is_pos,
        "first_pos_paths": first_pos_paths,
        "first_pos_sims": first_pos_sims,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser("Paper-style identity forgetting eval: proto split + gap + AUC + TopK re-id (big gallery)")

    parser.add_argument("--proto_personA_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\forget_identity\person_A",
                        help="Build prototype from this split (train/proto split).")

    parser.add_argument("--eval_personA_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\eval\id_reid\person_A",
                        help="Queries/positives for eval (person_A).")

    parser.add_argument("--eval_others_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\eval\id_reid\others",
                        help="Negatives for eval (others).")

    # big gallery sources
    parser.add_argument("--retain_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\retain\celeba",
                        help="Extra negatives from retain pool (optional).")

    parser.add_argument("--attr_matched_dir", type=str,
                        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\attribute_matched",
                        help="Extra hard negatives from attribute_matched (optional).")

    parser.add_argument("--gallery_neg_sources", type=str, default="eval,retain,attr",
                        help="Comma-separated: eval,retain,attr. Controls where negatives come from.")

    parser.add_argument("--gallery_max_neg", type=int, default=30000,
                        help="Cap number of negative gallery images (0 = no cap).")

    parser.add_argument("--gallery_pos_max", type=int, default=0,
                        help="Cap number of positive gallery images (0 = no cap).")

    # parser.add_argument("--mu_encoder_path", type=str, default="mu_clip_encoder_no_path_ablation_celeba.pth")

    parser.add_argument(
        "--orig_visual_path",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\outputs_celeba_pair_special_attr_ft\clip_celeba_pair_attr_member_plus_retain_visual_only_epoch5_visual.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\outputs_celeba_clean_ft\clip_celeba_pair_attr_retain_only_visual_only_epoch3_visual.pth",
        help="CelebA fine-tuned visual encoder used as the Original baseline. Leave empty to use vanilla CLIP."
    )

    parser.add_argument(
        "--mu_encoder_path",
        type=str,
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ft\ft_celeba.pth",
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\IMU\IMU.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba_knn\IMU\IMU_knn.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\IMU\IMU_noisy30_2.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ga\ga_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\graddiff\graddiff_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\kl\kl_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\cliperase\cliperase_celeba.pth",
        # default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\Revision\MU_celeba\ruclip\ruclip_celeba.pth",
        help="MU / unlearned checkpoint. Can be visual-only or full CLIP checkpoint."
    )


    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--do_topk", type=int, default=1, help="Enable Top-K retrieval evaluation (1/0).")
    parser.add_argument("--topk", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Loading models...")

    model_orig, preprocess = clip.load("ViT-B/32", device=device)
    load_visual_encoder_into_clip(
        model_orig,
        visual_path=args.orig_visual_path,
        device=device,
        name="ORIGINAL"
    )
    model_orig.eval()

    model_mu, _ = clip.load("ViT-B/32", device=device)
    load_mu_model_into_clip(model_mu, args.mu_encoder_path, device=device)
    model_mu.eval()


    print("\nBuilding prototypes from proto split...")
    proto_orig = build_prototype(model_orig, preprocess, device, args.proto_personA_dir, args.batch_size)
    proto_mu   = build_prototype(model_mu,   preprocess, device, args.proto_personA_dir, args.batch_size)

    # Build eval set (person_A eval + others eval)
    eval_personA = list_images(args.eval_personA_dir)
    eval_others  = list_images(args.eval_others_dir)

    if len(eval_personA) == 0 or len(eval_others) == 0:
        raise RuntimeError("Eval folders empty. Check eval_personA_dir / eval_others_dir.")

    items = [ImgItem(p, 1) for p in eval_personA] + [ImgItem(p, 0) for p in eval_others]
    ds = SimpleImageDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Encode eval set in each model space
    print("\nEncoding eval set with ORIGINAL...")
    feats_o, labels, paths = encode_images(model_orig, dl, device)
    print("Encoding eval set with MU...")
    feats_m, labels_m, paths_m = encode_images(model_mu, dl, device)
    assert np.array_equal(labels, labels_m)

    # Alignment metrics
    print("\n=== Alignment / Forgetting Metrics (Proto-split) ===")
    m_orig = alignment_metrics(
        feats_o, labels,
        proto=proto_orig,
        fixed_proto_from_original=proto_orig,  # Original 上：fixed 就是自己
    )
    m_mu = alignment_metrics(
        feats_m, labels,
        proto=proto_mu,
        fixed_proto_from_original=proto_orig,  # MU 上：fixed 强制用 proto_orig
    )

    def show(tag, m):
        print(f"{tag}")
        # per-model (保持你原来的三行)
        print(f"  sim_pos(person_A): {m['sim_pos']:.4f}")
        print(f"  sim_neg(others):   {m['sim_neg']:.4f}")
        print(f"  gap(pos-neg):      {m['gap']:.4f}")
        print(f"  AUC(per_model_proto):          {m['auc_per_model_proto']:.4f}")

        # fixed (新增，但不打乱原结构：作为一个“fixed:”小块)
        print(f"  fixed: sim_pos(person_A): {m['sim_pos_fixed']:.4f}")
        print(f"  fixed: sim_neg(others):   {m['sim_neg_fixed']:.4f}")
        print(f"  fixed: gap(pos-neg):      {m['gap_fixed']:.4f}")
        print(f"  AUC(fixed_proto_from_original): {m['auc_fixed_proto_from_original']:.4f}")

    show("Original CLIP / CelebA-adapted baseline", m_orig)
    show("MU CLIP", m_mu)

    print("\nDelta (Original - MU)")
    print(f"  gap drop (per-model):  {m_orig['gap'] - m_mu['gap']:.4f}")
    print(f"  gap drop (fixed):      {m_orig['gap_fixed'] - m_mu['gap_fixed']:.4f}")
    print(f"  AUC(per_model_proto) drop:          {m_orig['auc_per_model_proto'] - m_mu['auc_per_model_proto']:.4f}")
    print(f"  AUC(fixed_proto_from_original) drop: {m_orig['auc_fixed_proto_from_original'] - m_mu['auc_fixed_proto_from_original']:.4f}")

    # Top-K Re-ID with BIG gallery
    if args.do_topk:
        print(f"\n=== Top-{args.topk} Re-ID Retrieval Eval (BIG Gallery, No Self-Match) ===")

        # positives in gallery
        gallery_pos_paths = list_images(args.proto_personA_dir)
        if len(gallery_pos_paths) == 0:
            raise RuntimeError("Gallery positives empty. Check proto_personA_dir.")

        if args.gallery_pos_max and args.gallery_pos_max > 0 and len(gallery_pos_paths) > args.gallery_pos_max:
            gallery_pos_paths = random.sample(gallery_pos_paths, args.gallery_pos_max)

        # queries
        query_paths = list_images(args.eval_personA_dir)
        if len(query_paths) == 0:
            raise RuntimeError("No query images found in eval_personA_dir.")
        query_bns = set(os.path.basename(p).lower() for p in query_paths)

        # negatives from sources
        neg_sources = set([s.strip().lower() for s in args.gallery_neg_sources.split(",") if s.strip()])
        gallery_neg_paths = []

        if "eval" in neg_sources:
            gallery_neg_paths += list_images(args.eval_others_dir)
        if "retain" in neg_sources and os.path.exists(args.retain_dir):
            gallery_neg_paths += list_images(args.retain_dir)
        if "attr" in neg_sources and os.path.exists(args.attr_matched_dir):
            gallery_neg_paths += list_images(args.attr_matched_dir)

        # dedup
        gallery_neg_paths = sorted(list(set(gallery_neg_paths)))

        # remove any accidental overlap with positives (full path)
        pos_set = set(p.lower() for p in gallery_pos_paths)
        gallery_neg_paths = [p for p in gallery_neg_paths if p.lower() not in pos_set]

        # remove negatives whose basename overlaps with queries (defensive leak prevention)
        gallery_neg_paths = [p for p in gallery_neg_paths if os.path.basename(p).lower() not in query_bns]

        # cap negatives
        if args.gallery_max_neg and args.gallery_max_neg > 0 and len(gallery_neg_paths) > args.gallery_max_neg:
            gallery_neg_paths = random.sample(gallery_neg_paths, args.gallery_max_neg)

        print(f"Gallery positives (person_A): {len(gallery_pos_paths)}")
        print(f"Gallery negatives (others):   {len(gallery_neg_paths)}  (sources={args.gallery_neg_sources}, max={args.gallery_max_neg})")
        print(f"Queries (person_A):           {len(query_paths)}")

        # gallery dataset
        gallery_items = [ImgItem(p, 1) for p in gallery_pos_paths] + [ImgItem(p, 0) for p in gallery_neg_paths]
        gallery_ds = SimpleImageDataset(gallery_items, preprocess)
        gallery_dl = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("Encoding gallery (Original)...")
        g_o, g_labels, g_paths = encode_images(model_orig, gallery_dl, device)
        print("Encoding gallery (MU)...")
        g_m, g_labels2, g_paths2 = encode_images(model_mu, gallery_dl, device)
        assert np.array_equal(g_labels, g_labels2)

        # query dataset
        q_items = [ImgItem(p, 1) for p in query_paths]
        q_ds = SimpleImageDataset(q_items, preprocess)
        q_dl = DataLoader(q_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print("Encoding queries (Original)...")
        q_o, _, q_paths_o = encode_images(model_orig, q_dl, device)
        print("Encoding queries (MU)...")
        q_m, _, q_paths_m = encode_images(model_mu, q_dl, device)

        ks = (1, 15, 50, 200, 500)

        r_orig = topk_reid_multiK(g_o, g_labels, g_paths, q_o, q_paths_o, ks=ks)
        r_mu = topk_reid_multiK(g_m, g_labels, g_paths, q_m, q_paths_o, ks=ks)

        def fmt(v, nd=4):
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                return "inf" if np.isinf(v) else "nan"
            return f"{v:.{nd}f}"

        rows = []
        for k in ks:
            rows.append((f"hit@{k}", r_orig["hits"][k], r_mu["hits"][k], r_orig["hits"][k] - r_mu["hits"][k]))
        for k in (15, 50, 200, 500):
            rows.append((f"recall@{k}", r_orig["recalls"][k], r_mu["recalls"][k], r_orig["recalls"][k] - r_mu["recalls"][k]))
        rows.append(("MRR", r_orig["mrr"], r_mu["mrr"], r_orig["mrr"] - r_mu["mrr"]))
        rows.append(("mean_rank_first", r_orig["mean_rank_first"], r_mu["mean_rank_first"], r_orig["mean_rank_first"] - r_mu["mean_rank_first"]))

        header = ["Metric", "Original", "MU", "Delta (O-M)"]
        table = [header] + [[m, fmt(o), fmt(mu), fmt(d)] for (m, o, mu, d) in rows]

        widths = [max(len(str(table[r][c])) for r in range(len(table))) for c in range(len(header))]

        def render_row(vals):
            return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

        sep = "-+-".join("-" * w for w in widths)

        print("\n" + "=" * 90)
        print("Top-K Re-ID Retrieval Eval (BIG Gallery, No Self-Match): Original vs MU vs Delta")
        print("=" * 90)
        print(render_row(table[0]))
        print(sep)
        for r in range(1, len(table)):
            print(render_row(table[r]))

        print(f"\nGallery positives (person_A): {r_orig['pos_total']}")
        print(f"Queries (person_A):           {r_orig['num_queries']}\n")

        print("=" * 90)
        print("Per-query first_rank (smaller is better; inf means no positive found)")
        print("=" * 90)
        for i, qp in enumerate(q_paths_o):
            bn = os.path.basename(qp)

            fr_o = r_orig["first_ranks"][i]
            fr_m = r_mu["first_ranks"][i]

            fr_o_s = "inf" if np.isinf(fr_o) else f"{int(fr_o)}"
            fr_m_s = "inf" if np.isinf(fr_m) else f"{int(fr_m)}"

            mu_top1 = r_mu["top1_paths"][i]
            mu_top1_bn = os.path.basename(mu_top1) if mu_top1 else "None"
            mu_top1_pos = r_mu["top1_is_pos"][i]

            mu_fp = r_mu["first_pos_paths"][i]
            mu_fp_bn = os.path.basename(mu_fp) if mu_fp else "None"
            mu_fp_sim = r_mu["first_pos_sims"][i]

            print(
                f"{i:02d} | {bn:25s} | O:first_rank={fr_o_s:>6s} | MU:first_rank={fr_m_s:>6s} "
                f"| MU:top1={mu_top1_bn:25s} (pos={mu_top1_pos}) "
                f"| MU:first_pos={mu_fp_bn:25s} sim={mu_fp_sim:.4f}"
            )

        def summarize_first_ranks(fr_list: List[float], tag: str):
            inf_cnt = sum(1 for r in fr_list if np.isinf(r))
            finite = [r for r in fr_list if np.isfinite(r)]
            med = float(np.median(finite)) if finite else float("inf")
            p90 = float(np.percentile(finite, 90)) if finite else float("inf")
            print(f"\n[{tag}] first_rank summary:")
            print(f"  inf_count: {inf_cnt}/{len(fr_list)}")
            print(f"  median(first_rank): {med:.1f}")
            print(f"  p90(first_rank):    {p90:.1f}")

        summarize_first_ranks(r_orig["first_ranks"], "Original")
        summarize_first_ranks(r_mu["first_ranks"], "MU")

    print("\nDone.")


if __name__ == "__main__":
    main()
