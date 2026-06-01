import os
import glob
import random
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import clip



# =========================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(img_dir: str, recursive: bool = False) -> List[str]:
    if img_dir is None or not os.path.exists(img_dir):
        return []

    paths = []

    if recursive:
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(img_dir, "**", f"*{ext}"), recursive=True))
            paths.extend(glob.glob(os.path.join(img_dir, "**", f"*{ext.upper()}"), recursive=True))
    else:
        for fn in os.listdir(img_dir):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(img_dir, fn))

    return sorted(list(set(paths)))


def collect_basenames_from_dirs(dirs: Optional[List[str]]) -> set:
    names = set()

    if dirs is None:
        return names

    for d in dirs:
        if not d:
            continue

        if not os.path.exists(d):
            print(f"[WARN] dir not found, skip: {d}")
            continue

        for p in list_images(d, recursive=True):
            names.add(os.path.basename(p).lower())

    return names


def image_stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def readable_attr_name(attr: str) -> str:
    return attr.replace("_", " ").lower()


# =========================================================
def read_celeba_identity(id_file: str) -> Dict[str, str]:
    id_map = {}

    with open(id_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, pid = parts
                id_map[filename] = pid

    return id_map


def read_celeba_attrs(attr_file: str) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    with open(attr_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        raise RuntimeError(f"Invalid CelebA attr file: {attr_file}")

    attr_names = lines[1].split()
    attr_map = {}

    for line in lines[2:]:
        parts = line.split()
        if len(parts) != len(attr_names) + 1:
            continue

        fn = parts[0]
        vals = [int(x) for x in parts[1:]]
        attr_map[fn] = {a: v for a, v in zip(attr_names, vals)}

    return attr_names, attr_map



# =========================================================
def build_image_specific_prompt(filename: str, pid: str, prompt_template: str) -> str:
    stem = image_stem(filename)

    return prompt_template.format(
        pid=pid,
        stem=stem,
        filename=filename,
    )


def build_attr_text(
    filename: str,
    attr_map: Dict[str, Dict[str, int]],
    use_attrs: Optional[List[str]] = None,
    max_attrs: int = 8,
) -> str:
    if filename not in attr_map:
        return ""

    attrs = attr_map[filename]

    if use_attrs is None or len(use_attrs) == 0:
        selected = [a for a, v in attrs.items() if int(v) == 1]
    else:
        selected = [a for a in use_attrs if a in attrs and int(attrs[a]) == 1]

    selected = selected[:max_attrs]

    if len(selected) == 0:
        return ""

    return ", ".join(readable_attr_name(a) for a in selected)


def build_identity_text(
    fn_lower: str,
    pid: str,
    member_basenames: set,
    target_name: str,
) -> str:
    if fn_lower in member_basenames:
        return target_name

    return f"identity {pid}"


def build_three_prompts(
    filename: str,
    pid: str,
    prompt_template: str,
    attr_map: Dict[str, Dict[str, int]],
    member_basenames: set,
    target_name: str,
    use_attrs: Optional[List[str]],
    max_attrs: int,
) -> Tuple[str, str, str]:
    fn_lower = filename.lower()

    pair_prompt = build_image_specific_prompt(
        filename=filename,
        pid=pid,
        prompt_template=prompt_template,
    )

    identity_text = build_identity_text(
        fn_lower=fn_lower,
        pid=pid,
        member_basenames=member_basenames,
        target_name=target_name,
    )

    attr_text = build_attr_text(
        filename=filename,
        attr_map=attr_map,
        use_attrs=use_attrs,
        max_attrs=max_attrs,
    )

    if attr_text:
        attr_prompt = f"a face photo of a person with {attr_text}"
        id_attr_prompt = f"a face photo of {identity_text} with {attr_text}"
    else:
        attr_prompt = "a face photo of a person"
        id_attr_prompt = f"a face photo of {identity_text}"

    return pair_prompt, attr_prompt, id_attr_prompt



# =========================================================
class CelebAPairAttrDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        id_file: str,
        attr_file: str,
        preprocess,
        prompt_template: str,
        member_basenames: set,
        nonmember_basenames: set,
        exclude_basenames: set,
        target_name: str = "person_A",
        use_attrs: Optional[List[str]] = None,
        max_attrs: int = 8,
        max_images: int = 0,
        train_scope: str = "all",
        member_dir: Optional[str] = None,
        retain_dirs: Optional[List[str]] = None,
    ):
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.prompt_template = prompt_template
        self.member_basenames = member_basenames or set()
        self.nonmember_basenames = nonmember_basenames or set()
        self.exclude_basenames = exclude_basenames or set()
        self.target_name = target_name
        self.use_attrs = use_attrs
        self.max_attrs = max_attrs

        self.id_map = read_celeba_identity(id_file)
        _, self.attr_map = read_celeba_attrs(attr_file)

        if train_scope == "all":
            all_paths = list_images(img_dir, recursive=False)

        elif train_scope == "member_only":
            if member_dir is None:
                raise RuntimeError("train_scope=member_only requires --member_dir.")
            all_paths = list_images(member_dir, recursive=True)

        elif train_scope == "member_plus_retain":
            if member_dir is None:
                raise RuntimeError("train_scope=member_plus_retain requires --member_dir.")
            all_paths = list_images(member_dir, recursive=True)
            if retain_dirs:
                for d in retain_dirs:
                    all_paths.extend(list_images(d, recursive=True))
            all_paths = sorted(list(set(all_paths)))

        else:
            raise ValueError(f"Unknown train_scope: {train_scope}")

        valid_paths = []
        excluded = 0
        no_identity = 0
        no_attr = 0
        included_member = 0
        included_nonmember = 0

        for p in all_paths:
            fn = os.path.basename(p)
            fn_lower = fn.lower()

            if fn_lower in self.exclude_basenames:
                excluded += 1
                continue

            if fn not in self.id_map:
                no_identity += 1
                continue

            if fn not in self.attr_map:
                no_attr += 1
                continue

            valid_paths.append(p)

            if fn_lower in self.member_basenames:
                included_member += 1
            if fn_lower in self.nonmember_basenames:
                included_nonmember += 1

        if max_images > 0 and len(valid_paths) > max_images:
            random.shuffle(valid_paths)
            valid_paths = valid_paths[:max_images]

        self.paths = valid_paths

        print(f"[Dataset] train_scope: {train_scope}")
        print(f"[Dataset] valid training images: {len(self.paths)}")
        print(f"[Dataset] excluded by basename: {excluded}")
        print(f"[Dataset] no identity annotation: {no_identity}")
        print(f"[Dataset] no attr annotation: {no_attr}")
        print(f"[Audit] member basenames included in training: {included_member}")
        print(f"[Audit] nonmember basenames included in training: {included_nonmember}")

        if included_nonmember > 0:
            print("[WARN] Some non-member images are still included in training. "
                  "For strict pair MIA, non-member images should be excluded.")

        if len(self.member_basenames) > 0 and included_member == 0:
            print("[WARN] No member images detected in training. "
                  "Check whether member_dir basenames match CelebA image filenames.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        fn = os.path.basename(path)
        pid = self.id_map.get(fn, "unknown")

        img = Image.open(path).convert("RGB")
        image = self.preprocess(img)

        pair_prompt, attr_prompt, id_attr_prompt = build_three_prompts(
            filename=fn,
            pid=pid,
            prompt_template=self.prompt_template,
            attr_map=self.attr_map,
            member_basenames=self.member_basenames,
            target_name=self.target_name,
            use_attrs=self.use_attrs,
            max_attrs=self.max_attrs,
        )

        return image, pair_prompt, attr_prompt, id_attr_prompt, fn


# =========================================================
# CLIP losses
# =========================================================
def clip_contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    labels = torch.arange(image_features.size(0), device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2.0


def set_trainable(model, train_mode: str):
    for p in model.parameters():
        p.requires_grad = False

    if train_mode == "visual_only":
        for p in model.visual.parameters():
            p.requires_grad = True
        model.logit_scale.requires_grad = True

    elif train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")



# =========================================================
def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    model, preprocess = clip.load(args.clip_model, device=device)
    model.float()

    member_basenames = collect_basenames_from_dirs([args.member_dir]) if args.member_dir else set()
    nonmember_basenames = collect_basenames_from_dirs([args.nonmember_dir]) if args.nonmember_dir else set()
    exclude_basenames = collect_basenames_from_dirs(args.exclude_dirs)

    print(f"[Audit] member basenames: {len(member_basenames)}")
    print(f"[Audit] nonmember basenames: {len(nonmember_basenames)}")
    print(f"[Exclude] filenames to exclude: {len(exclude_basenames)}")

    overlap_nonmember_exclude = len(nonmember_basenames & exclude_basenames)
    print(f"[Audit] nonmember images covered by exclude_dirs: "
          f"{overlap_nonmember_exclude}/{len(nonmember_basenames)}")

    if args.strict_audit and len(nonmember_basenames) > 0:
        if overlap_nonmember_exclude < len(nonmember_basenames):
            raise RuntimeError(
                "Strict audit failed: not all non-member images are excluded. "
                "Please add nonmember_dir to --exclude_dirs."
            )

    overlap_member_exclude = len(member_basenames & exclude_basenames)
    print(f"[Audit] member images accidentally excluded: "
          f"{overlap_member_exclude}/{len(member_basenames)}")

    if args.strict_audit and overlap_member_exclude > 0:
        raise RuntimeError(
            "Strict audit failed: some member images are in exclude_dirs. "
            "For pair MIA, member images must be included in fine-tuning."
        )

    use_attrs = None
    if args.use_attrs:
        use_attrs = [a.strip() for a in args.use_attrs.split(",") if a.strip()]
        print(f"[Attrs] using selected attributes: {use_attrs}")
    else:
        print("[Attrs] using all positive CelebA attributes.")

    retain_dirs = args.retain_dirs if args.retain_dirs else []

    dataset = CelebAPairAttrDataset(
        img_dir=args.img_dir,
        id_file=args.id_file,
        attr_file=args.attr_file,
        preprocess=preprocess,
        prompt_template=args.prompt_template,
        member_basenames=member_basenames,
        nonmember_basenames=nonmember_basenames,
        exclude_basenames=exclude_basenames,
        target_name=args.target_name,
        use_attrs=use_attrs,
        max_attrs=args.max_attrs,
        max_images=args.max_images,
        train_scope=args.train_scope,
        member_dir=args.member_dir,
        retain_dirs=retain_dirs,
    )

    if len(dataset) == 0:
        raise RuntimeError("No valid training images found.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    set_trainable(model, args.train_mode)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Trainable params] {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("Training configuration")
    print("=" * 90)
    print(f"CLIP model:       {args.clip_model}")
    print(f"Train mode:       {args.train_mode}")
    print(f"Train scope:      {args.train_scope}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"LR:               {args.lr}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Target name:      {args.target_name}")
    print(f"Prompt template:  {args.prompt_template}")
    print(f"lambda_attr:      {args.lambda_attr}")
    print(f"lambda_id_attr:   {args.lambda_id_attr}")
    print("=" * 90 + "\n")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_pair = 0.0
        total_attr = 0.0
        total_id_attr = 0.0
        total_logit_scale = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, pair_texts, attr_texts, id_attr_texts, _ in pbar:
            images = images.to(device, non_blocking=True)

            pair_tokens = clip.tokenize(list(pair_texts), truncate=True).to(device)
            attr_tokens = clip.tokenize(list(attr_texts), truncate=True).to(device)
            id_attr_tokens = clip.tokenize(list(id_attr_texts), truncate=True).to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and args.amp)):
                image_features = model.encode_image(images)

                pair_text_features = model.encode_text(pair_tokens)
                attr_text_features = model.encode_text(attr_tokens)
                id_attr_text_features = model.encode_text(id_attr_tokens)

                logit_scale = model.logit_scale.exp()
                logit_scale = torch.clamp(logit_scale, max=100.0)

                loss_pair = clip_contrastive_loss(
                    image_features=image_features,
                    text_features=pair_text_features,
                    logit_scale=logit_scale,
                )

                loss_attr = clip_contrastive_loss(
                    image_features=image_features,
                    text_features=attr_text_features,
                    logit_scale=logit_scale,
                )

                loss_id_attr = clip_contrastive_loss(
                    image_features=image_features,
                    text_features=id_attr_text_features,
                    logit_scale=logit_scale,
                )

                loss = (
                    args.lambda_pair * loss_pair
                    + args.lambda_attr * loss_attr
                    + args.lambda_id_attr * loss_id_attr
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_pair += loss_pair.item()
            total_attr += loss_attr.item()
            total_id_attr += loss_id_attr.item()
            total_logit_scale += model.logit_scale.exp().item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pair": f"{loss_pair.item():.4f}",
                "attr": f"{loss_attr.item():.4f}",
                "id_attr": f"{loss_id_attr.item():.4f}",
                "logit_scale": f"{model.logit_scale.exp().item():.2f}",
            })

        n = max(1, len(loader))
        print(
            f"[Epoch {epoch}] "
            f"loss={total_loss / n:.4f}, "
            f"pair={total_pair / n:.4f}, "
            f"attr={total_attr / n:.4f}, "
            f"id_attr={total_id_attr / n:.4f}, "
            f"logit_scale={total_logit_scale / n:.2f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            tag = (
                f"clip_celeba_pair_attr_"
                f"{args.train_scope}_{args.train_mode}_epoch{epoch}"
            )

            full_path = os.path.join(args.output_dir, f"{tag}.pth")
            visual_path = os.path.join(args.output_dir, f"{tag}_visual.pth")

            torch.save({
                "model_state_dict": model.state_dict(),
                "clip_model": args.clip_model,
                "prompt_mode": "pair_attr_id_attr",
                "prompt_template": args.prompt_template,
                "target_name": args.target_name,
                "train_mode": args.train_mode,
                "train_scope": args.train_scope,
                "epoch": epoch,
                "args": vars(args),
            }, full_path)

            torch.save(model.visual.state_dict(), visual_path)

            print(f"[Save] full CLIP checkpoint: {full_path}")
            print(f"[Save] visual-only checkpoint: {visual_path}")


# =========================================================
# Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        "Fine-tune CLIP with image-specific + attribute captions for pair-MIA and attribute retrieval"
    )

    parser.add_argument(
        "--celeba_root",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\CelebA",
    )

    parser.add_argument(
        "--split_root",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA",
    )

    parser.add_argument(
        "--member_dir",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\forget_identity\person_A",
        help="Images included in fine-tuning and later treated as member pairs.",
    )

    parser.add_argument(
        "--nonmember_dir",
        type=str,
        default=r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\data_CelebA\eval\id_reid\person_A",
        help="Images excluded from fine-tuning and later treated as non-member pairs.",
    )

    parser.add_argument(
        "--retain_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Used only when train_scope=member_plus_retain.",
    )

    parser.add_argument(
        "--exclude_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Directories whose image basenames should be excluded from fine-tuning.",
    )

    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
    )

    parser.add_argument(
        "--target_name",
        type=str,
        default="person_A",
        help="Text name for member images.",
    )

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a face photo of {pid}, image {stem}",
        help="Image-specific prompt. Available fields: {pid}, {stem}, {filename}.",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        default="visual_only",
        choices=["visual_only", "full"],
    )

    parser.add_argument(
        "--train_scope",
        type=str,
        default="member_plus_retain",
        choices=["all", "member_only", "member_plus_retain"],
        help=(
            "member_only: only person_A member images. "
            "member_plus_retain: person_A member images plus retain dirs. "
            "all: all CelebA images except excluded basenames."
        ),
    )

    parser.add_argument(
        "--use_attrs",
        type=str,
        default="Smiling,Blond_Hair,Bangs,No_Beard,Wearing_Lipstick,Wearing_Earrings,Wearing_Necklace,Wavy_Hair,Young,Brown_Hair",
        help=(
            "Comma-separated CelebA attrs. Empty string means all positive attrs. "
            "Recommended to use stable attrs for person_A."
        ),
    )

    parser.add_argument("--max_attrs", type=int, default=8)

    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--lambda_attr", type=float, default=0.5)
    parser.add_argument("--lambda_id_attr", type=float, default=0.8)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--strict_audit", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"outputs_celeba_pair_special_attr_ft",
    )

    args = parser.parse_args()

    # CelebA image directory.
    cand1 = os.path.join(args.celeba_root, "img_align_celeba")
    cand2 = os.path.join(args.celeba_root, "Img", "img_align_celeba")

    if os.path.exists(cand1):
        args.img_dir = cand1
    elif os.path.exists(cand2):
        args.img_dir = cand2
    else:
        raise FileNotFoundError(
            f"Cannot find CelebA image directory. Tried:\n{cand1}\n{cand2}"
        )

    args.id_file = os.path.join(args.celeba_root, "Anno", "identity_CelebA.txt")
    args.attr_file = os.path.join(args.celeba_root, "Anno", "list_attr_celeba.txt")

    if not os.path.exists(args.id_file):
        raise FileNotFoundError(f"Cannot find identity file: {args.id_file}")

    if not os.path.exists(args.attr_file):
        raise FileNotFoundError(f"Cannot find attr file: {args.attr_file}")

    if args.exclude_dirs is None:
        args.exclude_dirs = [args.nonmember_dir]

    if args.retain_dirs is None:
        args.retain_dirs = [
            os.path.join(args.split_root, "retain"),
            os.path.join(args.split_root, "attribute_matched"),
        ]

    if args.use_attrs is not None and args.use_attrs.strip() == "":
        args.use_attrs = None

    print(f"[CelebA image dir] {args.img_dir}")
    print(f"[Identity file] {args.id_file}")
    print(f"[Attr file] {args.attr_file}")
    print(f"[Member dir] {args.member_dir}")
    print(f"[Non-member dir] {args.nonmember_dir}")
    print(f"[Retain dirs] {args.retain_dirs}")
    print(f"[Exclude dirs] {args.exclude_dirs}")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)