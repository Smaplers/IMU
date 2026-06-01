

import os
import random
import shutil
from collections import defaultdict, Counter
from tqdm import tqdm

# ================= CONFIG =================
CELEBA_ROOT = r"D:\Clip_MU\Minimal Clip Path-level Machine Unlearning\CelebA"   # <-- change if needed

IMG_DIR   = os.path.join(CELEBA_ROOT, "Img", "img_align_celeba")
ID_FILE   = os.path.join(CELEBA_ROOT, "Anno", "identity_CelebA.txt")
ATTR_FILE = os.path.join(CELEBA_ROOT, "Anno", "list_attr_celeba.txt")

OUT_ROOT = "data_CelebA"  # can be absolute path if you want

# ---------------- Scale knobs (CelebA-realistic) ----------------
MIN_IMGS_PER_ID = 25  # realistic threshold (20~30 recommended)

FORGET_TRAIN_N = 20
FORGET_EVAL_N  = 10

ATTR_MATCH_ID_COUNT = 200
ATTR_MATCH_N_PER_ID = 15

RETAIN_ID_COUNT = 2000
RETAIN_N_PER_ID = 10

EVAL_OTHER_ID_COUNT = 1000
EVAL_OTHER_N_PER_ID = 5

MATCH_ATTRS = ["Male", "Young"]
EVAL_ATTRS  = ["Male", "Young", "Smiling"]
EVAL_ATTR_N = 1000

# Practical options
CLEAN_OUT = True
LINK_MODE = "hardlink"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ================= UTILS =================
def mkdir(p):
    os.makedirs(p, exist_ok=True)

def rm_tree(p):
    if os.path.exists(p):
        shutil.rmtree(p)

def place_file(src, dst, mode="copy"):
    mkdir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown LINK_MODE: {mode}")

def sample_at_most(img_list, k, avoid_set=None):
    if avoid_set is None:
        avoid_set = set()
    candidates = [im for im in img_list if im not in avoid_set]
    if len(candidates) == 0:
        return []
    if len(candidates) <= k:
        return candidates
    return random.sample(candidates, k)

def read_identity_file(id_file):
    id_map = defaultdict(list)
    with open(id_file, "r", encoding="utf-8") as f:
        for line in f:
            img, pid = line.strip().split()
            id_map[pid].append(img)
    return id_map

def read_attr_file(attr_file):
    with open(attr_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    attr_names = lines[1].split()
    attr_map = {}
    for line in lines[2:]:
        parts = line.split()
        img = parts[0]
        vals = list(map(int, parts[1:]))
        attr_map[img] = {attr_names[i]: vals[i] for i in range(len(attr_names))}
    return attr_names, attr_map

def identity_attr_signature(pid_imgs, attr_map, attrs, n_sample=10):
    sample = pid_imgs[:n_sample] if len(pid_imgs) >= n_sample else pid_imgs
    sig = {}
    for a in attrs:
        votes = []
        for im in sample:
            if im in attr_map and a in attr_map[im]:
                votes.append(attr_map[im][a])
        if len(votes) == 0:
            sig[a] = None
        else:
            s = sum(1 if v == 1 else -1 for v in votes)
            sig[a] = 1 if s >= 0 else -1
    return sig

def src_path(img_name):
    return os.path.join(IMG_DIR, img_name)

# ================= MAIN =================
def main():
    if not os.path.exists(IMG_DIR):
        raise RuntimeError(
            f"Image directory not found: {IMG_DIR}. "
            "Please extract img_align_celeba first."
        )

    # Clean output
    if CLEAN_OUT:
        print(f"Cleaning OUT_ROOT: {OUT_ROOT}")
        rm_tree(OUT_ROOT)

    # Load metadata
    print("Loading identity labels...")
    id_map = read_identity_file(ID_FILE)

    print("Loading attribute labels...")
    _, attr_map = read_attr_file(ATTR_FILE)

    lens = sorted([len(v) for v in id_map.values()], reverse=True)
    print("Top-10 identity image counts:", lens[:10])
    print("Median identity image count:", lens[len(lens)//2])

    need_forget_total = FORGET_TRAIN_N + FORGET_EVAL_N
    threshold = max(MIN_IMGS_PER_ID, need_forget_total)
    valid_ids = [pid for pid, imgs in id_map.items() if len(imgs) >= threshold]
    print(f"Valid identities (>= {threshold} imgs): {len(valid_ids)}")

    if len(valid_ids) == 0:
        raise RuntimeError(
            "No valid identities found. Lower MIN_IMGS_PER_ID and/or FORGET_* counts."
        )

    forget_id = random.choice(valid_ids)
    print(f"Selected forget identity: {forget_id} (total imgs={len(id_map[forget_id])})")

    forget_all = random.sample(id_map[forget_id], need_forget_total)
    forget_train_imgs = forget_all[:FORGET_TRAIN_N]
    forget_eval_imgs  = forget_all[FORGET_TRAIN_N:]

    # Output dirs
    forget_dir = os.path.join(OUT_ROOT, "forget_identity", "person_A")
    attr_dir   = os.path.join(OUT_ROOT, "attribute_matched")
    retain_dir = os.path.join(OUT_ROOT, "retain", "celeba")

    eval_id_dir   = os.path.join(OUT_ROOT, "eval", "id_reid")
    eval_forget   = os.path.join(eval_id_dir, "person_A")
    eval_others   = os.path.join(eval_id_dir, "others")

    eval_attr_dir = os.path.join(OUT_ROOT, "eval", "attr_preserve")

    for p in [forget_dir, attr_dir, retain_dir, eval_forget, eval_others, eval_attr_dir]:
        mkdir(p)

    used_imgs = set()

    # Write forget train
    print("Writing forget_identity/person_A ...")
    for img in tqdm(forget_train_imgs, desc="forget_train"):
        used_imgs.add(img)
        place_file(src_path(img), os.path.join(forget_dir, img), mode=LINK_MODE)

    # Signature for attribute matching
    print("Selecting attribute-matched identities...")
    ref_sig = identity_attr_signature(id_map[forget_id], attr_map, MATCH_ATTRS)
    if any(ref_sig[a] is None for a in MATCH_ATTRS):
        raise RuntimeError("Reference identity missing attributes for MATCH_ATTRS; check attr file alignment.")

    matched_ids = []
    for pid in valid_ids:
        if pid == forget_id:
            continue
        sig = identity_attr_signature(id_map[pid], attr_map, MATCH_ATTRS)
        ok = True
        for a in MATCH_ATTRS:
            if sig[a] is None or sig[a] != ref_sig[a]:
                ok = False
                break
        if ok:
            matched_ids.append(pid)

    random.shuffle(matched_ids)
    if len(matched_ids) == 0:
        raise RuntimeError("No attribute-matched identities found. Change MATCH_ATTRS or lower constraints.")

    if len(matched_ids) < ATTR_MATCH_ID_COUNT:
        print(f"⚠️ matched_ids only {len(matched_ids)} (< {ATTR_MATCH_ID_COUNT}). Using all matched.")
        ATTR_MATCH_ID_COUNT_USED = len(matched_ids)
    else:
        ATTR_MATCH_ID_COUNT_USED = ATTR_MATCH_ID_COUNT

    matched_ids = matched_ids[:ATTR_MATCH_ID_COUNT_USED]
    print(f"Attribute-matched identities picked: {len(matched_ids)}")

    # Write attribute_matched
    written_attr = 0
    for i, pid in enumerate(tqdm(matched_ids, desc="attr_identities")):
        out = os.path.join(attr_dir, f"person_{i+1:03d}")
        mkdir(out)
        imgs = sample_at_most(id_map[pid], ATTR_MATCH_N_PER_ID, avoid_set=used_imgs)
        for img in imgs:
            used_imgs.add(img)
            place_file(src_path(img), os.path.join(out, img), mode=LINK_MODE)
            written_attr += 1
    print(f"attribute_matched images written: {written_attr}")

    # Write retain set (identity-balanced)
    print("Building retain set (identity-balanced)...")
    excluded_ids = set([forget_id] + matched_ids)

    retain_candidates = [pid for pid, imgs in id_map.items() if pid not in excluded_ids and len(imgs) >= 5]
    random.shuffle(retain_candidates)

    if len(retain_candidates) < RETAIN_ID_COUNT:
        print(f"⚠️ retain_candidates only {len(retain_candidates)} (< {RETAIN_ID_COUNT}). Using all candidates.")
        RETAIN_ID_COUNT_USED = len(retain_candidates)
    else:
        RETAIN_ID_COUNT_USED = RETAIN_ID_COUNT

    retain_ids = retain_candidates[:RETAIN_ID_COUNT_USED]

    written_retain = 0
    for pid in tqdm(retain_ids, desc="retain_identities"):
        imgs = sample_at_most(id_map[pid], RETAIN_N_PER_ID, avoid_set=used_imgs)
        for img in imgs:
            used_imgs.add(img)
            place_file(src_path(img), os.path.join(retain_dir, img), mode=LINK_MODE)
            written_retain += 1
    print(f"retain images written: {written_retain}")

    # Re-ID eval: queries
    print("Preparing re-identification eval set...")
    written_q = 0
    for img in tqdm(forget_eval_imgs, desc="reid_queries_person_A"):
        used_imgs.add(img)
        place_file(src_path(img), os.path.join(eval_forget, img), mode=LINK_MODE)
        written_q += 1
    print(f"re-id queries (person_A) written: {written_q}")

    # Re-ID eval: negatives
    other_candidates = [pid for pid, imgs in id_map.items()
                        if pid not in excluded_ids and pid not in set(retain_ids) and len(imgs) >= 2]
    random.shuffle(other_candidates)

    if len(other_candidates) < EVAL_OTHER_ID_COUNT:
        print(f"⚠️ other_candidates only {len(other_candidates)} (< {EVAL_OTHER_ID_COUNT}). Using all candidates.")
        EVAL_OTHER_ID_COUNT_USED = len(other_candidates)
    else:
        EVAL_OTHER_ID_COUNT_USED = EVAL_OTHER_ID_COUNT

    other_ids = other_candidates[:EVAL_OTHER_ID_COUNT_USED]

    written_neg = 0
    for pid in tqdm(other_ids, desc="reid_other_ids"):
        imgs = sample_at_most(id_map[pid], EVAL_OTHER_N_PER_ID, avoid_set=used_imgs)
        for img in imgs:
            used_imgs.add(img)
            place_file(src_path(img), os.path.join(eval_others, img), mode=LINK_MODE)
            written_neg += 1
    print(f"re-id negatives written: {written_neg}")

    # Attr preservation eval
    print("Preparing attribute preservation eval set...")
    for attr in EVAL_ATTRS:
        out = os.path.join(eval_attr_dir, attr)
        mkdir(out)

        candidates = [img for img, attrs in attr_map.items()
                      if img not in used_imgs and attrs.get(attr, -1) == 1]

        if len(candidates) == 0:
            print(f"⚠️ No candidates for attr={attr}. Skipping.")
            continue

        pick = candidates if len(candidates) <= EVAL_ATTR_N else random.sample(candidates, EVAL_ATTR_N)

        for img in tqdm(pick, desc=f"attr_eval_{attr}"):
            used_imgs.add(img)
            place_file(src_path(img), os.path.join(out, img), mode=LINK_MODE)

        print(f"attr {attr}: wrote {len(pick)} images")

    print("\nCelebA data_CelebA preparation complete.")
    print(f"Output root: {os.path.abspath(OUT_ROOT)}")
    print(f"Forget train: {FORGET_TRAIN_N}, Forget eval queries: {FORGET_EVAL_N}")
    print(f"Attr-matched: {len(matched_ids)} ids × up to {ATTR_MATCH_N_PER_ID} imgs")
    print(f"Retain: {len(retain_ids)} ids × up to {RETAIN_N_PER_ID} imgs (actual {written_retain})")
    print(f"ReID negatives: {len(other_ids)} ids × up to {EVAL_OTHER_N_PER_ID} imgs (actual {written_neg})")
    print(f"Attr eval per attr: up to {EVAL_ATTR_N}")
    print(f"LINK_MODE: {LINK_MODE}")

if __name__ == "__main__":
    main()

