import os
import argparse
import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from .load_model import load_trained_fader
except ImportError:
    from load_model import load_trained_fader

TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


TRAIN_SIZE = 162770
VALID_SIZE = 19867
TEST_START_0IDX = TRAIN_SIZE + VALID_SIZE  


def resolve_fname(img_dir: str, img_id: int) -> str:
    candidates = [
        f"{img_id}.jpg", f"{img_id:06d}.jpg", f"{img_id:07d}.jpg", f"{img_id:08d}.jpg",
        f"{img_id}.png", f"{img_id:06d}.png", f"{img_id:07d}.png", f"{img_id:08d}.png",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(img_dir, c)):
            return c
    raise FileNotFoundError(f"Image id {img_id} introuvable dans {img_dir}.")


def load_preprocessed_image(img_dir: str, fname: str, device: str) -> torch.Tensor:
    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    return TFM(img).unsqueeze(0).to(device)


def onehot_single_attr(attributes: dict, img_id: int, attr_name: str, device: str):
    idx = img_id - 1
    v = bool(attributes[attr_name][idx])  # True => attr=1
    y = torch.tensor([[0., 1.]] if v else [[1., 0.]], device=device)
    return y, v


def tensor01_to_pil(x01: torch.Tensor) -> Image.Image:
    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.clamp(0, 1)
    arr = (x01 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def draw_border(img: Image.Image, color=(0, 255, 0), thickness=5) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for t in range(thickness):
        d.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)
    return out


def make_rows_image(rows, bg=(0, 0, 0)):
    W = rows[0].size[0]
    H = sum(r.size[1] for r in rows)
    canvas = Image.new("RGB", (W, H), bg)
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.size[1]
    return canvas


def pick_random_test_ids(attributes: dict, k: int, seed: int = 0):
    N = len(next(iter(attributes.values())))
    start_id = TEST_START_0IDX + 1
    end_id = N
    rng = random.Random(None if seed == 0 else seed)
    return rng.sample(list(range(start_id, end_id + 1)), k)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="random test images")
    ap.add_argument("--model_pth", required=True)
    ap.add_argument("--attr_name", required=True)

    ap.add_argument("--random_test", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--img_dir", default="Data_preprocessed/Images_Preprocessed")
    ap.add_argument("--attr_pth", default="Data_preprocessed/attributes.pth")

    ap.add_argument("--n_interpolations", type=int, default=10)
    ap.add_argument("--alpha_min", type=float, default=1.0)
    ap.add_argument("--alpha_max", type=float, default=1.0)

    ap.add_argument("--border_color", type=str, default="0,255,0")
    ap.add_argument("--border_thickness", type=int, default=5)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae = load_trained_fader(args.model_pth, device=device, attr_name=args.attr_name, n_attr=1)

    # detect expected attribute format
    model_n_attr = getattr(ae, "n_attr", 1)

    def make_y(a: float) -> torch.Tensor:
        a = float(a)
        if model_n_attr == 2:
            return torch.tensor([[1.0 - a, a]], device=device, dtype=torch.float32)
        elif model_n_attr == 1:
            return torch.tensor([[a]], device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported ae.n_attr={model_n_attr} (expected 1 or 2)")

    attributes = torch.load(args.attr_pth, weights_only=False)
    if args.attr_name not in attributes:
        raise KeyError(f"Attribut '{args.attr_name}' absent de attributes.pth.")

    # random IDs then sort for stable folder name
    img_ids = pick_random_test_ids(attributes, k=args.random_test, seed=args.seed)
    img_ids_sorted = sorted(img_ids)
    ids_str = "-".join(str(i) for i in img_ids_sorted)

    print("Random TEST img_ids:", img_ids_sorted)

   
    
    out_dir = os.path.join(
        "results",
        "trained_models",
        "random_test_images",
        args.attr_name.lower(),
        f"ids_{ids_str}",
    )
    os.makedirs(out_dir, exist_ok=True)

    bc = tuple(int(x) for x in args.border_color.split(","))
    bt = args.border_thickness

    alphas = np.linspace(1 - args.alpha_min, args.alpha_max, args.n_interpolations)

    rows = []
    meta_lines = []

    for img_id in img_ids_sorted:
        fname = resolve_fname(args.img_dir, img_id)
        x = load_preprocessed_image(args.img_dir, fname, device)

        y_true_onehot, v = onehot_single_attr(attributes, img_id, args.attr_name, device)
        enc = ae.encode(x)

        before01 = (x + 1) / 2
        before_pil = tensor01_to_pil(before01)
        before_bordered = draw_border(before_pil, color=bc, thickness=bt)

        # recon with true label
        if model_n_attr == 2:
            y_true = y_true_onehot
        else:
            y_true = make_y(1.0 if v else 0.0)

        recon01 = (ae.decode(enc, y_true)[-1] + 1) / 2
        recon_pil = tensor01_to_pil(recon01)

        # direction-aware interpolation
        inter_pils = []
        for a in alphas:
            a = float(a)
            a_dir = (1.0 - a) if v else a
            ya = make_y(a_dir)
            out01 = (ae.decode(enc, ya)[-1] + 1) / 2
            inter_pils.append(tensor01_to_pil(out01))

        cells = [before_bordered, recon_pil] + inter_pils

        W, H = before_pil.size
        row = Image.new("RGB", (len(cells) * W, H), (0, 0, 0))
        for j, cell in enumerate(cells):
            row.paste(cell, (j * W, 0))
        rows.append(row)

        meta_lines.append(f"{img_id},{fname},true_attr={v}")
        print(f"  - {img_id} ({fname}) true_attr={v}")

    grid = make_rows_image(rows)

     
    out_png = os.path.join(out_dir, "interpolations.png")
    grid.save(out_png)

    out_meta = os.path.join(out_dir, f"meta_{ids_str}.txt")
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write(f"model={args.model_pth}\n")
        f.write(f"attr={args.attr_name}\n")
        f.write(f"ae.n_attr={model_n_attr}\n")
        f.write(f"random_test={args.random_test}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"img_ids={img_ids_sorted}\n")
        f.write(f"alpha_min={args.alpha_min}\n")
        f.write(f"alpha_max={args.alpha_max}\n")
        f.write(f"n_interpolations={args.n_interpolations}\n")
        f.write(f"border_color={bc}\n")
        f.write(f"border_thickness={bt}\n")
        f.write("alphas=" + ",".join([str(a) for a in alphas]) + "\n")
        f.write("\nper_image:\n")
        for line in meta_lines:
            f.write(line + "\n")

    print("Saved:", out_png)
    print("Saved:", out_meta)
    print("Done.")


if __name__ == "__main__":
    main()
