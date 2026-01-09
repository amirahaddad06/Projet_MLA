import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Expressions régulières pour extraire les valeurs du fichier de log
STEP_RE = re.compile(
    r"step=(\d+)\s+rec=([0-9.eE+-]+)\s+dis=([0-9.eE+-]+)\s+adv=([0-9.eE+-]+)\s+lam=([0-9.eE+-]+)\s+accD=([0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"\[VAL\]\s+epoch=(\d+)\s+val_rec=([0-9.eE+-]+)"
)

def parse_log(log_path: Path):
    steps, rec, dis, adv, lam, accD = [], [], [], [], [], []
    val_epoch, val_rec = [], []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                rec.append(float(m.group(2)))
                dis.append(float(m.group(3)))
                adv.append(float(m.group(4)))
                lam.append(float(m.group(5)))
                accD.append(float(m.group(6)))
                continue

            v = VAL_RE.search(line)
            if v:
                val_epoch.append(int(v.group(1)))
                val_rec.append(float(v.group(2)))

    return {
        "steps": steps, "rec": rec, "dis": dis, "adv": adv, "lam": lam, "accD": accD,
        "val_epoch": val_epoch, "val_rec": val_rec
    }

def save_plot(x, y, xlabel, ylabel, title, out_path: Path):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Chemin vers train.log (ex: log_training/train_male.log)")
    ap.add_argument("--out_dir", default="losses", help="Dossier de sortie des figures")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    
    # Créer le dossier de sortie s'il n'existe pas
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charger les données du fichier log
    data = parse_log(log_path)

    if len(data["steps"]) == 0:
        raise RuntimeError("Aucune ligne 'step=...' trouvée. Vérifie le format du log.")

    # Sauvegarder les courbes pour chaque type de perte
    save_plot(data["steps"], data["rec"], "step", "rec loss", "Train reconstruction loss (rec)", out_dir / "train_rec.png")
    save_plot(data["steps"], data["dis"], "step", "dis loss", "Train discriminator loss (dis)", out_dir / "train_dis.png")
    save_plot(data["steps"], data["adv"], "step", "adv loss", "Train adversarial term (adv)", out_dir / "train_adv.png")
    save_plot(data["steps"], data["accD"], "step", "accD", "Train discriminator accuracy (accD)", out_dir / "train_accD.png")
    save_plot(data["steps"], data["lam"], "step", "lambda", "Lambda schedule (lam)", out_dir / "train_lambda.png")

    # Sauvegarder la courbe de validation si disponible
    if len(data["val_epoch"]) > 0:
        save_plot(data["val_epoch"], data["val_rec"], "epoch", "val_rec", "Validation reconstruction loss (val_rec)", out_dir / "val_rec.png")

    print(f"OK: figures sauvegardées dans: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
