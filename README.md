# Projet_MLA — Fader Networks appliqués à CelebA

## Présentation du projet

Ce projet implémente un **pipeline complet de Fader Networks** appliqué au dataset **CelebA**.
L’objectif est de modifier **des attributs binaires du visage** (genre, lunettes, âge, sourire, etc.)
tout en conservant l’identité de la personne.

Le dépôt couvre :
- le **prétraitement** du dataset CelebA,
- l’**implémentation de l’architecture** des Fader Networks,
- l’**entraînement** des modèles pour différents attributs,
- la **génération d’interpolations** et l’analyse qualitative des résultats.

Toutes les commandes doivent être exécutées depuis la racine du projet :
```bash
Projet_MLA/
```

---

## 1. Dépendances et installation

### 1.1 Python
- Version recommandée : **Python 3.12**

### 1.2 Installation des dépendances
```bash
pip install torch torchvision pillow numpy tqdm matplotlib
```

---

## 2. Dataset CelebA

Le dataset **CelebA n’est pas inclus dans le dépôt GitHub** en raison de sa taille.

Lien officiel :
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Fichiers requis :
- img_align_celeba.zip
- list_attr_celeba.txt
- list_eval_partition

Organisation attendue :
```text
Projet_MLA/
└── CelebA/
    └── CelebA/
        └── images/
        └── Ano 
        └── Eval  
```

---

## 3. Prétraitement

Commande à lancer depuis la racine :
```bash
python preprocess/pretraitement_data.py
```

Sorties :
```text
Data_preprocessed/
├── Images_Preprocessed/
└── attributes.pth
```

---

## 4. Architecture (`codes_source/`)

- Encoder.py
- Decoder.py
- Discriminator.py
- loader.py
- entrainement/

---

## 5. Entraînement

```bash
python -m codes_source.entrainement.train_fader --attr  'nom_attribut' --root_dir . --out_dir modeles_entraines --epochs 500 --epoch_size 50000 --batch_size 32 --ckpt_every 5000 --save_every 2000 --log_every 200
```
  
 
---

## 6. Tests & Interpolations

Tests — Modèles entraînés (par l’équipe)

### 7.1 Où sont les modèles ?
Dans `Modeles_entraines/<attribut>/<attribut>.pth` :
- `Modeles_entraines\male\male.pth`
- `Modeles_entraines\eyeglasses\eyeglasses.pth`
- `Modeles_entraines\smiling\smiling.pth`
- `Modeles_entraines\young\young.pth`

vous trouverez aussi les logs des entrainements. 

### 7.2 Scripts disponibles
Dans `test_modeles_entraines/` :
- `test_trained_models_idx.py` : test sur IDs choisis (`--img_ids`)
- `test_trained_models_random.py` : test sur images random du split test (`--random_test`)

---

## 8) Commandes — Modèles entraînés

> **Important :** vous pouvez choisir **autant d’images que vous voulez**   :  
> - `--img_ids id1 id2 ...` (idx)  
> - `--random_test K` (random)

### 8.1 Test sur IDs précis (idx)

#### Male (3 images)
```powershell
python .\test_modeles_entraines\test_trained_models_idx.py --model_pth .\Modeles_entraines\male\male.pth --attr_name Male --img_ids 202524 202576 202595 --alpha_min 2 --alpha_max 2 --n_interpolations 10
```

#### Eyeglasses (5 images)
```powershell
python .\test_modeles_entraines\test_trained_models_idx.py --model_pth .\Modeles_entraines\eyeglasses\eyeglasses.pth --attr_name Eyeglasses --img_ids 202577 202583 202595 202505 202567 --alpha_min 2 --alpha_max 2 --n_interpolations 10
```

#### Young (gain fort, exemple alpha=10)
```powershell
python .\test_modeles_entraines\test_trained_models_idx.py --model_pth .\Modeles_entraines\young\young.pth --attr_name Young --img_ids 202577 202583 202595 202505 202567 --alpha_min 10 --alpha_max 10 --n_interpolations 10
```

### 8.2 Test random (K images du split test)

#### Male (K=5) par exemple 
```powershell
python .\test_modeles_entraines\test_trained_models_random.py --model_pth .\Modeles_entraines\male\male.pth --attr_name Male --random_test 5 --seed 0 --alpha_min 2 --alpha_max 2 --n_interpolations 10
```

#### Smiling (K=8)
```powershell
python .\test_modeles_entraines\test_trained_models_random.py --model_pth .\Modeles_entraines\smiling\smiling.pth --attr_name Smiling --random_test 8 --seed 0 --alpha_min 2 --alpha_max 2 --n_interpolations 10
```

---

---

## 7. Résultats

Les résultats sont générés dans :
```text
results/
```

---

 
