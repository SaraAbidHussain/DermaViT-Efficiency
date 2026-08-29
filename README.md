# DermaViT-Efficiency

Replication of Amangeldi et al., ["CNN and ViT Efficiency Study on Tiny ImageNet and DermaMNIST Datasets"](https://arxiv.org/abs/2505.08259) (arXiv:2505.08259), with an extension to PathMNIST to test whether the paper's finding generalizes across medical imaging subdomains.

**Full write-up: [reports/final-report.md](reports/final-report.md)**

## What this is

The original paper compares ResNet18 against Vision Transformer variants on DermaMNIST (skin lesion classification), reporting that ViT-Small (patch size 16) offers the best accuracy-efficiency trade-off. This project:

1. Replicates the paper's ResNet18, ViT-Base (P32), and ViT-Small (P16) results on DermaMNIST
2. Extends the same pipeline to PathMNIST (colorectal tissue pathology), a dataset the original paper does not test
3. Compares whether the paper's trade-off holds on a dataset outside its original scope

**Core finding:** the paper's trade-off does not reproduce on DermaMNIST under our training setup (all three models perform within a point of each other) — but does show up, more clearly than in our own replication, on PathMNIST, where ViT-Small (P16) outperforms both other models by a real margin. See the [full report](reports/final-report.md) for the complete analysis, including honest discussion of the accuracy gap versus the original paper and why.

## Results

**DermaMNIST**

| Model | Accuracy (%) — Paper | Accuracy (%) — Ours | Params (M) | FLOPs (G) |
|---|---|---|---|---|
| ResNet18 | 80.26 | 73.72 | 11.18 | 1.82 |
| ViT-Base (P32) | 80.46 | 74.06 | 87.38 | 4.36 |
| ViT-Small (P16) | 81.56 | 73.37 | 21.57 | 4.24 |

**PathMNIST (extension)**

| Model | Accuracy (%) | Train Time (s) | Inference (ms) | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|
| ResNet18 | 89.79 | 1890.43 | 3.11 | 11.18 | 1.82 |
| ViT-Base (P32) | 89.72 | 3718.14 | 5.52 | 87.38 | 4.36 |
| ViT-Small (P16) | 92.09 | 4053.46 | 5.34 | 21.57 | 4.24 |

## Repo structure

```
DermaViT-Efficiency/
├── README.md
├── .gitignore
├── notebooks/
│   ├── resnet18_dermamnist.py
│   ├── vit_base_p32_dermamnist.py
│   ├── vit_small_p16_dermamnist.py
│   ├── resnet18_pathmnist.py
│   ├── vit_base_p32_pathmnist.py
│   └── vit_small_p16_pathmnist.py
├── results/
│   └── (comparison tables / logged metrics)
└── reports/
    ├── replication-section.md
    ├── pathmnist-extension-section.md
    └── final-report.md
```

## Running this

Each script in `notebooks/` is self-contained — install dependencies (`medmnist`, `thop`, `timm`), then run top to bottom in a GPU-enabled environment (Google Colab, T4 GPU, was used for all reported results). Training hyperparameters (AdamW, lr=3e-4, cosine annealing, 20 epochs, batch size 32) are fixed across all models for a controlled comparison, since the original paper does not specify its own values — see the Methodology section of the final report for details on this and other scope decisions (e.g. the stratified 14,000-image subsample used for PathMNIST).

## Datasets

Both DermaMNIST and PathMNIST are from the [MedMNIST v2](https://medmnist.com/) collection (Yang et al., 2023) and download automatically via the `medmnist` package.
