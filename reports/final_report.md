# CNN vs. Vision Transformer Efficiency Trade-offs: A Replication on DermaMNIST and an Extension to PathMNIST

## Abstract

This report replicates a subset of Amangeldi et al.'s comparison of ResNet18 and Vision Transformer (ViT) architectures on DermaMNIST, then extends the comparison to PathMNIST, a medical imaging dataset the original study does not evaluate. We reproduce three of the paper's six DermaMNIST configurations — the ResNet18 baseline, ViT-Base (patch size 32), and ViT-Small (patch size 16), the configuration the original authors identify as the strongest accuracy-efficiency trade-off. Our replication reproduces the paper's architectural and computational-cost figures (parameter counts, FLOPs) almost exactly, but trails its reported accuracy by a consistent 6.5 to 8.2 points across all three models, a gap attributable to training hyperparameters the original paper does not specify. More notably, the specific trade-off the paper reports — ViT-Small (P16) nearly matching the largest model's accuracy at a fraction of its cost — does not appear in our DermaMNIST replication, where all three models perform within a point of each other. It does appear, more clearly than in our own DermaMNIST results, when the same pipeline is applied to PathMNIST: ViT-Small (P16) there outperforms both other models by roughly 2.3 points while remaining the more efficient of the two ViT variants. We conclude that the paper's central finding generalizes unevenly across medical imaging subdomains, and, under our setup, holds more clearly on a dataset the original study never tested than on the one it was built around.

## Introduction

Deploying deep learning models for real-time medical image classification, whether for dermatological screening on a smartphone or diagnostic support in resource-limited clinical settings, requires balancing predictive accuracy against inference speed and memory footprint. Amangeldi et al. (2025) address this trade-off directly, comparing a ResNet18 baseline against a family of Vision Transformer variants on two benchmarks: Tiny ImageNet and DermaMNIST, a seven-class dermatoscopic image dataset. Their central claim is that a moderately-sized ViT configuration, ViT-Small with a 16-pixel patch size, offers the best practical balance of the models they test, coming within a few points of their most accurate model's performance while requiring a fraction of its computational cost.

This report has two aims. The first is to replicate the DermaMNIST portion of that comparison as faithfully as the paper's methodology allows, and to report honestly where our results diverge from theirs and why. The second is to test whether the paper's specific conclusion — that ViT-Small (P16) represents the best available trade-off — holds outside the dataset it was derived from, by applying the identical pipeline to PathMNIST, a nine-class colorectal tissue pathology dataset from the same MedMNIST collection that the original study does not include.

## Research Question

Does the accuracy-efficiency trade-off reported by Amangeldi et al. for CNN and ViT architectures on DermaMNIST generalize to a different medical imaging subdomain, or is it specific to the dataset on which it was originally observed?

## Background

Convolutional networks such as ResNet (He et al., 2016) rely on local receptive fields and residual connections to extract spatial features efficiently, an inductive bias well suited to structured, lower-resolution data but limited in modeling long-range dependencies across an image. Vision Transformers (Dosovitskiy et al., 2020) instead divide an image into fixed-size patches and process them with self-attention, which captures global relationships between distant regions of an image at the cost of requiring substantially more training data to learn effectively, a well-documented weakness on small or low-resolution datasets. DermaMNIST and PathMNIST, both drawn from the MedMNIST v2 collection (Yang et al., 2023), are lightweight benchmarks built specifically to make this kind of comparison tractable at low computational cost, differing from each other in image domain, class count, and dataset size.

## Datasets

**DermaMNIST**: seven skin lesion classes, 10,000 images total (7,000 train / 1,000 validation / 2,000 test in the original paper's split; our downloaded copy returned 7,007 / 1,003 / 2,005, a negligible difference attributable to package version). Images are natively 28x28 grayscale, resized to 224x224 and replicated across three channels for compatibility with ImageNet-pretrained models.

**PathMNIST**: nine colorectal tissue classes, 107,180 images total (89,996 train / 10,004 validation / 7,180 test). Images are natively 28x28 RGB. Given the dataset's size relative to DermaMNIST and the time constraints of this project, we used a stratified subsample of 14,000 training images (evenly balanced across all nine classes), with 2,000 validation and 3,000 test images sampled the same way — a scope reduction directly analogous to the one the original paper applies to Tiny ImageNet, which it reduces from 200 classes and 100,000 training images down to 30 classes and 15,000 images for the same stated reason. PathMNIST's test split is drawn from a separate patient and institutional source than its training and validation splits, a deliberate design choice intended to measure cross-institution generalization rather than same-distribution performance.

## Methodology

We trained three model configurations on each dataset: ResNet18 (ImageNet-pretrained, fine-tuned with a replaced classification head), ViT-Base with 32-pixel patches (the paper's designated target model), and ViT-Small with 16-pixel patches (the paper's reported best trade-off). All models used identical preprocessing: resize to 224x224, random horizontal flip, random rotation up to 10 degrees, mild color jitter, and normalization with ImageNet's channel statistics, with augmentation disabled for validation and test evaluation.

The original paper reports that hyperparameters were tuned on ViT-Base and transferred to the other variants, but does not specify the optimizer, learning rate, batch size, or epoch count used. In the absence of this information, we trained all models under one fixed configuration to preserve a controlled, internally consistent comparison: AdamW with an initial learning rate of 3e-4, cosine annealing over 20 epochs, and batch size 32. Model weights were selected from the epoch with the best validation accuracy rather than the final epoch, since early experiments on DermaMNIST showed validation accuracy could decline well before training loss plateaued, indicating overfitting in later epochs.

Five metrics were logged for each model on each dataset: test accuracy, total training time, per-sample inference time (averaged over 50 forward passes after a warmup period), parameter count, and FLOPs. An efficiency ratio (accuracy divided by the sum of inference time and parameter count, in millions) was computed for each model to summarize the accuracy-cost trade-off in a single number, following the same general form as the paper's own metric; our attempts to reproduce the paper's specific reported ER values from its own accuracy, inference time, and parameter figures did not succeed, suggesting an unstated detail in their formula's units or construction. We therefore treat our ER values as internally comparable across our own three models rather than directly comparable to the paper's reported figures.

## Experimental Setup

All experiments were run on a single NVIDIA Tesla T4 GPU via Google Colab, using PyTorch and the `timm` library for ViT implementations. The original paper's experiments were run on a mix of Kaggle (Tesla P100) and Colab (Tesla T4) hardware; differences in raw GPU throughput between a P100 and T4 account for part of the training and inference time gap reported below.

## Results

**DermaMNIST**

| Model | Accuracy (%) — Paper | Accuracy (%) — Ours | Train Time (s) — Paper | Train Time (s) — Ours | Inference (ms) — Paper | Inference (ms) — Ours | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|---|---|---|
| ResNet18 | 80.26 | 73.72 | 176.35 | 1006.76 | 1.31 | 3.12 | 11.18 | 1.82 |
| ViT-Base (P32) | 80.46 | 74.06 | 357.40 | 1881.26 | 1.80 | 8.29 | 87.38 | 4.36 |
| ViT-Small (P16) | 81.56 | 73.37 | 430.86 | 2034.77 | 1.98 | 7.77 | 21.57 | 4.24 |

**PathMNIST (extension; no paper baseline available)**

| Model | Accuracy (%) | Train Time (s) | Inference (ms) | Params (M) | FLOPs (G) | ER |
|---|---|---|---|---|---|---|
| ResNet18 | 89.79 | 1890.43 | 3.11 | 11.18 | 1.82 | 6.28 |
| ViT-Base (P32) | 89.72 | 3718.14 | 5.52 | 87.38 | 4.36 | 0.97 |
| ViT-Small (P16) | 92.09 | 4053.46 | 5.34 | 21.57 | 4.24 | 3.42 |

## Analysis

Parameter counts and FLOPs match the paper's reported values almost exactly for all three models on DermaMNIST, confirming the architectures themselves were implemented correctly. Accuracy trails the paper's figures by a consistent 6.5 to 8.2 points, similar in magnitude across a small CNN and a much larger transformer alike, which points to our shared, undifferentiated training recipe as the more likely explanation than an error specific to one model.

A closer per-class examination of the ResNet18 DermaMNIST result supports this reading. DermaMNIST's majority class accounts for roughly two-thirds of the test set; a model that always predicted this class would score 66.9%, while our ResNet18 reached 72.2% on a repeated evaluation run, with meaningful recall (43-52%) on four of the seven classes, evidence of genuine class-specific learning rather than majority-class default behavior. The exception is the rarest class in the dataset, with only 29 test samples, where the model achieved zero recall, consistent with training under plain, unweighted cross-entropy loss on an imbalanced dataset, a limitation the original paper's methodology does not address either.

The more striking divergence from the paper is not one of magnitude but of pattern. Where the paper reports a real, if modest, accuracy spread across its three tested configurations (80.26 to 81.56), with ViT-Small (P16) nearly matching the top score at a fraction of the cost, our DermaMNIST replication shows all three models clustered within a point of each other (73.37 to 74.06), with ResNet18 winning cleanly on every efficiency metric and no accuracy advantage among the ViT variants large enough to justify their added cost. Under our training setup, DermaMNIST does not reproduce the paper's central claim.

The PathMNIST extension complicates this picture rather than resolving it. There, the same three models under the identical pipeline produce a genuine accuracy spread: ViT-Small (P16) reaches 92.09%, roughly 2.3 points ahead of both ResNet18 (89.79%) and ViT-Base (89.72%), while still requiring a quarter of ViT-Base's parameters and matching or beating it on inference speed. This is closer to the paper's own reported pattern than anything observed in our DermaMNIST replication, despite PathMNIST being a dataset the original study never tested. Taken together, the two results suggest that the specific advantage the paper attributes to ViT-Small (P16) is not a fixed property of the architecture but depends on properties of the dataset it is applied to, plausibly its scale, class separability, or resolution, in ways a single-dataset study cannot establish on its own.

All three PathMNIST models also show a consistent 7-8 point gap between validation and test accuracy, which we attribute to PathMNIST's deliberate train/validation-versus-test institutional split rather than to any modeling issue, since the gap's size and direction are consistent across architecturally different models.

## Limitations

- The original paper does not specify optimizer, learning rate, batch size, or epoch count; our choice of a single fixed configuration across all models, while necessary for internal consistency, likely departs from the paper's own per-model hyperparameter tuning and is the most probable source of our accuracy gap.
- Training was conducted on a single Tesla T4 GPU without a fixed random seed; a repeated ResNet18 run on DermaMNIST produced a 1.5-point difference in test accuracy from our originally logged run, indicating some run-to-run variance not captured by a single reported figure per model.
- We did not train the full set of eight ViT configurations reported in the original paper (four sizes at two patch sizes each), restricting our comparison to the three most relevant to the paper's own stated conclusion.
- The PathMNIST extension used a stratified subsample of the full dataset rather than its complete training set, a scope decision made under the same time constraints the original paper cites for its own Tiny ImageNet reduction.
- We were unable to reproduce the paper's reported Efficiency Ratio values from its own underlying accuracy, inference time, and parameter figures, suggesting the paper's formula involves an unstated unit convention or additional term; our own ER values are internally consistent but not directly comparable to the paper's.
- DermaMNIST's pronounced class imbalance was not addressed with class weighting or resampling, consistent with the original paper's apparent approach but a limitation nonetheless, particularly visible in the rarest class's zero recall.

## Future Work

A natural next step is repeating the PathMNIST comparison with per-model hyperparameter tuning rather than a single shared configuration, to test whether this narrows the accuracy gap to the paper's reported figures and whether it changes which architecture wins the trade-off on each dataset. Extending the same pipeline to additional MedMNIST subsets beyond PathMNIST would help establish whether the pattern observed here, the paper's finding holding more strongly outside its original dataset, is a general property of dataset scale and class structure or specific to the two datasets compared in this report. Applying class-balancing techniques to DermaMNIST specifically would also help separate imbalance-related accuracy loss from the architectural comparison itself.

## Conclusion

We replicated three of the six DermaMNIST configurations in Amangeldi et al.'s CNN-versus-ViT efficiency study, reproducing the paper's architectural and computational-cost figures closely while trailing its accuracy figures by a consistent, explainable margin attributable to unspecified training hyperparameters. The paper's central claim, that ViT-Small (P16) offers the best available accuracy-efficiency trade-off, did not reproduce on DermaMNIST under our setup, where all three models performed comparably and the CNN baseline won on efficiency outright. Extending the identical pipeline to PathMNIST, a dataset outside the original study's scope, produced a result closer to the paper's own pattern, with ViT-Small (P16) achieving a real accuracy advantage while remaining more efficient than the largest ViT variant. This suggests the trade-off the paper describes is real but dataset-dependent, generalizing unevenly across medical imaging subdomains rather than holding as a fixed property of the compared architectures.

## References

1. Amangeldi, A., Taigonyrov, A., Jawad, M. H., & Mbonu, C. E. (2025). CNN and ViT Efficiency Study on Tiny ImageNet and DermaMNIST Datasets. arXiv:2505.08259.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929.
4. Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 10, 41.
