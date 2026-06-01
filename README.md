# Etsy AI-Generated Image Detection

Detecting AI-generated product images on the Etsy marketplace using transfer learning and end-to-end fine-tuning. Industry-partnered challenge with Etsy — CSC1187 Advanced Machine Learning, DCU, Spring 2026.

> ⚠️ **GitHub may fail to render large notebooks.** View Phase 2 on [nbviewer](https://nbviewer.org/github/shubin-li/etsy-ai-image-detection/blob/main/AdvMLProject_Phase2.ipynb)

> **Note:** This project was conducted under NDA with Etsy. The dataset is proprietary and cannot be shared. Only source code and the report are included in this repository.

---

## Background

Etsy is a global two-sided marketplace connecting millions of independent sellers with buyers. With the rise of generative models (Midjourney, DALL-E, Stable Diffusion), some sellers have begun using AI-generated images to misrepresent product listings. This project addresses the challenge of automatically detecting such images to protect marketplace integrity.

The task is framed as a **binary image classification** problem:

- `1` → AI-generated image
- `0` → Authentic product photograph

Evaluation metric: **F1 score** on a hidden test set (labels withheld from participants).

## Approach

The project follows a three-phase methodology, progressively increasing in sophistication:

### Phase 1 — Hybrid Feature Extraction + Classical ML

Frozen pretrained CNNs (ResNet50) used as feature extractors, feeding into classical classifiers (SVM, Random Forest, Gradient Boosting). This phase establishes baselines and demonstrates that deep embeddings (2048-d ResNet50 features) substantially outperform hand-crafted descriptors (YCrCb, Sobel edges, colour histograms).

**Best Phase 1 result:** ResNet50 + SVM → F1 = 0.8134

### Phase 2 — End-to-End Fine-Tuning (My Contribution)

Fine-tuning **EfficientNetV2-S** (ImageNet-pretrained, ~22M params, 384×384 input) end-to-end with BCEWithLogitsLoss. Deliberately minimal pipeline — Adam optimizer, horizontal flip augmentation only, 10 epochs — to establish a clean reference point.

**Result:** F1 = 0.9076 (+9.4 points over best Phase 1)

Key ablation findings on this single model:
- **Aggressive augmentation hurts** — RandomCrop + ColorJitter + RandomErasing dropped F1 by 1.3 points, likely destroying the subtle, spatially-localized texture artifacts that distinguish AI-generated images
- **TTA provides no benefit** — the model already learned flip-invariant predictions during training
- **Threshold tuning is noise** — the optimal threshold shift (0.50→0.53) changed only 1 prediction out of 960 validation images

### Phase 3 — Multi-Model Ensemble

Three-model ensemble (EfficientNetV2-S + FFT branch, ConvNeXt-Small, ConvNeXt-Base) with Focal Loss, Mixup, SWA, and 5-view TTA. Ensemble reached F1 = 0.9317, though notably **ConvNeXt-Base alone scored 0.9356** — demonstrating that ensemble diversity quality matters more than quantity.

## Results Summary

| Phase | Model | Val F1 |
|-------|-------|--------|
| 1 | ResNet50 (frozen) + SVM | 0.8134 |
| 2 | EfficientNetV2-S (fine-tuned) | **0.9076** |
| 3 | ConvNeXt-Base (single) | 0.9356 |
| 3 | 3-Model Ensemble + TTA | 0.9317 |

## Repository Structure

```
├── AdvMLProject_Phase2.ipynb        # Phase 2: EfficientNetV2-S fine-tuning (my code)
├── AdvMLProject_Phase1_3.ipynb      # Phase 1 & 3: baselines and ensemble (teammate's code)
├── AdvMLProject_Report.pdf          # Full IEEE-format project report
└── README.md
```

## Tech Stack

- **Deep Learning:** PyTorch, timm (EfficientNetV2-S, ConvNeXt)
- **Classical ML:** scikit-learn (SVM, Random Forest, Gradient Boosting)
- **Image Processing:** torchvision, OpenCV, PIL
- **Training Infra:** NVIDIA RTX 4070 Ti (Phase 2), Google Colab T4 (Phase 1 & 3)

## Key Takeaways

1. **End-to-end fine-tuning dominates frozen-backbone + classical ML** on forensic image classification — the backbone must adapt to learn subtle AI artifact features.
2. **More augmentation ≠ better performance** when the discriminative signal is spatially localized texture artifacts. Aggressive augmentation can destroy the very patterns the model needs to learn.
3. **Ensembles don't always help** — when component models differ substantially in quality, weaker models drag down the ensemble below the best individual model.

## Dataset

The dataset (~4,800 training images, ~2,058 test images) was provided by Etsy under NDA and is **not included** in this repository. The images contain a mix of authentic product photographs and AI-generated images from various generative models.

## Authors

- **Shubin Li** — Phase 2 (end-to-end fine-tuning, ablation studies)
- **Megha Kanojia** — Phase 1 & Phase 3 (hybrid extraction, ensemble)

MSc Computing (Data Analytics), Dublin City University, 2026
