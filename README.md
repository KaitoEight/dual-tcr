# Dual-TCR: Dual-Branch Target Consistency Representation for Training-Free Personalized Segmentation

[![PWC](https://img.shields.io/badge/Personalized%20Segmentation-93.42%25%20mIoU-green)](https://paperswithcode.com)

Official implementation of **Dual-TCR: A Dual-Branch Target Consistency Representation for Training-Free Personalized Image Segmentation**.

## Abstract

We propose **Dual-TCR**, a training-free framework for personalized image segmentation that decouples semantic localization from geometric mask generation. By leveraging both SAM's native geometric features and NVIDIA RADIO's multi-teacher distilled semantic features, Dual-TCR achieves state-of-the-art performance of **93.42% mIoU** on the PerSeg benchmark.

## Key Features

- **🎯 Training-Free**: No fine-tuning required - ready to use with a single reference mask
- **⚡ Fast**: Inference in seconds with pre-trained models
- **🔀 Dual-Branch Architecture**: Combines SAM (geometric) + RADIO (semantic)
- **🏆 State-of-the-Art**: 93.42% mIoU on PerSeg benchmark

## News
* **NEW**: RADIO-space scoring achieves **93.42% mIoU** (vs 92.34% baseline)
* **NEW**: Ablation experiments for scoring space comparison
* **NEW**: τ parameter analysis for Hybrid Refining Module
* Release Dual-TCR implementation with dual scoring configurations

## Method Overview

Dual-TCR introduces a dual-branch architecture:

1. **Geometric Branch (SAM)**: Uses native SAM features for boundary-accurate mask generation
2. **Semantic Branch (RADIO)**: Leverages multi-teacher distilled features for robust object localization
3. **Target Consistency Representation (TCR)**: Arbitrates between branches based on feature consistency
4. **Hybrid Refining Module (HRM)**: Fuses masks using region agreement and weighted logit fusion

```
┌─────────────────────────────────────────────────────────────┐
│                    Dual-TCR Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  Reference ──► SAM Encoder ──► Geometric Branch           │
│     Mask    ──► RADIO Encoder ──► Semantic Branch          │
│                      │                                      │
│                      ▼                                      │
│              Target Consistency                             │
│              Representation (TCR)                          │
│                      │                                      │
│                      ▼                                      │
│          Hybrid Refining Module (HRM)                       │
│                      │                                      │
│                      ▼                                      │
│              Final Segmented Mask                           │
└─────────────────────────────────────────────────────────────┘
```

## Scoring Space Configurations

We provide two scoring configurations:

| Configuration | Scoring Space | mIoU | Description |
|---------------|--------------|------|-------------|
| `dual_tcr_perseg_radio.py` | RADIO-space | **93.42%** | Multi-teacher semantic supervision (Recommended) |
| `dual_tcr_perseg_sam.py` | SAM-space | 92.43% | Decoder-aligned evaluation |

## Requirements

### Installation

```bash
git clone https://github.com/KaitoEight/Personalize-SAM.git
cd Personalize-SAM

conda create -n dual_tcr python=3.8
conda activate dual_tcr

pip install -r requirements.txt
```

### Preparation

1. Download **PerSeg** dataset from [Google Drive](https://drive.google.com/file/d/18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio/view?usp=sharing)
2. Download SAM checkpoint: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
3. Organize data:
```
data/
├── Annotations/
│   ├── category_1/
│   │   ├── 00.png
│   │   ├── 01.png
│   │   └── ...
│   └── ...
└── Images/
    ├── category_1/
    │   ├── 00.jpg
    │   ├── 01.jpg
    │   └── ...
    └── ...
sam_vit_h_4b8939.pth
```

## Getting Started

### Run Dual-TCR with RADIO-space Scoring (Recommended)

```bash
python dual_tcr_perseg_radio.py --data ./data --outdir outputs/radio_scoring
```

### Run Dual-TCR with SAM-space Scoring

```bash
python dual_tcr_perseg_sam.py --data ./data --outdir outputs/sam_scoring
```

### Evaluate Results

```bash
python eval_miou.py --pred_path outputs/radio_scoring
```

## Experimental Results

### PerSeg Benchmark

| Method | Training-Free | mIoU (%) |
|--------|-------------|----------|
| PerSAM (Original) | ✅ | 89.16 |
| PerSAM-F | ❌ | 95.30 |
| **Dual-TCR (SAM-space)** | ✅ | **92.43** |
| **Dual-TCR (RADIO-space)** | ✅ | **93.42** |

### Ablation Studies

#### Scoring Space Comparison

| Scoring Space | mIoU (%) | Notes |
|---------------|----------|-------|
| RADIO-space | **93.42** | Best - multi-teacher semantic |
| DUAL-space | 92.46 | 50% SAM + 50% RADIO |
| SAM-space | 92.43 | Decoder-aligned evaluation |

#### τ Parameter Analysis (HRM)

| τ | mIoU (%) |
|---|----------|
| 0.00 | 88.13 |
| **0.01** | **88.19** |
| 0.02 | 87.97 |
| 0.05 | 87.93 |

## Repository Structure

```
Personalize-SAM/
├── dual_tcr_perseg_radio.py      # RADIO-space scoring (93.42%)
├── dual_tcr_perseg_sam.py        # SAM-space scoring (92.43%)
├── eval_miou.py                  # mIoU evaluation
├── per_segment_anything/          # Modified SAM implementation
│   ├── predictor.py              # Custom predictor with attn_sim
│   └── modeling/                # Mask decoder modifications
├── ablation_experiments/         # Ablation study scripts
│   ├── eval_prompt_localization.py
│   ├── ablation_tau_v7tcr.py
│   ├── ablation_scoring_space_full.py
│   └── RevisionLetter            # Response to reviewers
└── data/                        # PerSeg dataset (download separately)
```

## Citation

```bibtex
@article{dual_tcr_2024,
  title={Dual-TCR: A Dual-Branch Target Consistency Representation for Training-Free Personalized Image Segmentation},
  author={Le Minh Khanh and Dong Van Nguyen and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

*Note: Please update with complete author list and correct journal/conference name.*

## Acknowledgements

This work builds upon:
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [NVIDIA RADIO](https://github.com/NVlabs/RADIO)
- [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM)

## Contact

For questions about this implementation, please open an issue or contact the authors.
