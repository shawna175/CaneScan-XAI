# CaneScan-XAI

**An Explainable Ensemble Deep Learning Framework for Automated Sugarcane Leaf Disease Classification**

[![Conference](https://img.shields.io/badge/IEEE-RAAICON%202026-00629B)](https://www.ieee.org/)
![Python](https://img.shields.io/badge/Python-3.x-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM-2E8B57)

CaneScan-XAI is an explainable deep-learning framework for **14-class sugarcane leaf disease classification**. The study evaluates transfer-learning CNNs, enhances the strongest models with **SE** and **CBAM** attention, combines them through **weighted soft voting**, and uses **Grad-CAM** for visual explanation. A lightweight **Streamlit** application demonstrates image-based disease prediction with explainable outputs.

> **Status:** Accepted and presented at the **5th IEEE International Conference on Robotics, Automation, Artificial-intelligence and Internet-of-Things (RAAICON 2026)**, 18–19 September 2026, Jashore University of Science and Technology (JUST), Jashore, Bangladesh.

---

## Highlights

- **9,269** sugarcane leaf images
- **14 disease/health classes**
- Baseline comparison: **Xception, InceptionV3, DenseNet201**
- Attention mechanisms: **Squeeze-and-Excitation (SE)** and **CBAM**
- Ensemble strategy: **weighted soft voting**
- Explainability: **Grad-CAM**
- Deployment prototype: **Streamlit**
- Best model: **Xception-CBAM + InceptionV3-CBAM ensemble**
- Best reported accuracy: **97.20%**

---

## Method Overview

The project follows the pipeline below:

1. Merge and prepare two publicly available sugarcane leaf datasets.
2. Resize images to **224 × 224** and split the data into **70% training, 20% validation, and 10% testing**.
3. Train and evaluate **Xception, InceptionV3, and DenseNet201** baseline models.
4. Apply **SE** and **CBAM** attention to the strongest baseline architectures.
5. Evaluate weighted soft-voting ensembles.
6. Select the best CBAM-based ensemble.
7. Generate **Grad-CAM** explanations for model predictions.
8. Demonstrate the final framework through a **Streamlit web application**.

---

## Performance

| Model / Configuration | Accuracy |
|---|---:|
| Xception | 94.72% |
| InceptionV3 | 90.62% |
| DenseNet201 | 90.19% |
| Xception + SE | 95.04% |
| Xception + CBAM | 94.50% |
| InceptionV3 + SE | 94.07% |
| InceptionV3 + CBAM | 95.15% |
| Baseline Ensemble | 97.09% |
| SE-based Ensemble | 96.34% |
| **CBAM-based Ensemble (CaneScan-XAI)** | **97.20%** |

The final CaneScan-XAI ensemble reported **97.20% accuracy, 97.37% precision, 97.20% recall, and 97.22% F1-score**.

---

## Repository Structure

```text
CaneScan-XAI/
├── README.md
├── notebooks/
│   ├── 01_baseline_models.ipynb
│   ├── 02_se_cbam_attention.ipynb
│   └── 03_ensemble_model.ipynb
├── app/
│   └── app.py
├── models/
│   └── README.md
├── assets/
│   ├── workflow.png
│   ├── gradcam_example.png
│   └── app_demo.png
├── paper/
│   └── CaneScan-XAI_Preprint.pdf
├── requirements.txt
└── .gitignore
```

> The `paper/` entry is intended for an author/preprint version that can be shared publicly under the applicable publication policy.

---

## Notebooks

### `01_baseline_models.ipynb`
Training and evaluation of the baseline CNN architectures used in the study:
- Xception
- InceptionV3
- DenseNet201
- class-weight handling for class imbalance
- training/validation evaluation

### `02_se_cbam_attention.ipynb`
Attention-enhanced experiments using:
- Squeeze-and-Excitation (SE)
- Convolutional Block Attention Module (CBAM)
- Xception and InceptionV3 backbones

### `03_ensemble_model.ipynb`
Evaluation of three ensemble configurations:
- baseline-model ensemble
- SE-enhanced ensemble
- CBAM-enhanced ensemble

The **CBAM ensemble of Xception and InceptionV3** achieved the best result and was selected as CaneScan-XAI.

---

## Streamlit Application

The repository includes a Streamlit prototype for interactive inference.

The app:
- accepts a sugarcane leaf image,
- predicts one of the 14 classes,
- reports prediction confidence,
- loads the final Xception-CBAM and InceptionV3-CBAM models,
- combines predictions using weighted soft voting,
- provides explainability visualizations using **Grad-CAM, Grad-CAM++, and Eigen-CAM**.

### Model weights

The trained model files are intentionally not tracked in the normal Git repository because they are large:

```text
fixed_xception_cbam.h5
fixed_inception_cbam.h5
```

Place the model files inside the `app/` directory before running the application. A separate release/download location can be added later for reproducibility.

### Run the app

```bash
cd app
streamlit run app.py
```

---

## Dataset

The study combines two publicly available sugarcane leaf datasets from Mendeley Data:

1. **Sugarcane Leaf Disease Dataset**  
   DOI: `10.17632/9424skmnrk.1`

2. **Sugarcane Leaf Image Dataset**  
   DOI: `10.17632/9twjtv92vk.1`

After merging, the experimental dataset contains **9,269 images across 14 classes**:

`Banded Chlorosis`, `Brown Spot`, `Brown Rust`, `Dried Leaves`, `Grassy Shoot`, `Healthy Leaves`, `Mosaic`, `Pokkah Boeng`, `Red Rot`, `Rust`, `Sett Rot`, `Smut`, `Viral Disease`, and `Yellow Leaf`.

The dataset itself is **not redistributed in this repository**.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shawna175/CaneScan-XAI.git
cd CaneScan-XAI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project uses libraries including:

- PyTorch
- torchvision
- timm
- NumPy
- Pillow
- h5py
- Streamlit
- pytorch-grad-cam

---

## Explainability

Grad-CAM is used to highlight image regions that contribute strongly to the predicted disease class. The Streamlit prototype also supports **Grad-CAM++** and **Eigen-CAM** visualizations for additional inspection of model attention.

These explanations are intended to improve model transparency and should not be interpreted as independent evidence of disease beyond the trained classifier.

---

## Limitations

The experiments use publicly available image datasets that may not capture the full variability of real sugarcane fields. Changes in lighting, background, image resolution, camera conditions, and disease appearance may affect generalization.

Future work includes:
- collecting additional field images,
- evaluating the framework under broader real-world conditions,
- improving deployment usability,
- further testing robustness and generalization.

---

## Authors

- **Mahfuz Uddin Ahmed** — Department of CSE, East West University
- **Shawna Akter** — Department of CSE, East West University  
  [GitHub](https://github.com/shawna175) · [LinkedIn](https://www.linkedin.com/in/shawna-akter)
- **Rafid Bin Taher** — Department of CSE, East West University
- **Moin Uddin Ahmed** — Department of CSE, East West University

---

## Citation

If you use this work, please cite the conference paper once the final bibliographic record/DOI is available.

```bibtex
@inproceedings{canescanxai2026,
  title     = {CaneScan-XAI: An Explainable Ensemble Deep Learning Framework for Automated Sugarcane Leaf Disease Classification},
  author    = {Ahmed, Mahfuz Uddin and Akter, Shawna and Taher, Rafid Bin and Ahmed, Moin Uddin},
  booktitle = {5th IEEE International Conference on Robotics, Automation, Artificial-intelligence and Internet-of-Things (RAAICON)},
  year      = {2026}
}
```

---

## Acknowledgment

This repository accompanies the research work presented at IEEE RAAICON 2026. The project explores how transfer learning, attention mechanisms, ensemble learning, and visual explainability can be combined for automated sugarcane leaf disease classification.

---

## License

No open-source license is currently attached to this repository. Please contact the authors before reusing the code or model weights beyond academic review and reference.
