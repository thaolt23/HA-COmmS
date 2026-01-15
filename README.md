# HA-SemCom  
**Hedge Algebra–Guided Semantic Communication for UAV Image Transmission**

This repository provides a **research-oriented prototype implementation** of  
**HA-SemCom**, a semantic communication framework guided by **Hedge Algebra (HA)**,  
designed for **UAV image transmission** with a strong focus on:

- **Explainability**
- **Task-oriented communication**
- **Low computational complexity (O(n))**
- **Embedded-system friendliness**

---

## 🔍 Background & Motivation

Traditional communication systems follow the **Shannon paradigm**, focusing on
bit-level fidelity while ignoring the *meaning* and *task relevance* of transmitted data.

Recent advances in **semantic communication** (e.g., DeepJSCC) demonstrate that
task-aware transmission can significantly improve robustness and efficiency.
However, existing approaches still suffer from:

- ❌ Black-box attention mechanisms (lack of explainability)
- ❌ High computational complexity (O(n²))
- ❌ Limited suitability for UAV embedded platforms

**HA-SemCom** addresses these limitations by introducing **Hedge Algebra** as a
**semantic abstraction layer**, enabling **structured, interpretable, and efficient**
semantic communication.

---

## ✨ Key Contributions

- ✅ Hedge Algebra–guided semantic importance modeling  
- ✅ Explicit **linguistic semantic levels** (*very low → very high*)  
- ✅ Explainable resource allocation decisions  
- ✅ Attention-free design with **O(n)** complexity  
- ✅ End-to-end semantic communication prototype  
- ✅ Suitable for UAV and edge AI scenarios  

---

## 🧠 System Overview

### Pipeline

```text
Image
  ↓
Feature Extractor (CNN)
  ↓
Importance Network (O(n))
  ↓
Hedge Algebra Mapping
  ↓
HA-Guided Bit Allocation
  ↓
Wireless Channel (AWGN)
  ↓
Decoder / Reconstruction
  ↓
Explainability Report
```

### Interpretation

- Only a small fraction of image regions are semantically critical  
- Transmission resources are selectively allocated  
- Decisions are fully interpretable using linguistic semantics  

---

## 📊 Hedge Algebra Distribution Example

An example output of the explainability module:

```text
=== HA Explainability Report ===
very low     : 26.8%
low          : 18.4%
little low   : 14.9%
medium       : 21.2%
little high  : 9.5%
high         : 6.1%
very high    : 3.1%
```

This distribution shows that only a small portion of image regions is assigned
*high* or *very high* semantic importance, confirming the task-oriented and
resource-efficient nature of HA-SemCom.

---

## 📁 Repository Structure

```text
ha_semcom/
│
├── main.py              # Training + HA report
├── model.py             # Feature, Importance, Detection
├── hedge_algebra.py     # HA definition
├── dataset.py           # UAV-style dataset
├── explain.py           # Explainability output
│
├── figures/
│   ├── pipeline.png
│   └── ha_distribution.png
│
├── requirements.txt
├── LICENSE
└── README.md

```

---

## ⚙️ Requirements

- Python ≥ 3.8  
- PyTorch (CPU version)  
- torchvision  
- numpy  
- matplotlib  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run a simple end-to-end demo:

```bash
python main.py
```

The script performs:
- Feature extraction and importance estimation  
- Hedge Algebra semantic mapping  
- HA-guided quantization and transmission  
- Image reconstruction  
- Generation of a **Hedge Algebra explainability report**

---

## 🧪 Implemented Algorithms

- **Algorithm 1**: Importance Network with linear complexity  
- **Algorithm 2**: Hedge Algebra–guided semantic quantization  
- **Algorithm 3**: Semantic feature transmission over a wireless channel  
- **Algorithm 4**: End-to-end HA-SemCom pipeline  
- **Algorithm 5**: Explainability module  

These algorithms correspond directly to the HA-SemCom framework described in the paper.

---

## 📖 Research Context

This repository accompanies the research idea:

> **HA-SemCom: Hedge Algebra–Guided Semantic Communication for UAV Image Transmission**

The work bridges multiple research directions:
- Semantic communication (DeepJSCC)  
- Task-oriented communication  
- Explainable AI  
- Edge and UAV systems  

---

## 🚧 Limitations

- Research prototype (not optimized for real-time deployment)  
- Single-image processing (no video support)  
- Fixed Hedge Algebra parameters  

---

## 🔮 Future Work

- Adaptive Hedge Algebra parameters  
- Video-based semantic communication  
- Multi-UAV cooperative transmission  
- Direct comparison with DeepJSCC baselines  

---

## 📜 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

---

## 🎓 Academic Use & Citation

This repository is intended for:
- Academic research  
- Thesis and dissertation experiments  
- Seminar and teaching demonstrations  

If you use this code in academic work, please cite the corresponding paper.
