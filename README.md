# Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=black)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=pytest)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)

[![numpydoc Docstring Style](https://img.shields.io/badge/docstring-numpydoc-blue.svg)](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue.svg)](.pre-commit-config.yaml)

This repository is a implementation of the paper [Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method](/paper.pdf).

For implementations of other deep learning papers, check the centralized [implementations](https://github.com/seungjaeryanlee/implementations) repository!

---

### Table of Contents 📜

- [Summary](#summary-)
- [Installation](#installation-)
- [Running](#running-)
- [Results](#results-)
- [Differences from the Paper](#differences-from-the-paper-)
- [Reproducibility](#reproducibility-)
 
---
 
### Summary 📝

Neural Fitted Q-Iteration used a deep neural network for a Q-network, with its input being observation (s) and action (a) and its output being its action value (Q(s, a)). Instead of online Q-learning, the paper proposes **batch offline updates** by collecting experience throughout the episode and updating with that batch. The paper also suggests **hint-to-goal** method, where the neural network is trained explicitly in goal regions so that it can correctly estimate the value of the goal region.

### Installation 🧱

This repository has [requirements.txt](/requirements.txt) for easy installation of PyPI packages.

```bash
pip install -r requirements.txt
```

You can read more about each package in the comments of the [requirements.txt](/requirements.txt) file!

### Running 🏃


### Results 📊

This repository uses **TensorBoard** for offline logging and **Weights & Biases** for online logging. You can see the all the metrics in [my summary report at Weights & Biases](https://app.wandb.ai/seungjaeryanlee/implementations-nfq/reports?view=seungjaeryanlee%2FSummary)!

| | |
|-|-|
| Train Episode Length | ![Train Episode Length](https://user-images.githubusercontent.com/6107926/62005353-07af6e80-b16d-11e9-8fc9-798af69de2e4.png) |
| Train Episode Length | ![Evaluation Episode Length](https://user-images.githubusercontent.com/6107926/62005354-08480500-b16d-11e9-9c03-facb5f3c6b87.png) |
| Train Episode Cost | ![Train Episode Cost](https://user-images.githubusercontent.com/6107926/62005355-08480500-b16d-11e9-9b82-6516677deec6.png) |
| Evaluation Episode Cost | ![Evaluation Episode Cost](https://user-images.githubusercontent.com/6107926/62005356-08480500-b16d-11e9-95ed-09259728e1c3.png) |
| Train Loss | ![Train Loss](https://user-images.githubusercontent.com/6107926/62005357-08480500-b16d-11e9-91ca-52368d49dce5.png) |
| Total Cycle | ![Total Cycle](https://user-images.githubusercontent.com/6107926/62005359-08e09b80-b16d-11e9-949a-88313763992d.png) |
| Total Cost | ![Total Cost](https://user-images.githubusercontent.com/6107926/62005360-08e09b80-b16d-11e9-9c89-a4f0f4e075a6.png) |

### Differences from the Paper 👥

### Reproducibility 🎯

