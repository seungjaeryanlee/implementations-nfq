# Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=black)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=flake8)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=isort)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=pytest)](https://travis-ci.com/seungjaeryanlee/implementations-nfq)

This repository is a implementation of the paper [Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method](/paper.pdf).

For implementations of other deep learning papers, check the centralized [implementations](https://github.com/seungjaeryanlee/implementations) repository!

### Summary 📝

Neural Fitted Q-Iteration used a deep neural network for a Q-network, with its input being observation (s) and action (a) and its output being its action value (Q(s, a)). Instead of online Q-learning, the paper proposes **batch offline updates** by collecting experience throughout the episode and updating with that batch. The paper also suggests **hint-to-goal** method, where the neural network is trained explicitly in goal regions so that it can correctly estimate the value of the goal region.

### Results 📊

This repository uses **TensorBoard** for offline logging and **Weights & Biases** for online logging. You can see the all the metrics in [my summary report at Weights & Biases](https://app.wandb.ai/seungjaeryanlee/implementations-nfq/reports?view=seungjaeryanlee%2FSummary)!

| | |
|-|-|
| ![Test Episode Length](https://user-images.githubusercontent.com/6107926/61712085-83d23c80-ad90-11e9-9e11-326e0ab618ef.png) | ![Train Episode Length](https://user-images.githubusercontent.com/6107926/61712087-83d23c80-ad90-11e9-92be-1c43255da327.png) |

### Installation 🧱

This repository has [requirements.txt](/requirements.txt) for simple installation of PyPI packages.

```bash
pip install -r requirements.txt
```
