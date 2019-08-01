# Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method

[![black Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=black)](https://black.readthedocs.io/en/stable/)
[![flake8 Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=flake8)](http://flake8.pycqa.org/en/latest/)
[![isort Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=isort)](https://pypi.org/project/isort/)
[![pytest Build Status](https://img.shields.io/travis/com/seungjaeryanlee/implementations-nfq.svg?label=pytest)](https://docs.pytest.org/en/latest/)

[![numpydoc Docstring Style](https://img.shields.io/badge/docstring-numpydoc-blue.svg)](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue.svg)](https://pre-commit.com/)

This repository is an implementation of the paper [Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method (Riedmiller, 2005)](/paper.pdf).

**Please ⭐ this repository if you found it useful!**


---

### Table of Contents 📜

- [Summary](#summary-)
- [Installation](#installation-)
- [Running](#running-)
- [Results](#results-)
- [Differences from the Paper](#differences-from-the-paper-)
- [Reproducibility](#reproducibility-)
 
For implementations of other deep learning papers, check the **[implementations](https://github.com/seungjaeryanlee/implementations) repository**!

---
 
### Summary 📝

Neural Fitted Q-Iteration used a deep neural network for a Q-network, with its input being observation (s) and action (a) and its output being its action value (Q(s, a)). Instead of online Q-learning, the paper proposes **batch offline updates** by collecting experience throughout the episode and updating with that batch. The paper also suggests **hint-to-goal** method, where the neural network is trained explicitly in goal regions so that it can correctly estimate the value of the goal region.

### Installation 🧱

First, clone this repository from GitHub. Since this repository contains submodules, you should use the `--recursive` flag.

```bash
git clone --recursive https://github.com/seungjaeryanlee/implementations-nfq.git
```

If you already cloned the repository without the flag, you can download the submodules separately with the `git submodules` command:

```bash
git clone https://github.com/seungjaeryanlee/implementations-nfq.git
git submodule update --init --recursive
```

After cloing the repository, use the [requirements.txt](/requirements.txt) for simple installation of PyPI packages.

```bash
pip install -r requirements.txt
```

You can read more about each package in the comments of the [requirements.txt](/requirements.txt) file!

### Running 🏃

You can train the NFQ agent on Cartpole Regulator using the given configuration file with the below command:
```
python train_eval.py -c cartpole.conf
```

For a reproducible run, use the `--RANDOM_SEED` flag.
```
python train_eval.py -c cartpole.conf --RANDOM_SEED=1
```

To save a trained agent, use the `--SAVE_PATH` flag.
```
python train_eval.py -c cartpole.conf --SAVE_PATH=saves/cartpole.pth
```

To load a trained agent, use the `--LOAD_PATH` flag.
```
python train_eval.py -c cartpole.conf --LOAD_PATH=saves/cartpole.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval.py -c cartpole.conf --USE_TENSORBOARD --USE_WANDB
```

### Results 📊

This repository uses **TensorBoard** for offline logging and **Weights & Biases** for online logging. You can see the all the metrics in [my summary report at Weights & Biases](https://app.wandb.ai/seungjaeryanlee/implementations-nfq/reports?view=seungjaeryanlee%2FSummary)!

<p align="center">
  <img alt="Train Episode Length" src="https://user-images.githubusercontent.com/6107926/62005353-07af6e80-b16d-11e9-8fc9-798af69de2e4.png" width="49%">
  <img alt="Evaluation Episode Length" src="https://user-images.githubusercontent.com/6107926/62005354-08480500-b16d-11e9-9c03-facb5f3c6b87.png" width="49%">
</p>
<p align="center">
  <img alt="Train Episode Cost" src="https://user-images.githubusercontent.com/6107926/62005355-08480500-b16d-11e9-9b82-6516677deec6.png" width="49%">
  <img alt="Evaluation Episode Cost" src="https://user-images.githubusercontent.com/6107926/62005356-08480500-b16d-11e9-95ed-09259728e1c3.png" width="49%">
</p>
<p align="center">
  <img alt="Total Cycle" src="https://user-images.githubusercontent.com/6107926/62005359-08e09b80-b16d-11e9-949a-88313763992d.png" width="32%">
  <img alt="Total Cost" src="https://user-images.githubusercontent.com/6107926/62005360-08e09b80-b16d-11e9-9c89-a4f0f4e075a6.png" width="32%">
  <img alt="Train Loss" src="https://user-images.githubusercontent.com/6107926/62005357-08480500-b16d-11e9-91ca-52368d49dce5.png" width="32%">
</p>

### Differences from the Paper 👥

- From the 3 environments (Pole Balancing, Mountain Car, Cartpole Regulator), only the Cartpole Regulator environment was implemented and tested. It is the most difficult environment.
- For the Cartpole Regulator, the success state is relaxed so that the state is successful whenever the pole angle is at most 24 degrees away from upright position. In the original paper, the cart must also be in the center with 0.05 tolerance.
- Evaluation of the trained policy is only done in 1 evaluation environment, instead of 1000.

### Reproducibility 🎯

Despite having no open-source code, the paper had sufficient details to implement NFQ. However, the results were not fully reproducible: we had to relax the definition of goal states and simplify evaluation. Still, the agent was able to learn to balance a CartPole for 3000 steps while only training from 100-step environment.

Few nits:

- There is no specification of pole angle for goal and forbidden states. We set 0~24 degrees from upright position as a requirement for goal state and any state with 90+ degrees forbidden.
- The paper randomly initializes network weights within [−0.5, 0.5], but does not mention bias initialization.
- The goal velocity of the success states is not mentioned. We use a normal distribution to randomly generate velocities for the hint-to-goal variant.
- It is unclear whether to add experience after or before training the agent for each epoch. We assume adding experience before training.
- The learning rate for the Rprop optimizer is not specified.

