# PTP-Simulation: Past-Token Prediction for Long-Context Diffusion Policies

This repository contains the implementation of 'Learning Long-Context Diffusion Policies via Past-Token Prediction' paper, specifically focused on simulated manipulation tasks in ManiSkill environment.

## Overview

This implementation demonstrates Past-Token Prediction (PTP), based on the [Learning Long-Context Diffusion Policies via Past-Token Prediction](https://arxiv.org/abs/2505.09561) paper, a novel approach that improves the performance of long-context diffusion policies by explicitly regularizing the retention of past information. The method addresses the challenge where recent diffusion policies often fail to capture essential dependencies between past and future actions.

## Key Features

- Implementation of Past-Token Prediction (PTP) for diffusion policies
- Support for long-context observation processing
- Action predictability evaluation tools
- Training scripts for ManiSkill simulation environment

## Installation

```bash
# Create conda environment
conda create -n ptp_sim python=3.8
conda activate ptp_sim

# Install PyTorch and other dependencies
pip install torch torchvision
pip install mani-skill
pip install diffusers transformers
```

## Usage

### Training

To train a PTP model:

```bash
# Start training
./train.sh
```

### Evaluation

To evaluate action predictability:

```bash
# Run evaluation
./eval.sh
```

## File Structure

- `train_ptp.py`: Main training script implementing PTP
- `eval_action_predictability.py`: Evaluation script for measuring action dependencies
- `train.sh`: Training script with hyperparameters
- `eval.sh`: Evaluation script with configuration

## Citation

If you find this code useful, please cite:

```bibtex
@article{torne2025learning,
  title={Learning Long-Context Diffusion Policies via Past-Token Prediction},
  author={Torne, Marcel and Tang, Andy and Liu, Yuejiang and Finn, Chelsea},
  journal={arXiv preprint arXiv:2505.09561v2},
  year={2025}
}
```

and

[1] Gu, J., Xiang, F., Li, X., Ling, Z., Liu, X., Mu, T., Tang, Y., Tao, S., Wei, X., Yao, Y., Yuan, X., Xie, P., Huang, Z., Chen, R., & Su, H. (2023). ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills. ICLR 2023. https://arxiv.org/abs/2302.04659


