# Ascend-Pangu-Infer

Inference Optimization for Open-Source Pangu 1B/7B on Huawei Ascend Platform


## 1. Project Introduction

This repo focuses on inference deployment and optimization of the open-source Pangu 1B/7B language models on Huawei Ascend chips (e.g., Ascend 910, Ascend 310). It integrates multiple optimization techniques to balance inference speed and model accuracy, making Pangu models more efficient for practical applications on Ascend hardware.


## 2. Core Features

- Supports basic inference for Pangu 1B/7B on Ascend platform.
- Implements key optimization methods:
  - **Quantization**: INT8/FP16 quantization to reduce memory usage and speed up inference.
  - **Pruning**: Structured/unstructured pruning to slim down model size without significant accuracy loss.
  - **Speculative Inference**: Uses a small "draft model" to predict high-probability tokens, reducing computational overhead of the large Pangu model.
- Provides easy-to-use scripts for environment setup, model conversion, and inference testing.
