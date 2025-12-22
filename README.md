# 🚀 Pangu 推理性能优化

本项目致力于通过多种优化技术系统性地优化**盘古 (Pangu) 系列模型**（1B 和 7B 版本）推理阶段的端到端延迟与服务吞吐量。针对当前盘古模型缺乏量化版本的现状，我们率先实现了盘古模型的量化适配代码，并结合 NPU Graph、Chunk Pre-fill、Radix Attention 等技术开展优化工作，通过详尽的测试验证各类优化手段的实际收益，最终为用户提供盘古模型在实际应用场景下最优的部署方案指导。

### ✨ 主要目标

* **盘古量化实现与优化：** 完成盘古模型量化适配代码开发（支持 A8W8、A4W4 等多比特模式及多种量化算法），并将量化模型转化为实际的量化加速。✅
* **延迟优化：** 重点优化并测试盘古模型推理的端到端延迟（含首个 Token 延迟、后续 Token 延迟），量化评估 NPU Graph、量化、Chunk Prefill、Radix Attention 等优化策略对延迟的改善效果。✅
* **吞吐量优化：** 通过并发请求、Radix Attention及优化手段的组合应用，提升盘古模型推理服务的吞吐能力，确定不同场景下系统可达到的最大请求量和 Token 速率。🚧
* **优化策略对比：** 系统性评估 NPU Graph、量化、Chunk Prefill、Radix Attention 等技术在盘古模型上的性能收益（延迟降低、吞吐提升幅度），明确各策略的适用场景与组合最优方案。🚧

### ⚙️ 技术栈


| 组件     | 描述                  |
| -------- | --------------------- |
| LLM 模型 | 盘古 1B / 盘古 7B模型 |
| 推理框架 | vLLM                  |
| 硬件平台 | 昇腾 NPU 910B2        |
| 操作系统 | Huawei Cloud EulerOS  |


### 📂 目录说明

项目结构清晰地按照模型规模和测试类型进行了组织：
```
├── model_quant：模型量化代码
├── pangu-1B：pangu-1B推理
│   ├── latency：latency优化
│   │   ├── chunk_prefill：chunk_prefill推理分析
│   │   ├── NPU_graph：NPU_graph推理分析
│   │   ├── io_len：io_len推理分析
│   │   ├── quant：quant推理分析
│   │   ├── optim：综合优化
│   │   │   ├── optim.sh：执行代码
│   │   │   └── optim.txt：日志及结果
│   │   └── README.md：实验结果分析
│   └── throughput：throughput优化
├── pangu-7B：pangu-7B推理
│   ├── latency
│   │   ├── chunk_prefill
│   │   ├── NPU_graph
│   │   ├── io_len
│   │   ├── quant
│   │   ├── optim
│   │   └── README.md
│   └── throughput
└── README.md
```

### ⚡ 代码执行

量化代码执行参考目录下的README文件，对于推理代码：

#### 1. 环境准备

请确保您已完成 vLLM 环境、NPU/GPU 驱动及相关依赖的安装，并已将 Pangu 模型权重放置在可访问的路径中。
环境加载：
```
conda activate pangu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

```

#### 2. 配置模型路径与参数

在运行任何测试脚本前，您需要检查并修改每个 Bash 脚本内设置的模型路径和硬件配置参数：

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
--model /opt/pangu/openPangu-Embedded-1B-V1.1
```

#### 3. 执行测试

您可以针对特定模型和测试类型运行脚本。

运行延迟基准测试示例 (Pangu 1B)：

```bash
cd pangu-1B/latency/optim/
sh optim.sh
```

#### 4. 查看结果

测试目录中包含一个 results.md 文件。这些文件用于汇总和分析相应测试类型的数据。

1B 模型的延迟测试结果分析： pangu-1B/latency/results.md

7B 模型的延迟测试结果分析： pangu-7B/latency/results.md
