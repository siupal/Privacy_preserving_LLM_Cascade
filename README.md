# P³Defer: 隐私保护LLM级联框架

基于论文《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》的实现。

## 项目概述

P³Defer是一个创新的隐私保护LLM级联框架，通过链式思考(CoT)增强的策略学习实现高效的决策制定，同时保护用户隐私。该框架解决了传统LLM级联系统仅关注性能-成本权衡而忽略隐私保护的问题。

### 核心特性

- **隐私感知决策**：超越传统基于置信度和对数概率的级联方法
- **策略学习框架**：使用PPO算法优化决策过程
- **私有内存机制**：基于Levenshtein距离检测和掩码隐私令牌
- **链式思考增强**：提升本地模型的推理能力

## 项目结构

```
Privacy_preserving_LLM_Cascade/
├── README.md                  # 项目说明文档
├── requirements.txt           # 项目依赖
├── p3defer/                   # 主要源代码目录
│   ├── __init__.py            # 包初始化文件
│   ├── p3defer.py             # P³Defer核心实现
│   ├── models/                # 模型相关代码
│   │   ├── __init__.py        # 模型包初始化
│   │   ├── ollama_client.py   # Ollama客户端
│   │   ├── private_memory.py  # 隐私内存实现
│   │   └── policy_network.py  # 策略网络实现
│   └── utils/                 # 工具函数
│       └── __init__.py        # 工具包初始化
├── examples/                  # 示例代码
│   ├── example.py             # 简单示例
│   └── run_demo.py            # 交互式演示
├── scripts/                   # 脚本文件
│   ├── train.py               # 训练脚本
│   └── evaluate.py            # 评估脚本
├── notebooks/                 # Jupyter笔记本
│   └── Cascade_LLM.ipynb      # 级联LLM实验
├── data/                      # 数据目录
│   └── privacy_tokens.json    # 隐私令牌示例
├── tests/                     # 测试代码
│   └── test_p3defer.py        # 单元测试
└── docs/                      # 文档
    └── paper.pdf              # 原始论文
```

## 系统架构

P³Defer框架包含以下主要组件：

1. **本地LLM**：部署在设备上的小型模型（如Gemma-2B）
2. **服务器LLM**：更强大的云端模型（如Gemma-7B/9B）
3. **延迟决策模块**：基于策略学习的代理，决定如何处理查询
4. **私有内存**：存储和检测隐私令牌的机制

## 安装与依赖

```bash
pip install -r requirements.txt
```

### 依赖项

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Python-Levenshtein >= 0.21.0
- NLTK >= 3.8.1
- 其他依赖见requirements.txt

## 快速开始

### 准备工作

1. 安装Ollama并启动服务
2. 确保已安装gemma:2b模型

```bash
ollama pull gemma:2b
```

### 运行示例

```bash
python examples/example.py
```

### 运行交互式演示

```bash
python examples/run_demo.py --mode interactive
```

## 使用方法

```python
from p3defer import P3Defer, PrivateMemory

# 创建P³Defer实例
p3defer = P3Defer()

# 处理查询
query = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？"
output = p3defer.process_query(query)
print(output)
```

## 主要组件说明

### PrivateMemory

私有内存组件负责检测和掩码隐私令牌，使用Levenshtein距离进行模糊匹配。

```python
from p3defer import PrivateMemory

# 创建私有内存实例
privacy_memory = PrivateMemory()

# 检测隐私令牌
query = "小明有5个苹果"
privacy_tokens = privacy_memory.detect_privacy(query)
print(f"检测到的隐私令牌: {privacy_tokens}")

# 掩码隐私信息
masked_query = privacy_memory.mask_privacy(query)
print(f"掩码后的查询: {masked_query}")
```

### PolicyNetwork & ValueNetwork

策略网络和价值网络实现了PPO算法的核心，用于优化决策过程。

### OllamaClient

提供与本地Ollama模型交互的接口，简化了模型调用过程。

```python
from p3defer.models import OllamaClient

# 创建Ollama客户端
client = OllamaClient(model_name="gemma:2b")

# 生成文本
response = client.generate("计算 2 + 2 的结果")
print(response)
```

## 训练与评估

### 训练模型

```bash
python scripts/train.py --local_model gemma:2b --train_data data/train_data.json --output_dir models
```

### 评估模型

```bash
python scripts/evaluate.py --local_model gemma:2b --test_data data/test_data.json --model_path models/p3defer_final.pt
```

## 未来工作

- 实现服务器模型加载
- 优化私有内存更新机制
- 支持多GPU分布式训练
- 扩展到更多任务和数据集

## 引用

如果您使用了本项目，请引用原论文：

```
@article{zhang2025privacy,
  title={Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning},
  author={Zhang et al.},
  journal={arXiv preprint},
  year={2025}
}
```
