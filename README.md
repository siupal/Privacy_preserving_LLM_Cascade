# P³Defer: 隐私保护LLM级联框架

基于论文《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》的实现。

## 项目概述

P³Defer是一个创新的隐私保护LLM级联框架，通过链式思考(CoT)增强的策略学习实现高效的决策制定，同时保护用户隐私。该框架解决了传统LLM级联系统仅关注性能-成本权衡而忽略隐私保护的问题。

### 核心特性

- **隐私感知决策**：超越传统基于置信度和对数概率的级联方法
- **策略学习框架**：使用PPO算法优化决策过程
- **私有内存机制**：基于Levenshtein距离检测和掩码隐私令牌
- **链式思考增强**：提升本地模型的推理能力

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
python example.py
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

### PolicyNetwork & ValueNetwork

策略网络和价值网络实现了PPO算法的核心，用于优化决策过程。

### OllamaClient

提供与本地Ollama模型交互的接口，简化了模型调用过程。

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
