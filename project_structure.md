# P³Defer 项目结构

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
│   ├── Cascade_LLM.ipynb      # 级联LLM实验
│   ├── CoT.ipynb              # 链式思考实验
│   └── evaluation.ipynb       # 评估实验
├── data/                      # 数据目录
│   └── privacy_tokens.json    # 隐私令牌示例
├── tests/                     # 测试代码
│   └── test_p3defer.py        # 单元测试
└── docs/                      # 文档
    └── paper.pdf              # 原始论文
```

## 目录说明

### p3defer/
包含框架的核心实现，包括P³Defer主类和各种模型组件。

### examples/
包含示例代码，展示如何使用P³Defer框架。

### scripts/
包含用于训练和评估模型的脚本。

### notebooks/
包含Jupyter笔记本，用于实验和分析。

### data/
包含示例数据和预训练模型。

### tests/
包含单元测试和集成测试。

### docs/
包含项目文档和相关论文。
