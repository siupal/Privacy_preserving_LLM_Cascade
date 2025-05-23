# P³Defer 框架使用指南

本指南将帮助你完整实现《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》论文中的效果。

## 1. 环境准备

首先，确保你已经安装了所有必要的依赖：

```bash
# 激活你的环境
conda activate 你的环境名称

# 安装依赖
pip install -r requirements.txt
```

## 2. 数据准备

论文使用了包含隐私信息的数据集进行训练和评估。我们提供了数据处理脚本来准备这样的数据集：

```bash
# 准备数据集
python scripts/prepare_data.py --input_file data/sample_dataset.json --output_dir data --privacy_ratio 0.5 --train_ratio 0.8
```

这将生成以下文件：
- `data/train_data.json`：训练数据集
- `data/test_data.json`：测试数据集
- `data/privacy_tokens.json`：隐私令牌列表
- `data/example_data.json`：示例数据

如果你有自己的数据集，可以替换 `--input_file` 参数。

## 3. 启动Ollama服务

确保Ollama服务正在运行，并且已经拉取了必要的模型：

```bash
# 启动Ollama服务
ollama serve

# 在另一个终端中拉取模型
ollama pull gemma:2b
ollama pull gemma:7b  # 如果你有足够的资源
```

## 4. 训练模型

使用训练脚本训练P³Defer模型：

```bash
python scripts/train.py --local_model gemma:2b --server_model gemma:7b --train_data data/train_data.json --output_dir models --num_epochs 5 --batch_size 32 --learning_rate 5e-5
```

如果你的计算资源有限，可以只使用本地模型：

```bash
python scripts/train.py --local_model gemma:2b --train_data data/train_data.json --output_dir models --num_epochs 5
```

训练过程中，模型会学习何时使用本地模型、何时直接转发到服务器模型、何时在转发前掩码隐私信息。

## 5. 评估模型

使用评估脚本评估模型性能：

```bash
python scripts/evaluate.py --local_model gemma:2b --server_model gemma:7b --test_data data/test_data.json --model_path models/p3defer_final.pt --output_file results/evaluation_results.json
```

这将生成评估结果文件 `results/evaluation_results.json`，包含隐私保护效果、回答质量和处理时间等指标。

## 6. 分析结果

使用分析脚本分析评估结果：

```bash
python scripts/analyze_results.py --results_file results/evaluation_results.json --output_dir results
```

这将生成各种图表和分析结果，帮助你理解模型的性能。

## 7. 运行演示

使用演示脚本与P³Defer框架交互：

```bash
# 交互式模式
python examples/run_demo.py --mode interactive --local_model gemma:2b --server_model gemma:7b --model_path models/p3defer_final.pt

# 示例模式
python examples/run_demo.py --mode example --local_model gemma:2b --server_model gemma:7b --model_path models/p3defer_final.pt
```

## 8. 优化超参数

根据评估结果，你可能需要调整以下超参数来优化模型性能：

- **隐私检测阈值**：在 `data/privacy_tokens.json` 中的 `threshold` 字段
- **质量评估阈值**：在 `p3defer/p3defer.py` 中的 `encode_state` 方法
- **PPO算法参数**：在 `p3defer/models/policy_network.py` 中的 `PPOAgent` 类

## 9. 实现论文效果的关键点

1. **数据集准备**：
   - 确保训练数据包含足够的隐私样本和非隐私样本
   - 使用真实的隐私令牌（如人名）
   - 为每个问题提供高质量的答案

2. **模型配置**：
   - 本地模型应该是较小的模型（如gemma:2b）
   - 服务器模型应该是较大的模型（如gemma:7b）
   - 使用CoT提示模板增强本地模型的推理能力

3. **训练策略**：
   - 使用足够的训练轮数（至少5轮）
   - 平衡隐私保护和回答质量的奖励
   - 保存最佳模型

4. **评估方法**：
   - 使用多种指标评估模型性能
   - 分析不同决策对隐私保护和回答质量的影响
   - 比较不同配置的性能

## 10. 常见问题解答

**Q: 如何处理自定义数据集？**  
A: 将你的数据集转换为JSON格式，每个样本包含 `question` 和 `answer` 字段，然后使用 `prepare_data.py` 脚本处理。

**Q: 如何调整隐私保护和回答质量之间的平衡？**  
A: 在 `p3defer/p3defer.py` 中的 `compute_reward` 方法中调整 `privacy_weight` 参数。

**Q: 如何添加新的隐私令牌？**  
A: 编辑 `data/privacy_tokens.json` 文件，添加新的令牌到 `privacy_tokens` 列表中。

**Q: 如何使用其他模型？**  
A: 修改 `--local_model` 和 `--server_model` 参数，使用Ollama支持的其他模型。

**Q: 如何在多GPU环境中运行？**  
A: 修改 `p3defer/p3defer.py` 中的 `device` 参数，使用特定的GPU设备。

## 11. 参考资源

- [原论文](https://arxiv.org/abs/2025.xxxxx)
- [P³Defer项目文档](README.md)
- [Ollama文档](https://ollama.ai/docs)
- [PPO算法介绍](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

希望本指南能帮助你成功实现论文中的效果！如有任何问题，请随时提问。
