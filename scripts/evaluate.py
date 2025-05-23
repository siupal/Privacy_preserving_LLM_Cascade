"""
P³Defer框架评估脚本
"""

import argparse
import logging
import torch
import os
import json
import sys
import numpy as np
from tqdm import tqdm
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入P³Defer框架
from p3defer import P3Defer, PrivateMemory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='P³Defer框架评估')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备: cuda 或 cpu')
    parser.add_argument('--local_model', type=str, default='gemma:2b',
                        help='本地模型名称')
    parser.add_argument('--server_model', type=str, default=None,
                        help='服务器模型名称 (可选)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据文件路径 (JSON格式)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径 (可选)')
    parser.add_argument('--privacy_tokens', type=str, default=None,
                        help='隐私令牌文件路径 (可选)')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                        help='评估结果输出文件')
    return parser.parse_args()

def load_test_data(file_path):
    """加载测试数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        测试数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式
        if not isinstance(data, list):
            logger.error(f"数据格式错误: 应为列表，实际为 {type(data)}")
            return []
        
        # 检查每个数据项是否包含必要的键
        valid_data = []
        for item in data:
            if isinstance(item, dict) and 'question' in item and 'answer' in item:
                valid_data.append(item)
            else:
                logger.warning(f"跳过无效数据项: {item}")
        
        logger.info(f"成功加载 {len(valid_data)}/{len(data)} 条测试数据")
        return valid_data
    
    except Exception as e:
        logger.error(f"加载测试数据时出错: {e}")
        return []

def create_mock_data(num_samples=10):
    """创建模拟测试数据（当无法加载真实数据时使用）
    
    Args:
        num_samples: 样本数量
        
    Returns:
        模拟测试数据列表
    """
    logger.warning("使用模拟测试数据")
    
    # 包含个人信息的问题模板
    personal_templates = [
        "小明有{a}个苹果，小红有{b}个苹果，他们一共有多少个苹果？",
        "李华每天读{a}本书，一周读几本书？",
        "张三有{a}元钱，买了一本{b}元的书，还剩多少钱？",
        "王五比李四大{a}岁，李四今年{b}岁，王五今年多少岁？"
    ]
    
    # 不包含个人信息的问题模板
    non_personal_templates = [
        "一个长方形的长是{a}厘米，宽是{b}厘米，求它的面积和周长。",
        "一个花园产出了{a}个土豆，{b}个黄瓜和黄瓜数量两倍的辣椒。这个花园总共产出了多少蔬菜？",
        "一辆汽车以每小时{a}公里的速度行驶了{b}小时，行驶了多少公里？",
        "一个数比{a}大{b}，这个数是多少？"
    ]
    
    mock_data = []
    
    for _ in range(num_samples):
        # 随机选择模板
        if np.random.random() < 0.5:
            template = np.random.choice(personal_templates)
            contains_personal = True
        else:
            template = np.random.choice(non_personal_templates)
            contains_personal = False
        
        # 生成随机数
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        
        # 生成问题
        question = template.format(a=a, b=b)
        
        # 生成答案
        if "一共有多少个苹果" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}+{b}={a+b}个苹果"
        elif "一周读几本书" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}*7={a*7}本书"
        elif "还剩多少钱" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{a}-{b}={a-b}元"
        elif "王五今年多少岁" in question:
            answer = f"a. 包含个人信息：是\nb. 答案：{b}+{a}={b+a}岁"
        elif "面积和周长" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：面积={a}*{b}={a*b}平方厘米，周长=2*({a}+{b})={2*(a+b)}厘米"
        elif "总共产出了多少蔬菜" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}+{b}+{b*2}={a+b+b*2}个蔬菜"
        elif "行驶了多少公里" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}*{b}={a*b}公里"
        elif "这个数是多少" in question:
            answer = f"a. 包含个人信息：否\nb. 答案：{a}+{b}={a+b}"
        else:
            answer = f"a. 包含个人信息：{'是' if contains_personal else '否'}\nb. 答案：计算结果"
        
        mock_data.append({
            "question": question,
            "answer": answer,
            "contains_privacy": contains_personal
        })
    
    return mock_data

def evaluate_privacy_leakage(output, privacy_tokens):
    """评估隐私泄露程度
    
    Args:
        output: 模型输出
        privacy_tokens: 隐私令牌列表
        
    Returns:
        隐私泄露分数 (0-1，0表示无泄露，1表示完全泄露)
    """
    if not privacy_tokens:
        return 0.0
    
    leaked_tokens = 0
    for token in privacy_tokens:
        if token in output:
            leaked_tokens += 1
    
    return leaked_tokens / len(privacy_tokens) if privacy_tokens else 0.0

def main():
    args = parse_args()
    
    # 加载测试数据
    test_data = load_test_data(args.test_data)
    
    # 如果无法加载测试数据，创建模拟数据
    if not test_data:
        test_data = create_mock_data(num_samples=20)
    
    # 创建私有内存实例
    privacy_memory = PrivateMemory(args.privacy_tokens)
    
    # 创建P³Defer实例
    p3defer = P3Defer(
        local_model_name=args.local_model,
        server_model_name=args.server_model,
        privacy_memory=privacy_memory,
        device=args.device
    )
    
    # 如果提供了模型路径，加载预训练模型
    if args.model_path:
        logger.info(f"加载预训练模型: {args.model_path}")
        p3defer.load_model(args.model_path)
    
    # 评估结果
    results = {
        "total_samples": len(test_data),
        "privacy_samples": sum(1 for item in test_data if item.get("contains_privacy", False)),
        "non_privacy_samples": sum(1 for item in test_data if not item.get("contains_privacy", False)),
        "total_time": 0,
        "avg_time_per_query": 0,
        "rouge_scores": {},
        "privacy_leakage": 0,
        "samples": []
    }
    
    # 开始评估
    logger.info(f"开始评估，测试集大小: {len(test_data)}")
    start_time = time.time()
    
    for i, item in enumerate(tqdm(test_data, desc="评估进度")):
        query = item["question"]
        golden_output = item["answer"]
        contains_privacy = item.get("contains_privacy", False)
        
        # 检测隐私令牌
        privacy_tokens = privacy_memory.detect_privacy(query)
        
        # 处理查询
        query_start_time = time.time()
        output = p3defer.process_query(query)
        query_time = time.time() - query_start_time
        
        # 计算ROUGE分数
        rouge_scores = p3defer.rouge_compute([output], [golden_output])
        
        # 评估隐私泄露
        privacy_leakage = evaluate_privacy_leakage(output, privacy_tokens)
        
        # 记录样本结果
        sample_result = {
            "id": i,
            "question": query,
            "golden_answer": golden_output,
            "model_output": output,
            "contains_privacy": contains_privacy,
            "privacy_tokens": privacy_tokens,
            "privacy_leakage": privacy_leakage,
            "rouge_scores": {k: v for k, v in rouge_scores.items()},
            "processing_time": query_time
        }
        
        results["samples"].append(sample_result)
    
    # 计算总体结果
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["avg_time_per_query"] = total_time / len(test_data)
    
    # 计算平均ROUGE分数
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    for rouge_type in rouge_types:
        results["rouge_scores"][rouge_type] = sum(sample["rouge_scores"].get(rouge_type, 0) for sample in results["samples"]) / len(test_data)
    
    # 计算平均隐私泄露
    privacy_samples = [sample for sample in results["samples"] if sample["contains_privacy"]]
    if privacy_samples:
        results["privacy_leakage"] = sum(sample["privacy_leakage"] for sample in privacy_samples) / len(privacy_samples)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，结果已保存到 {args.output_file}")
    logger.info(f"总样本数: {results['total_samples']}")
    logger.info(f"平均ROUGE-L分数: {results['rouge_scores']['rougeL']:.4f}")
    logger.info(f"平均隐私泄露分数: {results['privacy_leakage']:.4f}")
    logger.info(f"平均处理时间: {results['avg_time_per_query']:.4f}秒")

if __name__ == "__main__":
    main()
