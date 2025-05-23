"""
结果分析脚本 - 用于分析训练和评估结果
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description='分析训练和评估结果')
    parser.add_argument('--results_file', type=str, required=True,
                        help='评估结果文件路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    return parser.parse_args()

def analyze_privacy_protection(results):
    """分析隐私保护效果
    
    Args:
        results: 评估结果
        
    Returns:
        隐私保护分析结果
    """
    privacy_leakage = results.get('privacy_leakage', 0)
    privacy_samples = results.get('privacy_samples', 0)
    
    # 按决策类型分析隐私泄露
    decision_leakage = {0: [], 1: [], 2: []}  # 0: 本地模型, 1: 直接转发, 2: 掩码后转发
    
    for sample in results.get('samples', []):
        if sample.get('contains_privacy', False):
            decision = sample.get('decision', 0)
            leakage = sample.get('privacy_leakage', 0)
            decision_leakage[decision].append(leakage)
    
    # 计算每种决策的平均隐私泄露
    avg_leakage_by_decision = {}
    for decision, leakages in decision_leakage.items():
        if leakages:
            avg_leakage_by_decision[decision] = sum(leakages) / len(leakages)
        else:
            avg_leakage_by_decision[decision] = 0
    
    return {
        'overall_privacy_leakage': privacy_leakage,
        'privacy_samples': privacy_samples,
        'avg_leakage_by_decision': avg_leakage_by_decision
    }

def analyze_answer_quality(results):
    """分析回答质量
    
    Args:
        results: 评估结果
        
    Returns:
        回答质量分析结果
    """
    rouge_scores = results.get('rouge_scores', {})
    
    # 按决策类型分析回答质量
    decision_quality = {0: [], 1: [], 2: []}  # 0: 本地模型, 1: 直接转发, 2: 掩码后转发
    
    for sample in results.get('samples', []):
        decision = sample.get('decision', 0)
        rouge_l = sample.get('rouge_scores', {}).get('rougeL', 0)
        decision_quality[decision].append(rouge_l)
    
    # 计算每种决策的平均回答质量
    avg_quality_by_decision = {}
    for decision, qualities in decision_quality.items():
        if qualities:
            avg_quality_by_decision[decision] = sum(qualities) / len(qualities)
        else:
            avg_quality_by_decision[decision] = 0
    
    return {
        'overall_rouge_scores': rouge_scores,
        'avg_quality_by_decision': avg_quality_by_decision
    }

def analyze_decision_distribution(results):
    """分析决策分布
    
    Args:
        results: 评估结果
        
    Returns:
        决策分布分析结果
    """
    decisions = [sample.get('decision', 0) for sample in results.get('samples', [])]
    decision_counts = Counter(decisions)
    
    # 计算每种决策的比例
    total_samples = len(decisions)
    decision_ratios = {decision: count / total_samples for decision, count in decision_counts.items()}
    
    # 按是否包含隐私分析决策分布
    privacy_decisions = []
    non_privacy_decisions = []
    
    for sample in results.get('samples', []):
        decision = sample.get('decision', 0)
        if sample.get('contains_privacy', False):
            privacy_decisions.append(decision)
        else:
            non_privacy_decisions.append(decision)
    
    privacy_decision_counts = Counter(privacy_decisions)
    non_privacy_decision_counts = Counter(non_privacy_decisions)
    
    # 计算包含隐私和不包含隐私样本的决策比例
    privacy_total = len(privacy_decisions)
    non_privacy_total = len(non_privacy_decisions)
    
    privacy_decision_ratios = {decision: count / privacy_total for decision, count in privacy_decision_counts.items()} if privacy_total > 0 else {}
    non_privacy_decision_ratios = {decision: count / non_privacy_total for decision, count in non_privacy_decision_counts.items()} if non_privacy_total > 0 else {}
    
    return {
        'overall_decision_counts': dict(decision_counts),
        'overall_decision_ratios': decision_ratios,
        'privacy_decision_counts': dict(privacy_decision_counts),
        'privacy_decision_ratios': privacy_decision_ratios,
        'non_privacy_decision_counts': dict(non_privacy_decision_counts),
        'non_privacy_decision_ratios': non_privacy_decision_ratios
    }

def analyze_efficiency(results):
    """分析效率
    
    Args:
        results: 评估结果
        
    Returns:
        效率分析结果
    """
    total_time = results.get('total_time', 0)
    avg_time_per_query = results.get('avg_time_per_query', 0)
    
    # 按决策类型分析处理时间
    decision_times = {0: [], 1: [], 2: []}  # 0: 本地模型, 1: 直接转发, 2: 掩码后转发
    
    for sample in results.get('samples', []):
        decision = sample.get('decision', 0)
        processing_time = sample.get('processing_time', 0)
        decision_times[decision].append(processing_time)
    
    # 计算每种决策的平均处理时间
    avg_time_by_decision = {}
    for decision, times in decision_times.items():
        if times:
            avg_time_by_decision[decision] = sum(times) / len(times)
        else:
            avg_time_by_decision[decision] = 0
    
    return {
        'total_time': total_time,
        'avg_time_per_query': avg_time_per_query,
        'avg_time_by_decision': avg_time_by_decision
    }

def plot_privacy_protection(privacy_analysis, output_dir):
    """绘制隐私保护效果图
    
    Args:
        privacy_analysis: 隐私保护分析结果
        output_dir: 输出目录
    """
    # 绘制每种决策的平均隐私泄露
    decision_labels = {0: "本地模型", 1: "直接转发", 2: "掩码后转发"}
    decisions = sorted(privacy_analysis['avg_leakage_by_decision'].keys())
    leakages = [privacy_analysis['avg_leakage_by_decision'][d] for d in decisions]
    labels = [decision_labels.get(d, f"决策{d}") for d in decisions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, leakages, color='red')
    plt.title("各决策的平均隐私泄露率")
    plt.ylabel("隐私泄露率")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "privacy_leakage_by_decision.png"))
    plt.close()

def plot_answer_quality(quality_analysis, output_dir):
    """绘制回答质量图
    
    Args:
        quality_analysis: 回答质量分析结果
        output_dir: 输出目录
    """
    # 绘制每种决策的平均回答质量
    decision_labels = {0: "本地模型", 1: "直接转发", 2: "掩码后转发"}
    decisions = sorted(quality_analysis['avg_quality_by_decision'].keys())
    qualities = [quality_analysis['avg_quality_by_decision'][d] for d in decisions]
    labels = [decision_labels.get(d, f"决策{d}") for d in decisions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, qualities, color='blue')
    plt.title("各决策的平均回答质量 (ROUGE-L)")
    plt.ylabel("ROUGE-L分数")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "quality_by_decision.png"))
    plt.close()
    
    # 绘制整体ROUGE分数
    rouge_types = sorted(quality_analysis['overall_rouge_scores'].keys())
    rouge_scores = [quality_analysis['overall_rouge_scores'][t] for t in rouge_types]
    
    plt.figure(figsize=(10, 6))
    plt.bar(rouge_types, rouge_scores, color='green')
    plt.title("整体ROUGE分数")
    plt.ylabel("分数")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "overall_rouge_scores.png"))
    plt.close()

def plot_decision_distribution(decision_analysis, output_dir):
    """绘制决策分布图
    
    Args:
        decision_analysis: 决策分布分析结果
        output_dir: 输出目录
    """
    decision_labels = {0: "本地模型", 1: "直接转发", 2: "掩码后转发"}
    
    # 绘制整体决策分布
    decisions = sorted(decision_analysis['overall_decision_counts'].keys())
    counts = [decision_analysis['overall_decision_counts'][d] for d in decisions]
    labels = [decision_labels.get(d, f"决策{d}") for d in decisions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.title("整体决策分布")
    plt.ylabel("样本数量")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "overall_decision_distribution.png"))
    plt.close()
    
    # 绘制包含隐私和不包含隐私样本的决策分布
    privacy_decisions = sorted(decision_analysis['privacy_decision_counts'].keys())
    privacy_counts = [decision_analysis['privacy_decision_counts'][d] for d in privacy_decisions]
    privacy_labels = [decision_labels.get(d, f"决策{d}") for d in privacy_decisions]
    
    non_privacy_decisions = sorted(decision_analysis['non_privacy_decision_counts'].keys())
    non_privacy_counts = [decision_analysis['non_privacy_decision_counts'][d] for d in non_privacy_decisions]
    non_privacy_labels = [decision_labels.get(d, f"决策{d}") for d in non_privacy_decisions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(privacy_labels, privacy_counts, color='red')
    ax1.set_title("包含隐私样本的决策分布")
    ax1.set_ylabel("样本数量")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(non_privacy_labels, non_privacy_counts, color='green')
    ax2.set_title("不包含隐私样本的决策分布")
    ax2.set_ylabel("样本数量")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_vs_non_privacy_decision_distribution.png"))
    plt.close()
    
    # 绘制决策比例
    privacy_ratios = [decision_analysis['privacy_decision_ratios'].get(d, 0) for d in range(3)]
    non_privacy_ratios = [decision_analysis['non_privacy_decision_ratios'].get(d, 0) for d in range(3)]
    
    x = np.arange(len(decision_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, privacy_ratios, width, label='包含隐私')
    rects2 = ax.bar(x + width/2, non_privacy_ratios, width, label='不包含隐私')
    
    ax.set_ylabel('比例')
    ax.set_title('包含隐私vs不包含隐私样本的决策比例')
    ax.set_xticks(x)
    ax.set_xticklabels([decision_labels.get(i, f"决策{i}") for i in range(3)])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_ratios.png"))
    plt.close()

def plot_efficiency(efficiency_analysis, output_dir):
    """绘制效率图
    
    Args:
        efficiency_analysis: 效率分析结果
        output_dir: 输出目录
    """
    # 绘制每种决策的平均处理时间
    decision_labels = {0: "本地模型", 1: "直接转发", 2: "掩码后转发"}
    decisions = sorted(efficiency_analysis['avg_time_by_decision'].keys())
    times = [efficiency_analysis['avg_time_by_decision'][d] for d in decisions]
    labels = [decision_labels.get(d, f"决策{d}") for d in decisions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color='purple')
    plt.title("各决策的平均处理时间")
    plt.ylabel("时间 (秒)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "processing_time_by_decision.png"))
    plt.close()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载评估结果
    with open(args.results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 分析结果
    privacy_analysis = analyze_privacy_protection(results)
    quality_analysis = analyze_answer_quality(results)
    decision_analysis = analyze_decision_distribution(results)
    efficiency_analysis = analyze_efficiency(results)
    
    # 绘制图表
    plot_privacy_protection(privacy_analysis, args.output_dir)
    plot_answer_quality(quality_analysis, args.output_dir)
    plot_decision_distribution(decision_analysis, args.output_dir)
    plot_efficiency(efficiency_analysis, args.output_dir)
    
    # 保存分析结果
    analysis_results = {
        'privacy_analysis': privacy_analysis,
        'quality_analysis': quality_analysis,
        'decision_analysis': decision_analysis,
        'efficiency_analysis': efficiency_analysis
    }
    
    with open(os.path.join(args.output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n===== 结果分析摘要 =====")
    print(f"总样本数: {results.get('total_samples', 0)}")
    print(f"包含隐私样本数: {results.get('privacy_samples', 0)}")
    print(f"不包含隐私样本数: {results.get('non_privacy_samples', 0)}")
    print(f"整体隐私泄露率: {privacy_analysis['overall_privacy_leakage']:.4f}")
    print(f"整体ROUGE-L分数: {quality_analysis['overall_rouge_scores'].get('rougeL', 0):.4f}")
    print(f"平均处理时间: {efficiency_analysis['avg_time_per_query']:.4f}秒")
    print("\n决策分布:")
    for decision, count in decision_analysis['overall_decision_counts'].items():
        decision_label = {0: "本地模型", 1: "直接转发", 2: "掩码后转发"}.get(decision, f"决策{decision}")
        ratio = decision_analysis['overall_decision_ratios'][decision]
        print(f"  {decision_label}: {count} ({ratio:.2%})")
    
    print("\n分析结果和图表已保存到目录:", args.output_dir)

if __name__ == "__main__":
    main()
