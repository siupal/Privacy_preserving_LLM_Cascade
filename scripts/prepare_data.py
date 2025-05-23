"""
数据准备脚本 - 用于创建训练和测试数据集
"""

import json
import random
import os
import argparse
import nltk
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='准备训练和测试数据集')
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入数据文件路径')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='输出目录')
    parser.add_argument('--privacy_tokens_file', type=str, default=None,
                        help='隐私令牌文件路径（如果有）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--privacy_ratio', type=float, default=0.5,
                        help='包含隐私信息的样本比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()

def load_privacy_tokens(file_path=None):
    """加载隐私令牌
    
    Args:
        file_path: 隐私令牌文件路径
        
    Returns:
        隐私令牌列表
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('privacy_tokens', [])
    
    # 如果没有提供文件或文件不存在，使用默认的隐私令牌
    try:
        nltk.download('names', quiet=True)
        from nltk.corpus import names
        english_names = list(names.words())
        chinese_names = ["小明", "小红", "小张", "小李", "小王", "李华", "张三", "李四", "王五", "赵六"]
        return english_names + chinese_names
    except:
        # 如果无法加载NLTK名称，使用默认的中文名称
        return ["小明", "小红", "小张", "小李", "小王", "李华", "张三", "李四", "王五", "赵六"]

def inject_privacy(question, privacy_tokens):
    """向问题中注入隐私信息
    
    Args:
        question: 原始问题
        privacy_tokens: 隐私令牌列表
        
    Returns:
        注入隐私信息后的问题和使用的隐私令牌
    """
    # 通用替换词
    generic_terms = ["someone", "a person", "a student", "a man", "a woman", 
                     "people", "students", "某人", "一个人", "一个学生"]
    
    # 随机选择1-3个隐私令牌
    num_tokens = random.randint(1, 3)
    selected_tokens = random.sample(privacy_tokens, min(num_tokens, len(privacy_tokens)))
    
    # 注入隐私信息
    modified_question = question
    used_tokens = []
    
    for token in selected_tokens:
        # 随机选择一个通用词进行替换
        generic_term = random.choice(generic_terms)
        if generic_term in modified_question:
            modified_question = modified_question.replace(generic_term, token, 1)
            used_tokens.append(token)
        else:
            # 如果没有通用词，尝试在问题开头添加隐私信息
            prefix = f"{token}的问题：" if random.random() < 0.5 else f"{token}想知道："
            modified_question = prefix + modified_question
            used_tokens.append(token)
    
    return modified_question, used_tokens

def create_answer_with_privacy_label(answer, contains_privacy):
    """创建包含隐私标签的答案
    
    Args:
        answer: 原始答案
        contains_privacy: 是否包含隐私信息
        
    Returns:
        包含隐私标签的答案
    """
    privacy_label = "是" if contains_privacy else "否"
    if "a." in answer and "b." in answer:
        # 如果答案已经有a和b的格式，替换a部分
        parts = answer.split("b.")
        return f"a. 包含个人信息：{privacy_label}\nb." + parts[1]
    else:
        # 否则添加新的格式
        return f"a. 包含个人信息：{privacy_label}\nb. 答案：{answer}"

def prepare_dataset(input_file, privacy_tokens, privacy_ratio=0.5):
    """准备数据集
    
    Args:
        input_file: 输入数据文件路径
        privacy_tokens: 隐私令牌列表
        privacy_ratio: 包含隐私信息的样本比例
        
    Returns:
        准备好的数据集
    """
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 尝试按行读取
            f.seek(0)
            data = []
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except:
                    continue
    
    # 确保数据格式正确
    processed_data = []
    for item in data:
        if isinstance(item, dict):
            question = item.get('question', item.get('input', item.get('query', '')))
            answer = item.get('answer', item.get('output', item.get('response', '')))
            
            if question and answer:
                processed_data.append({
                    'question': question,
                    'answer': answer
                })
    
    # 创建包含隐私和不包含隐私的版本
    all_data = []
    
    for item in tqdm(processed_data, desc="处理数据"):
        # 决定是否添加隐私信息
        if random.random() < privacy_ratio:
            # 创建包含隐私信息的版本
            privacy_question, used_tokens = inject_privacy(item['question'], privacy_tokens)
            privacy_answer = create_answer_with_privacy_label(item['answer'], True)
            
            all_data.append({
                'question': privacy_question,
                'answer': privacy_answer,
                'contains_privacy': True,
                'privacy_tokens': used_tokens,
                'original_question': item['question'],
                'original_answer': item['answer']
            })
        else:
            # 创建不包含隐私信息的版本
            non_privacy_answer = create_answer_with_privacy_label(item['answer'], False)
            
            all_data.append({
                'question': item['question'],
                'answer': non_privacy_answer,
                'contains_privacy': False,
                'privacy_tokens': [],
                'original_question': item['question'],
                'original_answer': item['answer']
            })
    
    return all_data

def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载隐私令牌
    privacy_tokens = load_privacy_tokens(args.privacy_tokens_file)
    print(f"加载了 {len(privacy_tokens)} 个隐私令牌")
    
    # 保存隐私令牌
    privacy_tokens_file = os.path.join(args.output_dir, 'privacy_tokens.json')
    with open(privacy_tokens_file, 'w', encoding='utf-8') as f:
        json.dump({
            'privacy_tokens': privacy_tokens,
            'threshold': 0.8,
            'last_updated': '2025-05-23'
        }, f, ensure_ascii=False, indent=2)
    print(f"隐私令牌已保存到 {privacy_tokens_file}")
    
    # 准备数据集
    all_data = prepare_dataset(args.input_file, privacy_tokens, args.privacy_ratio)
    print(f"共处理 {len(all_data)} 个样本")
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 分割训练集和测试集
    split = int(len(all_data) * args.train_ratio)
    train_data = all_data[:split]
    test_data = all_data[split:]
    
    # 保存数据集
    train_file = os.path.join(args.output_dir, 'train_data.json')
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"训练集已保存到 {train_file}，包含 {len(train_data)} 个样本")
    
    test_file = os.path.join(args.output_dir, 'test_data.json')
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存到 {test_file}，包含 {len(test_data)} 个样本")
    
    # 创建示例数据
    example_data = test_data[:5] if len(test_data) >= 5 else test_data
    example_file = os.path.join(args.output_dir, 'example_data.json')
    with open(example_file, 'w', encoding='utf-8') as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    print(f"示例数据已保存到 {example_file}，包含 {len(example_data)} 个样本")

if __name__ == "__main__":
    main()
