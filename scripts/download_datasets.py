"""
数据集下载脚本 - 自动下载和处理常用数据集
"""

import os
import json
import argparse
import requests
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='下载和处理数据集')
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'mmlu', 'math', 'all'],
                        help='要下载的数据集名称')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='输出目录')
    return parser.parse_args()

def download_file(url, output_path):
    """下载文件
    
    Args:
        url: 文件URL
        output_path: 输出路径
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(output_path, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), total=total_size//block_size, desc="下载中"):
            f.write(data)

def download_gsm8k(output_dir):
    """下载GSM8K数据集
    
    Args:
        output_dir: 输出目录
    """
    print("下载GSM8K数据集...")
    
    # 创建输出目录
    gsm8k_dir = os.path.join(output_dir, 'gsm8k')
    os.makedirs(gsm8k_dir, exist_ok=True)
    
    # GSM8K数据集URL
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    
    # 下载训练集
    train_path = os.path.join(gsm8k_dir, 'train.jsonl')
    if not os.path.exists(train_path):
        download_file(train_url, train_path)
    
    # 下载测试集
    test_path = os.path.join(gsm8k_dir, 'test.jsonl')
    if not os.path.exists(test_path):
        download_file(test_url, test_path)
    
    # 处理数据集
    print("处理GSM8K数据集...")
    
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            train_data.append({
                'question': item['question'],
                'answer': item['answer']
            })
    
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            test_data.append({
                'question': item['question'],
                'answer': item['answer']
            })
    
    # 保存处理后的数据集
    processed_train_path = os.path.join(gsm8k_dir, 'train.json')
    with open(processed_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    processed_test_path = os.path.join(gsm8k_dir, 'test.json')
    with open(processed_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"GSM8K数据集已下载和处理完成，共 {len(train_data)} 个训练样本和 {len(test_data)} 个测试样本")
    
    return processed_train_path, processed_test_path

def download_mmlu(output_dir):
    """下载MMLU数据集
    
    Args:
        output_dir: 输出目录
    """
    print("下载MMLU数据集...")
    
    # 创建输出目录
    mmlu_dir = os.path.join(output_dir, 'mmlu')
    os.makedirs(mmlu_dir, exist_ok=True)
    
    # MMLU数据集URL
    mmlu_url = "https://github.com/hendrycks/test/archive/refs/heads/master.zip"
    
    # 下载MMLU数据集
    zip_path = os.path.join(mmlu_dir, 'mmlu.zip')
    if not os.path.exists(zip_path):
        download_file(mmlu_url, zip_path)
    
    # 解压数据集
    extract_dir = os.path.join(mmlu_dir, 'extract')
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 处理数据集
    print("处理MMLU数据集...")
    
    # 选择一些代表性的科目
    subjects = ['high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 
                'high_school_biology', 'high_school_computer_science']
    
    train_data = []
    test_data = []
    
    for subject in subjects:
        subject_dir = os.path.join(extract_dir, 'test-master', 'data', subject)
        
        # 读取训练集
        train_path = os.path.join(subject_dir, 'train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            for _, row in df.iterrows():
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer_idx = ord(row[5]) - ord('A')
                answer = options[answer_idx]
                
                train_data.append({
                    'question': f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}",
                    'answer': f"答案是 {row[5]}. {answer}"
                })
        
        # 读取测试集
        test_path = os.path.join(subject_dir, 'test.csv')
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            for _, row in df.iterrows():
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer_idx = ord(row[5]) - ord('A')
                answer = options[answer_idx]
                
                test_data.append({
                    'question': f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}",
                    'answer': f"答案是 {row[5]}. {answer}"
                })
    
    # 保存处理后的数据集
    processed_train_path = os.path.join(mmlu_dir, 'train.json')
    with open(processed_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    processed_test_path = os.path.join(mmlu_dir, 'test.json')
    with open(processed_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"MMLU数据集已下载和处理完成，共 {len(train_data)} 个训练样本和 {len(test_data)} 个测试样本")
    
    return processed_train_path, processed_test_path

def download_math(output_dir):
    """下载MATH数据集
    
    Args:
        output_dir: 输出目录
    """
    print("下载MATH数据集...")
    
    # 创建输出目录
    math_dir = os.path.join(output_dir, 'math')
    os.makedirs(math_dir, exist_ok=True)
    
    # MATH数据集URL
    math_url = "https://github.com/hendrycks/math/archive/refs/heads/main.zip"
    
    # 下载MATH数据集
    zip_path = os.path.join(math_dir, 'math.zip')
    if not os.path.exists(zip_path):
        download_file(math_url, zip_path)
    
    # 解压数据集
    extract_dir = os.path.join(math_dir, 'extract')
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # 处理数据集
    print("处理MATH数据集...")
    
    train_data = []
    test_data = []
    
    # 读取训练集
    train_path = os.path.join(extract_dir, 'math-main', 'train.json')
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                train_data.append({
                    'question': item['problem'],
                    'answer': item['solution']
                })
    
    # 读取测试集
    test_path = os.path.join(extract_dir, 'math-main', 'test.json')
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                test_data.append({
                    'question': item['problem'],
                    'answer': item['solution']
                })
    
    # 保存处理后的数据集
    processed_train_path = os.path.join(math_dir, 'train.json')
    with open(processed_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    processed_test_path = os.path.join(math_dir, 'test.json')
    with open(processed_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"MATH数据集已下载和处理完成，共 {len(train_data)} 个训练样本和 {len(test_data)} 个测试样本")
    
    return processed_train_path, processed_test_path

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'gsm8k' or args.dataset == 'all':
        train_path, test_path = download_gsm8k(args.output_dir)
        print(f"GSM8K训练集: {train_path}")
        print(f"GSM8K测试集: {test_path}")
    
    if args.dataset == 'mmlu' or args.dataset == 'all':
        train_path, test_path = download_mmlu(args.output_dir)
        print(f"MMLU训练集: {train_path}")
        print(f"MMLU测试集: {test_path}")
    
    if args.dataset == 'math' or args.dataset == 'all':
        train_path, test_path = download_math(args.output_dir)
        print(f"MATH训练集: {train_path}")
        print(f"MATH测试集: {test_path}")
    
    print("\n下载完成！现在你可以使用以下命令准备数据集：")
    print(f"python scripts/prepare_data.py --input_file [下载的数据集路径] --output_dir data")

if __name__ == "__main__":
    main()
