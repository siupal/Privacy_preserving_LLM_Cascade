"""
私有内存模块，用于存储和检测隐私令牌
"""

import os
import json
import Levenshtein
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrivateMemory:
    """私有内存类，用于存储和检测隐私令牌"""
    
    def __init__(self, privacy_tokens_file=None):
        """初始化私有内存
        
        Args:
            privacy_tokens_file: 包含隐私令牌的文件路径，如果为None则使用默认令牌
        """
        self.privacy_tokens = set()
        
        # 如果提供了文件，从文件加载隐私令牌
        if privacy_tokens_file and os.path.exists(privacy_tokens_file):
            with open(privacy_tokens_file, 'r', encoding='utf-8') as f:
                self.privacy_tokens = set(json.load(f))
        else:
            # 默认的隐私令牌列表
            self.privacy_tokens = {
                "姓名", "名字", "小明", "小红", "小张", "小李", "小王", "张三", "李四", "王五", 
                "电话", "手机", "地址", "邮箱", "身份证", "账号", "密码", "银行卡",
                "病历", "病情", "症状", "诊断", "医院", "医生", "李华", "Hector", "Todd", "Alisha", "Bobby",
                "Carl", "Maria", "John", "David", "Michael", "Sarah", "Jennifer", "Emily", "Jessica"
            }
        
        self.memory_dict = {}  # 用于存储检测到的隐私令牌
    
    def detect_privacy(self, text, threshold=0.8):
        """检测文本中的隐私令牌
        
        Args:
            text: 要检查的文本
            threshold: Levenshtein距离的阈值，用于模糊匹配
            
        Returns:
            检测到的隐私令牌列表
        """
        detected_tokens = []
        
        # 将文本分割成词
        words = text.split()
        
        for word in words:
            # 精确匹配
            if word in self.privacy_tokens:
                detected_tokens.append(word)
                self.memory_dict[len(self.memory_dict)] = word
                continue
            
            # 模糊匹配
            for token in self.privacy_tokens:
                # 计算Levenshtein距离的相似度
                if len(token) > 0 and len(word) > 0:
                    similarity = 1 - Levenshtein.distance(word, token) / max(len(word), len(token))
                    if similarity >= threshold:
                        detected_tokens.append(word)
                        self.memory_dict[len(self.memory_dict)] = word
                        break
        
        return detected_tokens
    
    def mask_privacy(self, text, mask_token="[MASK]"):
        """掩码文本中的隐私令牌
        
        Args:
            text: 要掩码的文本
            mask_token: 用于替换隐私令牌的掩码标记
            
        Returns:
            掩码后的文本
        """
        detected_tokens = self.detect_privacy(text)
        masked_text = text
        
        for token in detected_tokens:
            masked_text = masked_text.replace(token, mask_token)
        
        return masked_text
    
    def save_to_file(self, file_path):
        """将私有内存保存到文件
        
        Args:
            file_path: 保存路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.privacy_tokens), f, ensure_ascii=False, indent=2)
            logger.info(f"私有内存已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存私有内存时出错: {e}")
    
    def load_from_file(self, file_path):
        """从文件加载私有内存
        
        Args:
            file_path: 加载路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.privacy_tokens = set(json.load(f))
            logger.info(f"私有内存已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载私有内存时出错: {e}")
    
    def add_token(self, token):
        """添加隐私令牌到内存
        
        Args:
            token: 要添加的令牌
        """
        self.privacy_tokens.add(token)
        self.memory_dict[len(self.memory_dict)] = token
    
    def remove_token(self, token):
        """从内存中移除隐私令牌
        
        Args:
            token: 要移除的令牌
        """
        if token in self.privacy_tokens:
            self.privacy_tokens.remove(token)
            
            # 从memory_dict中移除
            keys_to_remove = []
            for key, value in self.memory_dict.items():
                if value == token:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_dict[key]
