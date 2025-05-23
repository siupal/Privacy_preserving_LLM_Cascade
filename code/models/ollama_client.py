"""
Ollama客户端模块，用于与本地Ollama模型交互
"""

import requests
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama API客户端，用于与本地Ollama模型交互"""
    
    def __init__(self, model_name="gemma:2b", base_url="http://localhost:11434"):
        """初始化Ollama客户端
        
        Args:
            model_name: Ollama模型名称
            base_url: Ollama API的基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
        # 测试连接
        try:
            response = requests.get(f"{base_url}/api/version")
            if response.status_code == 200:
                logger.info(f"成功连接到Ollama服务，版本: {response.json().get('version')}")
            else:
                logger.warning("无法获取Ollama版本信息")
        except Exception as e:
            logger.error(f"连接Ollama服务失败: {e}")
    
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        """生成文本
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成令牌数
            temperature: 采样温度
            
        Returns:
            生成的文本
        """
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(self.generate_url, json=data)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API错误: {response.status_code}, {response.text}")
                return ""
        except Exception as e:
            logger.error(f"调用Ollama API时出错: {e}")
            return ""
    
    def get_logits(self, prompt, temperature=0.0):
        """获取生成的logits（近似实现，Ollama API不直接提供logits）
        
        Args:
            prompt: 输入提示
            temperature: 采样温度
            
        Returns:
            生成的文本和近似的置信度
        """
        try:
            # 使用非常低的温度以获得确定性输出
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 1,  # 只预测一个token
                    "temperature": temperature
                }
            }
            
            response = requests.post(self.generate_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                
                # Ollama不直接提供logits，但提供了一些元数据
                # 我们可以使用eval_count作为置信度的近似
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 1)
                
                # 计算近似的置信度（越快生成的token可能置信度越高）
                # 这只是一个启发式方法，不是真正的logits
                confidence = 1.0 / (1.0 + (eval_duration / max(1, eval_count)))
                
                return text, confidence
            else:
                logger.error(f"Ollama API错误: {response.status_code}, {response.text}")
                return "", 0.0
        except Exception as e:
            logger.error(f"调用Ollama API时出错: {e}")
            return "", 0.0
