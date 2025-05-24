import logging
import re
import uuid
from typing import List, Dict, Any, Optional
import os
import json

from llama_index.core.llms import OpenAI
from app.config.settings import OPENAI_API_KEY, OPENAI_MODEL


class EntityExtractor:
    """实体提取器，负责从文本中提取实体"""
    
    def __init__(self):
        """初始化实体提取器"""
        # 设置OpenAI API密钥
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        # 初始化LLM
        self.llm = OpenAI(model=OPENAI_MODEL, temperature=0)
    
    def extract_entities_simple(self, text: str) -> List[Dict[str, Any]]:
        """使用简单的规则提取实体（备用方法）"""
        entities = []
        # 简单的正则匹配，实际应用中应当更为复杂
        keywords = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        
        for keyword in set(keywords):
            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            entities.append({
                "id": entity_id,
                "name": keyword,
                "type": "keyword"
            })
        
        return entities
    
    def extract_entities_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用LLM提取实体和关系"""
        try:
            # 限制文本长度，避免超出LLM的最大输入长度
            max_text_length = 4000  # 假设最大输入长度为4000字符
            if len(text) > max_text_length:
                text = text[:max_text_length]
            
            # 构建提示 - 修改为不要求LLM生成id字段
            prompt = f"""
            从以下文本中提取关键实体。结果应当是一个JSON格式的数组，每个实体包含name和type两个字段。
            
            例如：
            [
                {{"name": "人工智能", "type": "technology"}},
                {{"name": "深度学习", "type": "technology"}},
                {{"name": "OpenAI", "type": "organization"}}
            ]
            
            实体类型可以包括：person（人物）、organization（组织）、location（地点）、technology（技术）、
            product（产品）、event（事件）、time（时间）、concept（概念）等。
            
            要提取实体的文本：
            {text}
            
            JSON结果：
            """
            
            # 调用LLM提取实体
            response = self.llm.complete(prompt)
            
            # 解析结果
            entities_json = response.text.strip()
            
            # 尝试去除可能的代码块标记
            if entities_json.startswith("```") and entities_json.endswith("```"):
                entities_json = entities_json[3:-3].strip()
            if entities_json.startswith("```json") and entities_json.endswith("```"):
                entities_json = entities_json[7:-3].strip()
            
            # 解析JSON
            entities = json.loads(entities_json)
            
            # 系统添加唯一ID（不依赖LLM）
            for entity in entities:
                entity["id"] = f"entity_{uuid.uuid4().hex[:8]}"
            
            return entities
            
        except Exception as e:
            logging.error(f"使用LLM提取实体时出错: {str(e)}")
            # 如果LLM提取失败，回退到简单的规则提取
            return self.extract_entities_simple(text)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体，默认使用LLM方法，失败时回退到简单方法"""
        try:
            return self.extract_entities_llm(text)
        except Exception as e:
            logging.error(f"提取实体时出错: {str(e)}")
            return self.extract_entities_simple(text)
    
    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取三元组（主体-关系-客体）"""
        try:
            # 限制文本长度
            max_text_length = 4000
            if len(text) > max_text_length:
                text = text[:max_text_length]
            
            # 构建提示
            prompt = f"""
            从以下文本中提取知识三元组。结果应当是一个JSON格式的数组，每个三元组包含subject、relation、object三个字段。
            
            例如：
            [
                {{"subject": "人工智能", "relation": "包括", "object": "深度学习"}},
                {{"subject": "OpenAI", "relation": "开发", "object": "GPT-4"}},
                {{"subject": "深度学习", "relation": "应用于", "object": "计算机视觉"}}
            ]
            
            要提取三元组的文本：
            {text}
            
            JSON结果：
            """
            
            # 调用LLM提取三元组
            response = self.llm.complete(prompt)
            
            # 解析结果
            triples_json = response.text.strip()
            
            # 尝试去除可能的代码块标记
            if triples_json.startswith("```") and triples_json.endswith("```"):
                triples_json = triples_json[3:-3].strip()
            if triples_json.startswith("```json") and triples_json.endswith("```"):
                triples_json = triples_json[7:-3].strip()
            
            # 解析JSON
            triples = json.loads(triples_json)
            
            return triples
            
        except Exception as e:
            logging.error(f"提取三元组时出错: {str(e)}")
            return [] 