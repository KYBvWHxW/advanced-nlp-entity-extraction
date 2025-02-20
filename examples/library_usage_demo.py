from spacy import load
import sys
sys.path.append("../")
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.relation_extractor import RelationExtractor, Entity
from typing import List, Dict
import json

class RelationExtractorDemo:
    def __init__(self):
        """初始化关系抽取器"""
        self.nlp = load("zh_core_web_sm")
        self.extractor = RelationExtractor(self.nlp)
    
    def analyze_text(self, text: str, entities: List[Dict]) -> Dict:
        """分析文本中的实体关系
        
        Args:
            text: 要分析的文本
            entities: 实体列表，每个实体是一个字典，包含text, label, start, end等信息
            
        Returns:
            包含分析结果的字典
        """
        # 处理文本
        doc = self.nlp(text)
        
        # 转换实体格式
        entity_objects = [
            Entity(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                confidence=e.get("confidence", 1.0)
            )
            for e in entities
        ]
        
        # 提取关系
        relations = self.extractor.extract_relations(doc, entity_objects)
        
        # 格式化结果
        results = []
        for relation in relations:
            results.append({
                "subject": {
                    "text": relation.subject.text,
                    "label": relation.subject.label,
                    "position": (relation.subject.start, relation.subject.end)
                },
                "predicate": relation.predicate,
                "object": {
                    "text": relation.object.text,
                    "label": relation.object.label,
                    "position": (relation.object.start, relation.object.end)
                },
                "confidence": relation.confidence,
                "metadata": relation.metadata
            })
        
        return {
            "text": text,
            "entities": entities,
            "relations": results
        }
    
    def batch_analyze(self, texts: List[Dict[str, List[Dict]]]) -> List[Dict]:
        """批量分析多个文本
        
        Args:
            texts: 文本列表，每个元素是包含text和entities的字典
            
        Returns:
            分析结果列表
        """
        results = []
        for item in texts:
            result = self.analyze_text(item["text"], item["entities"])
            results.append(result)
        return results

def main():
    # 创建演示实例
    demo = RelationExtractorDemo()
    
    # 示例1：单个文本分析
    text1 = "这件红色衣服在柜子里，标价99元。"
    entities1 = [
        {
            "text": "红色衣服",
            "label": "PRODUCT",
            "start": 2,
            "end": 6,
            "confidence": 1.0
        },
        {
            "text": "柜子",
            "label": "FURNITURE",
            "start": 7,
            "end": 9,
            "confidence": 1.0
        },
        {
            "text": "99元",
            "label": "PRICE",
            "start": 12,
            "end": 15,
            "confidence": 1.0
        }
    ]
    
    result1 = demo.analyze_text(text1, entities1)
    print("\n单个文本分析结果:")
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    # 示例2：批量分析
    texts = [
        {
            "text": "会议将在明天下午3点在会议室举行",
            "entities": [
                {
                    "text": "会议",
                    "label": "EVENT",
                    "start": 0,
                    "end": 2,
                    "confidence": 1.0
                },
                {
                    "text": "明天下午3点",
                    "label": "TIME",
                    "start": 4,
                    "end": 10,
                    "confidence": 1.0
                },
                {
                    "text": "会议室",
                    "label": "LOCATION",
                    "start": 11,
                    "end": 14,
                    "confidence": 1.0
                }
            ]
        },
        {
            "text": "小明把蓝色书包放在了桌子上",
            "entities": [
                {
                    "text": "蓝色书包",
                    "label": "OBJECT",
                    "start": 3,
                    "end": 7,
                    "confidence": 1.0
                },
                {
                    "text": "桌子",
                    "label": "FURNITURE",
                    "start": 9,
                    "end": 11,
                    "confidence": 1.0
                }
            ]
        }
    ]
    
    results = demo.batch_analyze(texts)
    print("\n批量分析结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
