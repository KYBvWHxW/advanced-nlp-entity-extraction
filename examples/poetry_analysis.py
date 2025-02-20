import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.relation_extractor import RelationExtractor, Entity
from spacy import load

def analyze_poetry():
    # 初始化
    nlp = load("zh_core_web_sm")
    extractor = RelationExtractor(nlp)
    
    # 分析文本
    text = "树欲静而风不止"
    
    # 定义实体
    entities = [
        Entity(
            text="树",
            label="OBJECT",
            start=0,
            end=1,
            confidence=1.0,
            attributes={"role": "subject", "desire": "静"}
        ),
        Entity(
            text="静",
            label="STATE",
            start=2,
            end=3,
            confidence=1.0,
            attributes={"type": "desired_state"}
        ),
        Entity(
            text="风",
            label="OBJECT",
            start=4,
            end=5,
            confidence=1.0,
            attributes={"role": "subject", "action": "不止"}
        ),
        Entity(
            text="止",
            label="STATE",
            start=6,
            end=7,
            confidence=1.0,
            attributes={"type": "actual_state", "negation": True}
        )
    ]
    
    # 处理文本
    doc = nlp(text)
    
    # 提取关系
    relations = extractor.extract_relations(doc, entities)
    
    # 输出结果
    print("\n原文：", text)
    print("\n实体识别：")
    for entity in entities:
        print(f"- {entity.text} ({entity.label})")
    
    print("\n关系抽取：")
    for relation in relations:
        print(f"- {relation.subject.text} --[{relation.predicate}]--> {relation.object.text} (置信度: {relation.confidence:.2f})")
        if relation.metadata:
            print(f"  元数据: {relation.metadata}")

if __name__ == "__main__":
    analyze_poetry()
