import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.relation_extractor import RelationExtractor, Entity
from spacy import load

def analyze_phrase():
    # 初始化
    nlp = load("zh_core_web_sm")
    extractor = RelationExtractor(nlp)
    
    # 分析文本
    text = "我们冤家路窄 狭路相逢"
    
    # 定义实体
    entities = [
        Entity(
            text="我们",
            label="PERSON",
            start=0,
            end=2,
            confidence=1.0,
            attributes={"role": "subject", "relationship": "antagonist"}
        ),
        Entity(
            text="冤家",
            label="PERSON",
            start=2,
            end=4,
            confidence=1.0,
            attributes={"type": "relationship_state", "sentiment": "negative"}
        ),
        Entity(
            text="路窄",
            label="STATE",
            start=4,
            end=6,
            confidence=1.0,
            attributes={"type": "metaphor", "implies": "inevitable_confrontation"}
        ),
        Entity(
            text="狭路",
            label="LOCATION",
            start=7,
            end=9,
            confidence=1.0,
            attributes={"type": "metaphor", "symbolizes": "limited_space"}
        ),
        Entity(
            text="相逢",
            label="EVENT",
            start=9,
            end=11,
            confidence=1.0,
            attributes={"type": "encounter", "nature": "confrontational"}
        )
    ]
    
    # 处理文本
    doc = nlp(text)
    
    # 提取关系
    relations = extractor.extract_relations(doc, entities)
    
    # 输出结果
    print("\n原文：", text)
    print("\n实体分析：")
    for entity in entities:
        print(f"- {entity.text} ({entity.label})")
        if entity.attributes:
            print(f"  属性：{entity.attributes}")
    
    print("\n关系分析：")
    if relations:
        for relation in relations:
            print(f"- {relation.subject.text} --[{relation.predicate}]--> {relation.object.text} (置信度: {relation.confidence:.2f})")
            if relation.metadata:
                print(f"  元数据: {relation.metadata}")
    
    print("\n语义解析：")
    print("1. 人物关系：")
    print("   - '我们'与'冤家'形成对立关系")
    print("   - '冤家'暗示敌对或对立的关系状态")
    
    print("\n2. 空间隐喻：")
    print("   - '路窄'和'狭路'都暗示空间的局限性")
    print("   - 物理空间的局限暗示形势的紧迫")
    
    print("\n3. 事件性质：")
    print("   - '相逢'在此语境下暗示不可避免的对抗")
    print("   - 整体表达了对立双方必然相遇对抗的宿命感")
    
    print("\n4. 深层含义：")
    print("   - 表达了对立双方在特定环境下不可避免的遭遇")
    print("   - 暗示了事态发展的必然性和戏剧性")
    print("   - 蕴含着命运的安排和情势的必然")

if __name__ == "__main__":
    analyze_phrase()
