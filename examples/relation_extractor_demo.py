from spacy import load
import sys
sys.path.append("../")
from src.tools.relation_extractor import RelationExtractor, Entity

def print_relations(relations, text):
    print("\n文本:", text)
    print("\n提取的关系:")
    for relation in relations:
        print(f"主体: {relation.subject.text} ({relation.subject.label})")
        print(f"关系: {relation.predicate}")
        print(f"客体: {relation.object.text} ({relation.object.label})")
        print(f"置信度: {relation.confidence:.2f}")
        print(f"元数据: {relation.metadata}")
        print("-" * 50)

def main():
    # 加载中文模型
    nlp = load("zh_core_web_sm")
    extractor = RelationExtractor(nlp)

    # 测试案例1: 颜色关系
    text1 = "这件衣服是红色的，价格为99元。"
    doc1 = nlp(text1)
    entities1 = [
        Entity(text="衣服", label="PRODUCT", start=2, end=4, confidence=1.0),
        Entity(text="红色", label="COLOR", start=5, end=7, confidence=1.0),
        Entity(text="99元", label="PRICE", start=11, end=14, confidence=1.0)
    ]
    relations1 = extractor.extract_relations(doc1, entities1)
    print_relations(relations1, text1)

    # 测试案例2: 空间关系
    text2 = "书桌旁边是椅子，椅子上面放着书本。"
    doc2 = nlp(text2)
    entities2 = [
        Entity(text="书桌", label="FURNITURE", start=0, end=2, confidence=1.0),
        Entity(text="椅子", label="FURNITURE", start=4, end=6, confidence=1.0),
        Entity(text="书本", label="OBJECT", start=11, end=13, confidence=1.0)
    ]
    relations2 = extractor.extract_relations(doc2, entities2)
    print_relations(relations2, text2)

    # 测试案例3: 时间关系
    text3 = "会议将于明天下午在会议室举行。"
    doc3 = nlp(text3)
    entities3 = [
        Entity(text="会议", label="EVENT", start=0, end=2, confidence=1.0),
        Entity(text="明天下午", label="TIME", start=4, end=8, confidence=1.0),
        Entity(text="会议室", label="LOCATION", start=9, end=12, confidence=1.0)
    ]
    relations3 = extractor.extract_relations(doc3, entities3)
    print_relations(relations3, text3)

    # 测试案例4: 复杂关系组合
    text4 = "小明的红色书包放在蓝色椅子旁边，椅子位于教室。"
    doc4 = nlp(text4)
    entities4 = [
        Entity(text="红色书包", label="OBJECT", start=3, end=7, confidence=1.0),
        Entity(text="蓝色椅子", label="FURNITURE", start=9, end=13, confidence=1.0),
        Entity(text="教室", label="LOCATION", start=17, end=19, confidence=1.0)
    ]
    relations4 = extractor.extract_relations(doc4, entities4)
    print_relations(relations4, text4)

if __name__ == "__main__":
    main()
