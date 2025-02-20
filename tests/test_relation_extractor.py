"""
关系提取器的单元测试。
"""

import unittest
import spacy
from src.tools.relation_extractor import RelationExtractor, Entity, Relation

class TestRelationExtractor(unittest.TestCase):
    """关系提取器测试用例。"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境。"""
        cls.nlp = spacy.load("zh_core_web_lg")
        cls.extractor = RelationExtractor(cls.nlp)
    
    def test_chinese_relation_extraction(self):
        """测试中文关系提取。"""
        # 测试产品属性关系
        text = "这件纯棉T恤是深蓝色的，尺码为XL，售价299元。"
        doc = self.nlp(text)
        entities = [
            Entity(text="纯棉T恤", label="PRODUCT", start=2, end=6, confidence=0.9),
            Entity(text="深蓝色", label="COLOR", start=7, end=10, confidence=0.9),
            Entity(text="XL", label="SIZE", start=13, end=15, confidence=0.9),
            Entity(text="299元", label="PRICE", start=17, end=21, confidence=0.9)
        ]
        
        relations = self.extractor.extract_relations(doc, entities)
        
        # 验证关系提取结果
        self.assertTrue(any(
            r.subject.text == "纯棉T恤" and
            r.predicate == "has_color" and
            r.object.text == "深蓝色"
            for r in relations
        ))
        
        self.assertTrue(any(
            r.subject.text == "纯棉T恤" and
            r.predicate == "has_size" and
            r.object.text == "XL"
            for r in relations
        ))
        
        self.assertTrue(any(
            r.subject.text == "纯棉T恤" and
            r.predicate == "has_price" and
            r.object.text == "299元"
            for r in relations
        ))
    
    def test_spatial_relations(self):
        """测试空间关系提取。"""
        text = "公司总部位于北京市朝阳区建国路88号。"
        doc = self.nlp(text)
        entities = [
            Entity(text="公司总部", label="ORG", start=0, end=4, confidence=0.9),
            Entity(text="北京市朝阳区建国路88号", label="ADDRESS", start=6, end=18, confidence=0.9)
        ]
        
        relations = self.extractor.extract_relations(doc, entities)
        
        self.assertTrue(any(
            r.subject.text == "公司总部" and
            r.predicate == "located_in" and
            r.object.text == "北京市朝阳区建国路88号"
            for r in relations
        ))
    
    def test_temporal_relations(self):
        """测试时间关系提取。"""
        text = "新产品发布会将在2025年3月15日举行。"
        doc = self.nlp(text)
        entities = [
            Entity(text="新产品发布会", label="EVENT", start=0, end=6, confidence=0.9),
            Entity(text="2025年3月15日", label="TIME", start=8, end=19, confidence=0.9)
        ]
        
        relations = self.extractor.extract_relations(doc, entities)
        
        self.assertTrue(any(
            r.subject.text == "新产品发布会" and
            r.predicate == "temporal" and
            r.object.text == "2025年3月15日"
            for r in relations
        ))
    
    def test_bidirectional_relations(self):
        """测试双向关系提取。"""
        text = "蓝色衬衫在黑色裤子旁边。"
        doc = self.nlp(text)
        entities = [
            Entity(text="蓝色衬衫", label="PRODUCT", start=0, end=4, confidence=0.9),
            Entity(text="黑色裤子", label="PRODUCT", start=5, end=9, confidence=0.9)
        ]
        
        relations = self.extractor.extract_relations(doc, entities)
        
        # 验证正向关系
        self.assertTrue(any(
            r.subject.text == "蓝色衬衫" and
            r.predicate == "spatial" and
            r.object.text == "黑色裤子" and
            r.metadata["direction"] == "forward"
            for r in relations
        ))
        
        # 验证反向关系
        self.assertTrue(any(
            r.subject.text == "黑色裤子" and
            r.predicate == "spatial" and
            r.object.text == "蓝色衬衫" and
            r.metadata["direction"] == "reverse"
            for r in relations
        ))
    
    def test_relation_confidence(self):
        """测试关系置信度计算。"""
        text = "这件衬衫是纯棉制作的。"
        doc = self.nlp(text)
        entities = [
            Entity(text="衬衫", label="PRODUCT", start=2, end=4, confidence=0.9),
            Entity(text="纯棉", label="MATERIAL", start=5, end=7, confidence=0.9)
        ]
        
        relations = self.extractor.extract_relations(doc, entities)
        
        # 验证置信度分数
        self.assertTrue(all(0.0 <= r.confidence <= 1.0 for r in relations))
        
        # 验证元数据
        self.assertTrue(all(
            "distance" in r.metadata and
            "text_between" in r.metadata and
            "same_sentence" in r.metadata and
            "syntactic_distance" in r.metadata
            for r in relations
        ))
    
    def test_type_compatibility(self):
        """测试实体类型兼容性评估。"""
        compatibility = self.extractor._evaluate_type_compatibility("PRODUCT", "COLOR")
        self.assertGreater(compatibility, 0.8)
        
        compatibility = self.extractor._evaluate_type_compatibility("ORG", "ADDRESS")
        self.assertGreater(compatibility, 0.7)
        
        compatibility = self.extractor._evaluate_type_compatibility("PERSON", "PRICE")
        self.assertLess(compatibility, 0.6)

if __name__ == '__main__':
    unittest.main()
