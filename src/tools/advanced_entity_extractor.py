"""
Advanced entity extractor with enhanced NER, relation extraction, and attribute normalization.
"""

import spacy
import math
from typing import Any
from spacy.tokens import Doc, Span
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import json
from dataclasses import dataclass
import re
from collections import defaultdict
import numpy as np
from datetime import datetime

@dataclass
class Entity:
    """Enhanced entity representation."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_value: Optional[str] = None
    attributes: Dict[str, str] = None
    metadata: Dict[str, any] = None
    doc: Optional[Doc] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized_value": self.normalized_value,
            "attributes": self.attributes,
            "metadata": self.metadata
        }  # 添加doc引用

@dataclass
class Relation:
    """Represents a relationship between entities."""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    metadata: Dict[str, any] = None

class AttributeNormalizer:
    """Normalizes entity attributes to standard formats."""
    
    def __init__(self):
        self.color_map = {
            # Blue variants
            "azure": "blue",
            "navy": "blue",
            "cyan": "blue",
            "indigo": "blue",
            "turquoise": "blue",
            "teal": "blue",
            "aqua": "blue",
            "cerulean": "blue",
            "cobalt": "blue",
            "sapphire": "blue",
            
            # Red variants
            "crimson": "red",
            "maroon": "red",
            "ruby": "red",
            "scarlet": "red",
            "burgundy": "red",
            "wine": "red",
            "cardinal": "red",
            "carmine": "red",
            "coral": "red",
            "rose": "red",
            
            # Green variants
            "emerald": "green",
            "lime": "green",
            "olive": "green",
            "sage": "green",
            "forest": "green",
            "mint": "green",
            "jade": "green",
            "kelly": "green",
            "pine": "green",
            "moss": "green",
            
            # Other base colors
            "black": "black",
            "white": "white",
            "gray": "gray",
            "grey": "gray",
            "brown": "brown",
            "tan": "brown",
            "beige": "brown",
            "khaki": "brown",
            "purple": "purple",
            "violet": "purple",
            "lavender": "purple",
            "mauve": "purple",
            "pink": "pink",
            "magenta": "pink",
            "fuchsia": "pink",
            "yellow": "yellow",
            "gold": "yellow",
            "amber": "yellow",
            "orange": "orange",
            "peach": "orange",
            "apricot": "orange"
        }
        
        self.size_map = {
            "tiny": "XS",
            "small": "S",
            "medium": "M",
            "large": "L",
            "extra large": "XL",
            "extra-large": "XL"
        }
        
        self.number_pattern = re.compile(r'\d+(\.\d+)?')
        self.price_pattern = re.compile(r'[\$£€]?\s*\d+(\.\d{2})?')
        
    def normalize_color(self, value: str) -> str:
        """Normalize color values."""
        # Convert to lowercase and strip whitespace
        value = value.lower().strip()
        
        # Handle compound colors
        if " " in value:
            words = value.split()
            base_color = words[-1]
            modifier = words[0]
            
            # Handle special compound colors
            if modifier == "navy" and base_color == "blue":
                return "blue"
            if modifier == "sky" and base_color == "blue":
                return "blue"
            if modifier == "forest" and base_color == "green":
                return "green"
            if modifier == "blood" and base_color == "red":
                return "red"
            
            # For light/dark variants, normalize the base color
            if modifier in ["light", "dark", "deep", "pale"]:
                normalized_base = self.color_map.get(base_color)
                if normalized_base:
                    return normalized_base
            
            # Try the full compound color
            compound_color = self.color_map.get(value)
            if compound_color:
                return compound_color
            
            # Default to base color if it can be normalized
            normalized_base = self.color_map.get(base_color)
            if normalized_base:
                return normalized_base
            
            return base_color
        
        # Single word color
        return self.color_map.get(value, value)
        
    def normalize_size(self, value: str) -> str:
        """Normalize size values."""
        value = value.lower().strip()
        return self.size_map.get(value, value.upper())
        
    def normalize_price(self, value: str) -> float:
        """Normalize price values to float."""
        if match := self.price_pattern.search(value):
            return float(re.sub(r'[^\d.]', '', match.group()))
        return None
        
    def normalize_date(self, value: str) -> datetime:
        """Normalize date values to datetime."""
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return None

class RelationExtractor:
    """Enhanced relationship extraction between entities with bidirectional support."""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.verb_pattern = [
            {"POS": "VERB"},
            {"OP": "*"},
            {"DEP": {"IN": ["dobj", "pobj", "nsubj", "attr"]}}
        ]
        
        # Define semantic relation types with enhanced patterns
        self.semantic_relations = {
            # 物理组成关系
            "composition": {
                "patterns": ["made_of", "contains", "composed_of", "consists_of", "constructed_from"],
                "confidence": 0.85,
                "bidirectional": False
            },
            
            # 属性关系
            "attribute": {
                "patterns": [
                    "has_color", "has_size", "has_price", "has_material",
                    "colored_in", "sized_as", "priced_at", "made_from",
                    "features", "characterized_by", "described_as"
                ],
                "confidence": 0.9,
                "bidirectional": False
            },
            
            # 动作关系
            "action": {
                "patterns": [
                    "targets", "filters_by", "modifies", "acts_on",
                    "processes", "transforms", "generates", "produces",
                    "consumes", "utilizes", "applies_to"
                ],
                "confidence": 0.8,
                "bidirectional": True
            },
            
            # 比较关系
            "comparison": {
                "patterns": [
                    "same_as", "different_from", "similar_to", "matches",
                    "contrasts_with", "equals", "exceeds", "less_than",
                    "greater_than", "comparable_to", "alternative_to"
                ],
                "confidence": 0.75,
                "bidirectional": True
            },
            
            # 时间关系
            "temporal": {
                "patterns": [
                    "before", "after", "during", "starts", "ends",
                    "overlaps", "follows", "precedes", "coincides",
                    "occurs_in", "scheduled_for"
                ],
                "confidence": 0.85,
                "bidirectional": False
            },
            
            # 空间关系
            "spatial": {
                "patterns": [
                    "near", "inside", "on_top_of", "beneath", "adjacent_to",
                    "surrounds", "contains", "intersects", "aligned_with",
                    "positioned_at", "located_in"
                ],
                "confidence": 0.8,
                "bidirectional": True
            },
            
            # 属于关系
            "possession": {
                "patterns": [
                    "owns", "belongs_to", "possesses", "has_part",
                    "member_of", "includes", "contains", "comprises",
                    "associated_with", "affiliated_with"
                ],
                "confidence": 0.85,
                "bidirectional": False
            },
            
            # 状态变化关系
            "state_change": {
                "patterns": [
                    "transforms_to", "becomes", "changes_to", "converts_to",
                    "evolves_to", "develops_into", "degrades_to",
                    "improves_to", "deteriorates_to"
                ],
                "confidence": 0.75,
                "bidirectional": False
            },
            
            # 因果关系
            "causation": {
                "patterns": [
                    "causes", "results_in", "leads_to", "triggers",
                    "initiates", "prevents", "enables", "blocks",
                    "influences", "affects", "impacts"
                ],
                "confidence": 0.7,
                "bidirectional": False
            },
            
            # 功能关系
            "functionality": {
                "patterns": [
                    "functions_as", "serves_as", "acts_as", "operates_as",
                    "performs", "supports", "enables", "facilitates",
                    "provides", "delivers", "handles"
                ],
                "confidence": 0.8,
                "bidirectional": False
            }
        }
        
        # Confidence scoring weights
        self.confidence_weights = {
            "distance": 0.25,
            "syntactic": 0.35,
            "semantic": 0.4
        }
        
    def _get_entity_pairs(self, doc: Doc, entities: List[Entity]) -> List[Tuple[Entity, Entity]]:
        """Get potential entity pairs for relation extraction."""
        pairs = []
        entity_spans = {(e.start, e.end): e for e in entities}
        
        for e1 in entities:
            for e2 in entities:
                if e1 != e2:
                    # Check if entities are in a reasonable distance
                    if abs(e1.start - e2.end) < 10:
                        pairs.append((e1, e2))
                        
        return pairs
        
    def _extract_relation_type(self, doc: Doc, subject: Entity, object: Entity) -> Tuple[Optional[str], float]:
        """Extract the relationship type between two entities with confidence score."""
        # Get the text between entities
        start_idx = min(subject.end, object.end)
        end_idx = max(subject.start, object.start)
        between_tokens = doc[start_idx:end_idx]
        
        # Convert labels to strings if they're not already
        subject_label = str(subject.label)
        object_label = str(object.label)
        
        # Initialize confidence components
        syntactic_score = 0.0
        semantic_score = 0.0
        
        # 检查是否是产品-属性关系
        is_product_attribute = subject_label.startswith("PRODUCT_") and object_label in ["COLOR", "SIZE", "PRICE", "MATERIAL"]
        if is_product_attribute:
            semantic_score = 0.9  # 提高产品-属性关系的基础语义分数
            syntactic_score = 0.85  # 提高产品-属性关系的基础语法分数
            
            # 根据属性类型调整分数
            if object_label == "COLOR":
                semantic_score = 0.95  # 颜色属性最常见，给予更高的分数
                return "has_color", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
        
        # 检查语法关系
        for token in between_tokens:
            # 检查组成关系
            if token.lemma_ in ["contain", "include", "consist", "compose", "made", "comprise", "constitute"]:
                syntactic_score = 0.95
                semantic_score = 0.95
                
                # 检查直接依存关系
                if any(t.dep_ in ["dobj", "pobj", "nsubj", "attr"] for t in doc[object.start:object.end]):
                    syntactic_score = 1.0
                    
                # 检查主语-客语关系
                if subject.label.startswith("PRODUCT_") and object.label == "MATERIAL":
                    semantic_score = 1.0
                    
                # 检查关键短语
                between_text = doc[min(subject.end, object.end):max(subject.start, object.start)].text.lower()
                if "made of" in between_text or "made from" in between_text:
                    syntactic_score = 1.0
                    semantic_score = 1.0
                    
                return "composed_of", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
            
            # 检查属性关系
            if token.lemma_ in ["is", "be", "have", "with"]:
                syntactic_score = 0.8
                if is_product_attribute:
                    if object_label == "PRICE":
                        return "has_price", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
                    else:
                        return f"has_{object_label.lower()}", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
            
            # 检查特定动词
            if token.lemma_ in ["cost", "price", "worth"] and object_label == "PRICE":
                syntactic_score = 0.9
                return "has_price", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
            
            # 检查比较关系
            if token.lemma_ in ["like", "similar", "same", "match"]:
                syntactic_score = 0.8
                semantic_score = 0.7
                return "similar_to", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
        
        # 检查动作关系
        if subject_label.startswith("ACTION_"):
            syntactic_score = 0.7
            if object_label.startswith("PRODUCT_"):
                return "targets", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
            if object_label in ["COLOR", "SIZE", "MATERIAL", "PRICE"]:
                return "filters_by", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
        
        # 检查时间和位置关系
        for token in between_tokens:
            # 时间关系
            if token.lemma_ in ["before", "after", "during"]:
                syntactic_score = 0.85
                semantic_score = 0.8
                # 检查直接依存关系
                if any(t.dep_ in ["prep", "advmod"] for t in doc[token.i:token.i+1]):
                    syntactic_score = 0.95
                return token.lemma_, self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
            
            # 空间关系
            elif token.lemma_ in ["near", "next", "beside", "on", "in", "at", "above", "below", "under", "over", "between"]:
                syntactic_score = 0.85
                semantic_score = 0.8
                
                # 检查直接依存关系
                if any(t.dep_ in ["prep", "advmod"] for t in doc[token.i:token.i+1]):
                    syntactic_score = 0.95
                
                # 检查空间短语
                spatial_phrases = [
                    "next to", "close to", "in front of", "behind", "on top of",
                    "underneath", "inside of", "outside of", "to the left of", "to the right of"
                ]
                between_text = doc[min(subject.end, object.end):max(subject.start, object.start)].text.lower()
                if any(phrase in between_text for phrase in spatial_phrases):
                    syntactic_score = 0.95
                    semantic_score = 0.9
                
                # 根据不同的空间关系调整返回值
                if token.lemma_ in ["on", "above", "over"]:
                    return "on_top_of", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
                elif token.lemma_ in ["next", "beside", "near"]:
                    return "near", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
                elif token.lemma_ in ["in", "inside"]:
                    return "inside", self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
                else:
                    return token.lemma_, self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
        
        # 检查一般动词关系
        for token in between_tokens:
            if token.pos_ == "VERB":
                syntactic_score = 0.6
                return token.lemma_, self._calculate_confidence(subject, object, doc, syntactic_score, semantic_score)
        
        # 如果是产品-属性关系，即使没有明确的语法关系也返回关系
        if is_product_attribute and abs(subject.start - object.end) <= 3:
            if object_label == "PRICE":
                return "has_price", self._calculate_confidence(subject, object, doc, 0.6, semantic_score)
            return f"has_{object_label.lower()}", self._calculate_confidence(subject, object, doc, 0.6, semantic_score)
        
        return None, 0.0
        
    def _calculate_confidence(self, subject: Entity, object: Entity, doc: Doc, syntactic_score: float, semantic_score: float) -> float:
        """Calculate overall confidence score for a relationship."""
        # Calculate distance score with exponential decay
        distance = abs(subject.start - object.end)
        distance_score = math.exp(-distance / (len(doc) / 4))  # 使用指数衰减
        
        # 根据实体类型和依存关系调整分数
        if subject.label.startswith("PRODUCT_"):
            # 提高产品-属性关系的置信度
            if object.label in ["COLOR", "SIZE", "PRICE", "MATERIAL"]:
                semantic_score = max(semantic_score, 0.85)
                # 检查直接依存关系
                if any(token.dep_ in ["compound", "amod", "nmod"] for token in doc[subject.start:object.end]):
                    syntactic_score = max(syntactic_score, 0.8)
            
            # 提高组成关系的置信度
            if object.label.startswith("MATERIAL_") or object.label == "MATERIAL":
                semantic_score = max(semantic_score, 0.9)
                syntactic_score = max(syntactic_score, 0.85)
                
                # 检查特定的依存关系和词汇
                between_tokens = doc[subject.start:object.end]
                composition_indicators = ["made", "contains", "composed", "of", "with"]
                if any(token.lemma_ in composition_indicators for token in between_tokens):
                    syntactic_score = max(syntactic_score, 0.95)
        
        # 根据依存路径长度调整语法分数
        path_length = min(len(list(doc[subject.start].ancestors)) + 
                        len(list(doc[object.start].ancestors)), len(doc))
        path_score = math.exp(-path_length / (len(doc) / 4))
        syntactic_score = max(syntactic_score, path_score)
        
        # 使用sigmoid函数组合分数
        combined_score = (
            self.confidence_weights["distance"] * distance_score +
            self.confidence_weights["syntactic"] * syntactic_score +
            self.confidence_weights["semantic"] * semantic_score
        )
        
        # 应用sigmoid函数使得分数更加平滑
        confidence = 1 / (1 + math.exp(-4 * (combined_score - 0.5)))
        
        # 应用最小置信度阈值
        min_confidence = 0.4
        if confidence > min_confidence:
            confidence = max(min_confidence, confidence)
        
        return round(min(1.0, max(0.0, confidence)), 2)
        
    def extract_relations(self, doc: Doc, subjects: List[Entity], objects: List[Entity] = None) -> List[Relation]:
        """Extract bidirectional relationships between entities."""
        relations = []
        seen_pairs = set()  # Track processed entity pairs
        
        # If objects not provided, use all entities as potential objects
        if objects is None:
            objects = subjects
        
        # Try all possible subject-object pairs
        for subject in subjects:
            for object in objects:
                if subject != object:
                    # Create unique pair identifier
                    pair_id = tuple(sorted([id(subject), id(object)]))
                    if pair_id in seen_pairs:
                        continue
                    seen_pairs.add(pair_id)
                    
                    # Try both directions
                    forward_rel, forward_conf = self._extract_relation_type(doc, subject, object)
                    reverse_rel, reverse_conf = self._extract_relation_type(doc, object, subject)
                    
                    # 检查是否是产品-属性关系
                    is_product_attribute = (
                        subject.label.startswith("PRODUCT_") and
                        object.label in ["COLOR", "SIZE", "PRICE", "MATERIAL"]
                    )
                    
                    # 对于产品-属性关系，强制使用正向关系
                    if is_product_attribute:
                        if forward_rel:
                            relations.append(Relation(
                                subject=subject,
                                predicate=forward_rel,
                                object=object,
                                confidence=max(forward_conf, 0.8),  # 提高产品-属性关系的置信度
                                metadata={
                                    "distance": abs(subject.start - object.end),
                                    "text_between": doc[min(subject.end, object.end):max(subject.start, object.start)].text,
                                    "direction": "forward",
                                    "relation_type": "product_attribute"
                                }
                            ))
                    else:
                        # 处理其他类型的关系
                        if forward_rel:
                            relations.append(Relation(
                                subject=subject,
                                predicate=forward_rel,
                                object=object,
                                confidence=forward_conf,
                                metadata={
                                    "distance": abs(subject.start - object.end),
                                    "text_between": doc[min(subject.end, object.end):max(subject.start, object.start)].text,
                                    "direction": "forward",
                                    "relation_type": "general"
                                }
                            ))
                        
                        if reverse_rel and reverse_conf > forward_conf:
                            relations.append(Relation(
                                subject=object,
                                predicate=reverse_rel,
                                object=subject,
                                confidence=reverse_conf,
                                metadata={
                                    "distance": abs(subject.start - object.end),
                                    "text_between": doc[min(subject.end, object.end):max(subject.start, object.start)].text,
                                    "direction": "reverse",
                                    "relation_type": "general"
                                }
                            ))
        
        # Sort relations by confidence
        relations.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicate or conflicting relations
        final_relations = []
        seen_entity_pairs = set()
        
        for rel in relations:
            pair_key = (id(rel.subject), id(rel.object))
            reverse_key = (id(rel.object), id(rel.subject))
            
            # 如果是产品-属性关系，总是保留
            if rel.metadata.get("relation_type") == "product_attribute":
                final_relations.append(rel)
                seen_entity_pairs.add(pair_key)
                seen_entity_pairs.add(reverse_key)
            # 对于其他关系，只在没有冲突时添加
            elif pair_key not in seen_entity_pairs and reverse_key not in seen_entity_pairs:
                final_relations.append(rel)
                seen_entity_pairs.add(pair_key)
                seen_entity_pairs.add(reverse_key)
        
        return final_relations

class AdvancedEntityExtractor:
    """Advanced entity extraction with enhanced capabilities."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize the advanced entity extractor."""
        self.nlp = spacy.load(model_name)
        self.relation_extractor = RelationExtractor(self.nlp)
        self.attribute_normalizer = AttributeNormalizer()
        
        # 初始化实体规则器
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        # 加载并添加自定义模式
        self.custom_patterns = self._load_custom_patterns()
        for category, patterns in self.custom_patterns.items():
            for pattern in patterns:
                ruler.add_patterns([pattern])
        
    def _load_custom_patterns(self) -> Dict[str, List[Dict]]:
        """加载自定义实体模式"""
        patterns = {
            "product": [
                # 单词服装
                {"pattern": [{"LOWER": "shirt"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "t-shirt"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "tshirt"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "jacket"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "pants"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "jeans"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "trousers"}], "label": "PRODUCT_CLOTHING"},
                
                # 复合词服装
                {"pattern": [{"LOWER": "t"}, {"LOWER": "shirt"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "cotton"}, {"LOWER": "shirt"}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": "dress"}, {"LOWER": "shirt"}], "label": "PRODUCT_CLOTHING"},
                
                # 电子产品
                {"pattern": [{"LOWER": "phone"}], "label": "PRODUCT_ELECTRONICS"},
                {"pattern": [{"LOWER": "smartphone"}], "label": "PRODUCT_ELECTRONICS"},
                {"pattern": [{"LOWER": "headphones"}], "label": "PRODUCT_ELECTRONICS"},
                
                # 媒体产品
                {"pattern": [{"LOWER": "book"}], "label": "PRODUCT_MEDIA"},
                {"pattern": [{"LOWER": "magazine"}], "label": "PRODUCT_MEDIA"}
            ],
            "attribute": [
                # 基本颜色
                {"pattern": [{"LOWER": {"IN": list(self.attribute_normalizer.color_map.keys())}}], "label": "COLOR"},
                
                # 复合颜色模式 - 修饰词 + 颜色
                {"pattern": [{"LOWER": {"IN": ["light", "dark", "deep", "pale", "bright", "soft"]}},
                            {"LOWER": {"IN": list(self.attribute_normalizer.color_map.keys())}}], "label": "COLOR"},
                
                # 特殊复合颜色
                {"pattern": [{"LOWER": "navy"}, {"LOWER": "blue"}], "label": "COLOR"},
                {"pattern": [{"LOWER": "sky"}, {"LOWER": "blue"}], "label": "COLOR"},
                {"pattern": [{"LOWER": "forest"}, {"LOWER": "green"}], "label": "COLOR"},
                {"pattern": [{"LOWER": "blood"}, {"LOWER": "red"}], "label": "COLOR"},
                
                # 颜色组合
                {"pattern": [{"LOWER": {"IN": list(self.attribute_normalizer.color_map.keys())}},
                            {"LOWER": {"IN": list(self.attribute_normalizer.color_map.keys())}}], "label": "COLOR"},
                
                # 单词尺寸
                {"pattern": [{"LOWER": {"IN": list(self.attribute_normalizer.size_map.keys()) + ["xs", "s", "m", "l", "xl", "xxl", "xxxl"]}}], "label": "SIZE"},
                
                # 复合词尺寸
                {"pattern": [{"LOWER": "extra"}, {"LOWER": "small"}], "label": "SIZE"},
                {"pattern": [{"LOWER": "extra"}, {"LOWER": "large"}], "label": "SIZE"},
                {"pattern": [{"LOWER": "extra"}, {"LOWER": "extra"}, {"LOWER": "large"}], "label": "SIZE"},
                {"pattern": [{"LOWER": "size"}, {"LOWER": {"IN": ["xs", "s", "m", "l", "xl", "xxl", "xxxl"]}}], "label": "SIZE"},
                
                # 价格模式
                {"pattern": [{"TEXT": {"REGEX": "\\$\\d+(\\.\\d{2})?"}}], "label": "PRICE"},  # $XX.XX
                {"pattern": [{"TEXT": "$"}, {"LIKE_NUM": True}], "label": "PRICE"},  # $ XX
                {"pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["dollars", "usd"]}}], "label": "PRICE"},  # XX dollars/USD
                {"pattern": [{"LOWER": "under"}, {"TEXT": {"REGEX": "\\$\\d+(\\.\\d{2})?"}}], "label": "PRICE"},  # under $XX
                {"pattern": [{"LOWER": "less"}, {"LOWER": "than"}, {"TEXT": {"REGEX": "\\$\\d+(\\.\\d{2})?"}}], "label": "PRICE"},  # less than $XX
                {"pattern": [{"LOWER": "costs"}, {"TEXT": {"REGEX": "\\$\\d+(\\.\\d{2})?"}}], "label": "PRICE"},  # costs $XX
                
                # 材质
                {"pattern": [{"LOWER": {"IN": [
                    "cotton", "wool", "polyester", "leather",
                    "silk", "linen", "denim", "nylon",
                    "spandex", "rayon", "cashmere", "fleece",
                    "velvet", "suede", "canvas", "mesh"
                ]}}], "label": "MATERIAL"}
            ],
            "action": [
                # 搜索相关
                {"pattern": [{"LEMMA": {"IN": ["search", "find", "look", "browse", "show", "display"]}}], "label": "ACTION_SEARCH"},
                # 购买相关
                {"pattern": [{"LEMMA": {"IN": ["buy", "purchase", "order", "get", "want"]}}], "label": "ACTION_PURCHASE"},
                # 比较相关
                {"pattern": [{"LEMMA": {"IN": ["compare", "contrast", "versus", "vs"]}}], "label": "ACTION_COMPARE"},
                # 价格相关
                {"pattern": [{"LEMMA": {"IN": ["cost", "price", "worth"]}}], "label": "ACTION_PRICE"}
            ]
        }
        return patterns
        
    def _normalize_entity(self, entity: Entity) -> Entity:
        """Normalize entity values and attributes based on type."""
        if not entity.attributes:
            entity.attributes = {}
        
        # 保存原始文本
        entity.attributes["original_text"] = entity.text
        
        if entity.label == "COLOR":
            # 处理颜色实体
            text = entity.text.lower()
            words = text.split()
            entity.attributes["is_compound"] = len(words) > 1
            
            if entity.attributes["is_compound"]:
                # 处理特殊复合颜色
                if words[0] == "navy" and words[1] == "blue":
                    entity.normalized_value = "blue"
                    entity.attributes["modifier"] = "navy"
                    entity.attributes["original_color"] = "blue"
                    entity.attributes["is_standard"] = True
                    entity.confidence = 0.99
                    return entity
                
                # 处理其他复合颜色
                if words[-1] in self.attribute_normalizer.color_map:
                    # 如果最后一个词是标准颜色
                    main_color = words[-1]
                    entity.attributes["modifier"] = " ".join(words[:-1])
                    entity.normalized_value = self.attribute_normalizer.color_map[main_color]
                    entity.attributes["original_color"] = main_color
                    entity.attributes["is_standard"] = True
                    entity.confidence = 0.95
                else:
                    # 如果最后一个词不是标准颜色，尝试其他词
                    for word in words:
                        if word in self.attribute_normalizer.color_map:
                            main_color = word
                            entity.attributes["modifier"] = " ".join(w for w in words if w != word)
                            entity.normalized_value = self.attribute_normalizer.color_map[main_color]
                            entity.attributes["original_color"] = main_color
                            entity.attributes["is_standard"] = True
                            entity.confidence = 0.9
                            break
                    else:
                        main_color = words[-1]  # 如果没有找到标准颜色，使用最后一个词
                        entity.attributes["modifier"] = " ".join(words[:-1])
                        entity.normalized_value = main_color
                        entity.attributes["original_color"] = main_color
                        entity.attributes["is_standard"] = False
                        entity.confidence = 0.85
            else:
                # 处理单个颜色词
                main_color = text
                entity.attributes["modifier"] = ""
                
                # 检查是否是标准颜色
                if main_color in self.attribute_normalizer.color_map:
                    entity.normalized_value = self.attribute_normalizer.color_map[main_color]
                    entity.attributes["original_color"] = main_color
                    entity.attributes["is_standard"] = True
                    entity.confidence = 0.95
                else:
                    entity.normalized_value = main_color
                    entity.attributes["original_color"] = main_color
                    entity.attributes["is_standard"] = False
                    entity.confidence = 0.85
            
        elif entity.label == "SIZE":
            text = entity.text.lower()
            entity.normalized_value = self.attribute_normalizer.normalize_size(text)
            entity.confidence = 0.95 if text in self.attribute_normalizer.size_map else 0.9
            entity.attributes["is_standard"] = text in self.attribute_normalizer.size_map
            
        elif entity.label == "PRICE":
            text = entity.text.lower()
            normalized_price = self.attribute_normalizer.normalize_price(text)
            if normalized_price is not None:
                entity.normalized_value = str(normalized_price)
                entity.confidence = 0.95
                entity.attributes["currency"] = "USD"  # 默认使用USD
            
        elif entity.label == "MATERIAL":
            text = entity.text.lower()
            entity.normalized_value = text
            entity.confidence = 0.9
            
        elif entity.label.startswith("PRODUCT_"):
            text = entity.text.lower()
            entity.normalized_value = text
            entity.attributes["category"] = entity.label.split("_")[1].lower()
            entity.confidence = 0.9
        
        # 添加通用属性
        entity.attributes["normalized"] = True
        if "source" not in entity.attributes:
            entity.attributes["source"] = "pattern"
        if "match_id" not in entity.attributes:
            entity.attributes["match_id"] = str(hash(entity.text + entity.label))
        
        return entity
        
    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and their relationships from text with enhanced processing.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Dictionary containing extracted entities, relations and analysis
        """
        doc = self.nlp(text)
        
        # Get all potential entity spans from multiple sources
        spans = []
        
        # 1. Get spans from custom patterns
        matcher = spacy.matcher.Matcher(self.nlp.vocab)
        pattern_to_label = {}
        
        # 先处理复合颜色模式
        color_patterns = []
        for pattern in self.custom_patterns["attribute"]:
            if pattern["label"] == "COLOR":
                # 处理所有颜色模式
                pattern_id = self.nlp.vocab.strings.add("COLOR")
                pattern_to_label[pattern_id] = "COLOR"
                matcher.add("COLOR", [pattern["pattern"]])
                color_patterns.append(pattern)
        
        # 然后处理其他模式
        for category, patterns in self.custom_patterns.items():
            for pattern in patterns:
                # 跳过已处理的颜色模式
                if pattern in color_patterns:
                    continue
                    
                label = pattern["label"]
                pattern_id = self.nlp.vocab.strings.add(label)
                pattern_to_label[pattern_id] = label
                matcher.add(label, [pattern["pattern"]])
        
        matches = matcher(doc)
        for match_id, start, end in matches:
            match_text = doc[start:end].text
            label = pattern_to_label[match_id]
            
            # 处理颜色实体
            if label == "COLOR":
                words = match_text.split()
                is_compound = len(words) > 1
                confidence = 0.95 if is_compound else 0.9
                
                # 处理复合颜色
                if is_compound:
                    base_color = words[-1].lower()
                    modifier = " ".join(words[:-1]).lower()
                    
                    # 根据修饰词调整置信度
                    if modifier in ["light", "dark", "deep", "pale"]:
                        confidence = 0.95
                    elif modifier in ["navy", "sky", "forest", "blood"]:
                        confidence = 0.98
                        if base_color in ["blue", "green", "red"]:
                            confidence = 0.99
                else:
                    base_color = match_text.lower()
                    modifier = ""
                
                # 检查颜色是否在标准映射中
                normalized_color = self.attribute_normalizer.normalize_color(base_color)
                if normalized_color != base_color:
                    confidence = max(confidence, 0.95)
                
                spans.append({
                    "text": match_text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                    "source": "pattern",
                    "metadata": {
                        "match_id": str(match_id),
                        "is_compound": is_compound,
                        "original_text": match_text,
                        "base_color": base_color,
                        "modifier": modifier,
                        "normalized_color": normalized_color
                    }
                })
            else:
                spans.append({
                    "text": match_text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": 0.9,
                    "source": "pattern",
                    "metadata": {"match_id": str(match_id)}
                })
        
        # 2. Get spans from spaCy NER
        for ent in doc.ents:
            spans.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start,
                "end": ent.end,
                "confidence": 0.95,
                "source": "spacy",
                "metadata": {
                    "pos": ent.root.pos_,
                    "dep": ent.root.dep_
                }
            })
        
        # 3. Sort and deduplicate spans
        spans.sort(key=lambda x: (-x["confidence"], -len(x["text"]), x["start"]))  # Sort by confidence, length, and position
        
        # Group spans by sentence
        sentence_spans = {}
        for span in spans:
            sent = doc[span["start"]].sent
            sent_id = sent.start
            if sent_id not in sentence_spans:
                sentence_spans[sent_id] = []
            sentence_spans[sent_id].append(span)
        
        # Process spans sentence by sentence
        final_spans = []
        for sent_id, sent_spans in sentence_spans.items():
            used_spans = set()
            used_values = {}
            
            for span in sent_spans:
                span_range = (span["start"], span["end"])
                value_key = (span["label"], span["text"].lower())
                overlapping = False
                
                # 检查重叠跨度
                for used_span in used_spans:
                    if (span_range[0] <= used_span[1] and span_range[1] >= used_span[0]):
                        # 如果当前跨度是复合颜色，且置信度更高，删除之前的跨度
                        if span["label"] == "COLOR" and len(span["text"].split()) > 1 and span["confidence"] > 0.9:
                            final_spans = [s for s in final_spans if s["start"] != used_span[0] or s["end"] != used_span[1]]
                            used_spans.remove(used_span)
                            break
                        else:
                            overlapping = True
                            break
                
                # 检查句子内的重复值
                if value_key in used_values:
                    # 对于颜色实体，使用更智能的去重逻辑
                    if span["label"] == "COLOR":
                        existing_span = used_values[value_key]
                        # 如果当前颜色是复合颜色，且置信度更高
                        if len(span["text"].split()) > 1 and span["confidence"] > existing_span["confidence"]:
                            final_spans = [s for s in final_spans if s != existing_span]
                            used_values[value_key] = span
                            overlapping = False
                        else:
                            overlapping = True
                    # 对于其他实体，只在它们非常接近时去重
                    elif abs(span["start"] - used_values[value_key]["start"]) < 5:
                        overlapping = True
                
                if not overlapping:
                    final_spans.append(span)
                    used_spans.add(span_range)
                    used_values[value_key] = span
        
        # Convert final spans to entities
        entities = []
        for span in final_spans:
            entity = Entity(
                text=span["text"],
                label=span["label"],
                start=span["start"],
                end=span["end"],
                confidence=span["confidence"],
                doc=doc
            )
            
            # Normalize entity values
            self._normalize_entity(entity)
            entities.append(entity)
        
        # Add metadata to entities
        for entity in entities:
            span_data = next(s for s in final_spans if s["start"] == entity.start and s["end"] == entity.end)
            
            # Add metadata and source if available
            if "metadata" in span_data:
                entity.metadata = span_data["metadata"]
            if "source" in span_data:
                if not entity.metadata:
                    entity.metadata = {}
                entity.metadata["source"] = span_data["source"]
            
            # Add context metadata
            if not entity.metadata:
                entity.metadata = {}
            left_context = doc[max(0, entity.start-5):entity.start].text
            right_context = doc[entity.end:min(len(doc), entity.end+5)].text
            entity.metadata["context"] = {
                "left": left_context,
                "right": right_context
            }
        
        # Extract relations with enhanced context
        relations = []
        for i, e1 in enumerate(entities):
            # Look ahead up to 3 entities
            for j in range(i + 1, min(i + 4, len(entities))):
                e2 = entities[j]
                between_text = doc[e1.end:e2.start].text.strip()
                
                # Extract bidirectional relations
                for direction in [(e1, e2, "forward"), (e2, e1, "reverse")]:
                    subj, obj, dir_type = direction
                    if rels := self.relation_extractor.extract_relations(doc, [subj], [obj]):
                        for rel in rels:
                            rel.metadata.update({
                                "between_text": between_text,
                                "distance": j - i,
                                "direction": dir_type,
                                "sentence_id": subj.doc[subj.start].sent.start if subj.doc else None
                            })
                            relations.append(rel)
        
        # Deduplicate and adjust relation confidence
        unique_relations = []
        seen = set()
        
        for rel in sorted(relations, key=lambda x: (-x.confidence, x.metadata.get("distance", 0))):
            key = (id(rel.subject), rel.predicate, id(rel.object))
            if key not in seen:
                seen.add(key)
                # Adjust confidence based on distance and context
                distance = rel.metadata.get("distance", 1)
                same_sentence = rel.metadata.get("sentence_id") == rel.subject.doc[rel.subject.start].sent.start
                rel.confidence *= max(0.5, 1.0 - (distance - 1) * 0.1)
                if same_sentence:
                    rel.confidence *= 1.1
                unique_relations.append(rel)
        
        # Analyze entity distribution
        distribution = self.analyze_entity_distribution(entities)
        
        return {
            "entities": entities,
            "relations": unique_relations,
            "distribution": distribution,
            "meta": {
                "text_length": len(text),
                "entity_density": len(entities) / len(text) if text else 0,
                "unique_labels": len({e.label for e in entities}),
                "relation_density": len(unique_relations) / len(entities) if entities else 0
            }
        }
        
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities and duplicates, keeping ones with higher confidence."""
        # Group entities by sentence
        sentence_entities = {}
        for entity in entities:
            if entity.doc:
                sent = entity.doc[entity.start].sent
                sent_id = sent.start
                if sent_id not in sentence_entities:
                    sentence_entities[sent_id] = []
                sentence_entities[sent_id].append(entity)
        
        final_entities = []
        
        # Process each sentence separately
        for sent_id, sent_entities in sentence_entities.items():
            # Sort by confidence and text length within sentence
            sorted_entities = sorted(sent_entities, key=lambda x: (-x.confidence, -len(x.text)))
            used_spans = set()
            used_values = {}
            
            for entity in sorted_entities:
                span = (entity.start, entity.end)
                value_key = (entity.label, entity.normalized_value) if entity.normalized_value else None
                overlapping = False
                
                # Check for overlapping spans
                for used_span in used_spans:
                    if (span[0] <= used_span[1] and span[1] >= used_span[0]):
                        # Special handling for color entities
                        if entity.label == "COLOR" and entity.confidence > 0.95:
                            # Remove the existing entity if current one is better
                            final_entities = [e for e in final_entities if not (e.start == used_span[0] and e.end == used_span[1])]
                            used_spans.remove(used_span)
                            break
                        else:
                            overlapping = True
                            break
                
                # Check for duplicate normalized values within sentence
                if value_key and value_key in used_values:
                    # For color entities, use special handling
                    if entity.label == "COLOR":
                        existing_span = used_values[value_key]
                        existing_entity = next(e for e in final_entities if e.start == existing_span[0] and e.end == existing_span[1])
                        
                        # If current entity is a compound color and has higher confidence
                        if len(entity.text.split()) > 1 and entity.confidence > existing_entity.confidence:
                            final_entities.remove(existing_entity)
                            used_spans.remove(existing_span)
                            used_values.pop(value_key)
                        else:
                            overlapping = True
                    # For other entities, only deduplicate if they are very close
                    elif abs(span[0] - used_values[value_key][0]) <= 5:
                        overlapping = True
                
                if not overlapping:
                    final_entities.append(entity)
                    used_spans.add(span)
                    if value_key:
                        used_values[value_key] = span
        
        return sorted(final_entities, key=lambda x: x.start)
        
    def analyze_entity_distribution(self, entities: List[Entity]) -> Dict[str, Any]:
        """Analyze the distribution and patterns of entities with enhanced statistics.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            dict: Comprehensive distribution statistics and patterns
        """
        # Initialize statistics collectors
        stats = {
            "label_stats": defaultdict(lambda: {
                "count": 0,
                "confidences": [],
                "lengths": [],
                "positions": [],
                "normalized_values": defaultdict(int),
                "attributes": defaultdict(list)
            }),
            "global_stats": {
                "total_entities": len(entities),
                "unique_labels": len({e.label for e in entities}),
                "avg_confidence": sum(e.confidence for e in entities) / len(entities) if entities else 0,
                "entity_length_distribution": defaultdict(int)
            },
            "pattern_analysis": {
                "consecutive_pairs": defaultdict(int),
                "label_transitions": defaultdict(int),
                "position_clusters": defaultdict(list)
            },
            "label_counts": defaultdict(int),
            "normalized_counts": defaultdict(int)
        }
        
        # Process each entity
        prev_entity = None
        for entity in sorted(entities, key=lambda x: x.start):
            label_stat = stats["label_stats"][entity.label]
            
            # Update basic counts and lists
            label_stat["count"] += 1
            label_stat["confidences"].append(entity.confidence)
            label_stat["lengths"].append(entity.end - entity.start)
            label_stat["positions"].append(entity.start)
            
            # Update label counts
            stats["label_counts"][entity.label] += 1
            
            # Track normalized values
            if entity.normalized_value:
                label_stat["normalized_values"][entity.normalized_value] += 1
                stats["normalized_counts"][entity.normalized_value] += 1
                
                # Special handling for color entities
                if entity.label == "COLOR":
                    # Add both original and normalized values
                    original_value = entity.text.lower()
                    if original_value != entity.normalized_value:
                        label_stat["normalized_values"][original_value] += 1
                        stats["normalized_counts"][original_value] += 1
            
            # Track attributes
            if entity.attributes:
                for attr, value in entity.attributes.items():
                    label_stat["attributes"][attr].append(value)
                    
                    # Special handling for color attributes
                    if entity.label == "COLOR" and attr == "modifier":
                        compound_color = f"{value} {entity.attributes.get('original_color', '')}".strip()
                        if compound_color:
                            label_stat["normalized_values"][compound_color] += 1
                            stats["normalized_counts"][compound_color] += 1
            
            # Update global statistics
            length = entity.end - entity.start
            stats["global_stats"]["entity_length_distribution"][length] += 1
            
            # Analyze patterns
            if prev_entity:
                # Track consecutive entity pairs
                pair_key = f"{prev_entity.label}->{entity.label}"
                stats["pattern_analysis"]["consecutive_pairs"][pair_key] += 1
                
                # Track label transitions
                if entity.start - prev_entity.end <= 5:  # Close entities
                    transition_key = f"{prev_entity.label}->{entity.label}"
                    stats["pattern_analysis"]["label_transitions"][transition_key] += 1
                
                # Track position clusters
                if entity.start - prev_entity.end <= 10:  # Entities in same cluster
                    cluster_key = f"cluster_{len(stats['pattern_analysis']['position_clusters'])}"
                    current_clusters = stats["pattern_analysis"]["position_clusters"]
                    if not current_clusters or \
                       (len(current_clusters) > 0 and entity.start - list(current_clusters.values())[-1][-1].end > 10):
                        stats["pattern_analysis"]["position_clusters"][cluster_key] = [entity]
                    else:
                        stats["pattern_analysis"]["position_clusters"][cluster_key].append(entity)
            
            prev_entity = entity
        
        # Calculate detailed statistics for each label
        for label, label_stat in stats["label_stats"].items():
            if label_stat["count"] > 0:
                # Confidence statistics
                confidences = np.array(label_stat["confidences"])
                label_stat["confidence_stats"] = {
                    "mean": float(np.mean(confidences)),
                    "std": float(np.std(confidences)),
                    "min": float(np.min(confidences)),
                    "max": float(np.max(confidences)),
                    "quartiles": [float(x) for x in np.percentile(confidences, [25, 50, 75])]
                }
                
                # Length statistics
                lengths = np.array(label_stat["lengths"])
                label_stat["length_stats"] = {
                    "mean": float(np.mean(lengths)),
                    "std": float(np.std(lengths)),
                    "distribution": {int(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))}
                }
                
                # Position statistics
                positions = np.array(label_stat["positions"])
                label_stat["position_stats"] = {
                    "mean": float(np.mean(positions)),
                    "std": float(np.std(positions)),
                    "clusters": [[int(x) for x in g] for g in np.split(positions, np.where(np.diff(positions) > 10)[0] + 1)]
                }
                
                # Attribute analysis
                label_stat["attribute_stats"] = {
                    attr: {
                        "unique_values": len(set(values)),
                        "most_common": sorted(set(values), key=lambda x: values.count(x), reverse=True)[:3],
                        "distribution": dict(Counter(values))
                    }
                    for attr, values in label_stat["attributes"].items()
                }
                
                # Cleanup temporary lists
                del label_stat["confidences"]
                del label_stat["lengths"]
                del label_stat["positions"]
        
        # Convert defaultdicts and entities to regular dicts
        def convert_to_dict(obj):
            if isinstance(obj, Entity):
                return obj.to_dict()
            elif isinstance(obj, defaultdict):
                return dict(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return convert_to_dict(stats)

    def _calculate_entity_confidence(self, span: Span) -> float:
        """Calculate confidence score for an entity span with enhanced metrics."""
        base_confidence = 0.8
        
        # 基于实体长度调整置信度
        length_factor = min(1.0, max(0.0, 1.0 - (len(span) - 2) * 0.1))
        base_confidence *= length_factor
        
        # 基于实体标签调整置信度
        label_boost = {
            "PERSON": 0.15,
            "ORG": 0.12,
            "GPE": 0.12,
            "COLOR": 0.1,
            "PRODUCT": 0.08,
            "MATERIAL": 0.08
        }.get(span.label_, 0.0)
        base_confidence += label_boost
        
        # 基于句法结构调整置信度
        if span.root.dep_ in ["nsubj", "dobj", "pobj"]:
            base_confidence += 0.05
        
        # 基于上下文一致性调整置信度
        context_score = self._evaluate_context_consistency(span)
        base_confidence *= (1.0 + context_score * 0.2)
        
        return min(1.0, max(0.0, base_confidence))
        
    def _evaluate_context_consistency(self, span: Span) -> float:
        """评估实体的上下文一致性，使用增强的规则"""
        consistency_score = 0.0
        context_window = 3  # 上下文窗口大小
        
        # 获取左右上下文窗口
        left_window = span.doc[max(0, span.start - context_window):span.start]
        right_window = span.doc[span.end:min(len(span.doc), span.end + context_window)]
        
        # 1. 词性序列评估
        pos_patterns = {
            # 左侧模式
            "left": {
                ("DET", "ADJ", "NOUN"): 0.3,  # 完整的名词短语
                ("DET", "ADJ"): 0.25,       # 部分名词短语
                ("ADP", "DET"): 0.2,       # 介词短语
                ("VERB", "ADP"): 0.2,      # 动词短语
            },
            # 右侧模式
            "right": {
                ("VERB", "ADP", "DET"): 0.3,  # 动词短语
                ("PUNCT", "CCONJ"): 0.2,     # 并列结构
                ("ADP", "DET", "NOUN"): 0.25, # 介词短语
            }
        }
        
        # 检查左右上下文的词性模式
        left_pos = tuple(t.pos_ for t in left_window)
        right_pos = tuple(t.pos_ for t in right_window)
        
        for pattern, score in pos_patterns["left"].items():
            if pattern == left_pos[-len(pattern):]:
                consistency_score += score
                break
        
        for pattern, score in pos_patterns["right"].items():
            if pattern == right_pos[:len(pattern)]:
                consistency_score += score
                break
        
        # 2. 依存关系评估
        dep_patterns = {
            "compound": 0.15,    # 复合词
            "amod": 0.15,       # 形容词修饰
            "nmod": 0.1,        # 名词修饰
            "case": 0.1,        # 介词关系
            "det": 0.1         # 限定词
        }
        
        for token in span:
            if token.dep_ in dep_patterns:
                consistency_score += dep_patterns[token.dep_]
        
        # 3. 实体边界评估
        if span.start > 0:
            prev_ents = [e for e in span.doc.ents if e.end == span.start]
            if prev_ents:
                # 检查前一个实体的类型兼容性
                prev_compatibility = self._evaluate_type_compatibility(prev_ents[0].label_, span.label_)
                consistency_score += 0.1 * prev_compatibility
        
        if span.end < len(span.doc):
            next_ents = [e for e in span.doc.ents if e.start == span.end]
            if next_ents:
                # 检查后一个实体的类型兼容性
                next_compatibility = self._evaluate_type_compatibility(span.label_, next_ents[0].label_)
                consistency_score += 0.1 * next_compatibility
        
        # 4. 特殊模式评估
        if span.label_ == "PRODUCT":
            # 产品名称通常包含数字或大写字母
            if any(c.isupper() for c in span.text) or any(c.isdigit() for c in span.text):
                consistency_score += 0.1
        elif span.label_ == "COLOR":
            # 颜色前后通常有特定词汇
            color_context = {"bright", "dark", "light", "deep", "pale"}
            if any(t.text.lower() in color_context for t in left_window + right_window):
                consistency_score += 0.15
        
        return min(1.0, consistency_score)
        
    def _calculate_relation_confidence(self, source: Entity, target: Entity, path_length: int) -> float:
        """计算关系的置信度分数"""
        base_confidence = 0.7
        
        # 基于实体置信度
        entity_confidence = (source.confidence + target.confidence) / 2
        base_confidence *= entity_confidence
        
        # 基于路径长度调整
        path_factor = 1.0 / (1.0 + math.log(1 + path_length))
        base_confidence *= path_factor
        
        # 基于实体类型组合
        type_compatibility = self._evaluate_type_compatibility(source.label, target.label)
        base_confidence *= (1.0 + type_compatibility * 0.3)
        
        return min(1.0, max(0.0, base_confidence))
        
    def _evaluate_type_compatibility(self, source_type: str, target_type: str) -> float:
        """评估两个实体类型之间的兼容性，使用增强的兼容性矩阵"""
        # 定义实体类型兼容性矩阵
        compatibility_matrix = {
            # 产品相关
            ("PRODUCT", "COLOR"): 0.85,
            ("PRODUCT", "MATERIAL"): 0.85,
            ("PRODUCT", "SIZE"): 0.8,
            ("PRODUCT", "PRICE"): 0.8,
            ("PRODUCT", "BRAND"): 0.9,
            ("PRODUCT", "MODEL"): 0.9,
            
            # 属性间关系
            ("COLOR", "MATERIAL"): 0.7,
            ("COLOR", "PATTERN"): 0.75,
            ("SIZE", "PRICE"): 0.6,
            ("MATERIAL", "PRICE"): 0.65,
            ("BRAND", "PRICE"): 0.7,
            
            # 位置和时间关系
            ("LOCATION", "TIME"): 0.6,
            ("LOCATION", "EVENT"): 0.7,
            ("TIME", "EVENT"): 0.7,
            
            # 人物和组织关系
            ("PERSON", "ORG"): 0.8,
            ("PERSON", "ROLE"): 0.85,
            ("ORG", "LOCATION"): 0.75,
            
            # 数量和单位关系
            ("QUANTITY", "UNIT"): 0.9,
            ("PRICE", "CURRENCY"): 0.9,
            ("MEASUREMENT", "UNIT"): 0.9
        }
        
        # 检查正向组合
        forward_key = (source_type, target_type)
        if forward_key in compatibility_matrix:
            return compatibility_matrix[forward_key]
        
        # 检查反向组合
        backward_key = (target_type, source_type)
        if backward_key in compatibility_matrix:
            return compatibility_matrix[backward_key] * 0.95  # 反向关系略微降低置信度
        
        # 处理相同类型
        if source_type == target_type:
            return 0.5  # 相同类型之间有一定关联，但不应过高
        
        # 默认兼容性
        return 0.3
