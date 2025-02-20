"""
增强的实体关系提取器，支持中英文关系抽取。
"""

from typing import List, Tuple, Dict, Any, Optional
from spacy.tokens import Doc, Span
from dataclasses import dataclass
import numpy as np

@dataclass
class Entity:
    """实体表示"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_value: Optional[str] = None
    attributes: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    doc: Optional[Doc] = None

@dataclass
class Relation:
    """关系表示"""
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    metadata: Dict[str, Any] = None

class RelationExtractor:
    """增强的实体关系提取，支持双向关系和中文。"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        # 动词模式，支持中文和英文
        self.verb_patterns = {
            "en": [
                {"POS": "VERB"},
                {"OP": "*"},
                {"DEP": {"IN": ["dobj", "pobj", "nsubj", "attr"]}}
            ],
            "zh": [
                {"POS": "VERB"},
                {"OP": "*"},
                {"DEP": {"IN": ["dobj", "nsubj", "attr", "nmod"]}}
            ]
        }
        
        # 初始化关系模式
        self._init_relation_patterns()
        
    def _init_relation_patterns(self):
        """初始化关系模式"""
        # 中文关系模式
        self.zh_relation_patterns = {
            # 属性关系
            "has_color": {
                "patterns": ["是", "为", "颜色", "色的", "色是", "色为"],
                "confidence": 0.9,
                "bidirectional": False
            },
            "has_size": {
                "patterns": ["是", "为", "尺码", "尺寸", "尺码为", "尺寸为", "XL", "L", "M", "S"],
                "confidence": 0.9,
                "bidirectional": False
            },
            "has_price": {
                "patterns": ["售价", "价格", "价钱", "元", "元钱", "价格为", "售价为"],
                "confidence": 0.9,
                "bidirectional": False
            },
            # 产地关系
            "origin": {
                "patterns": [
                    "产自", "来自", "生产于", "制造于",
                    "加工于", "研发于"
                ],
                "confidence": 0.85,
                "bidirectional": False
            },
            # 时间关系
            "temporal": {
                "patterns": [
                    "在", "于", "之前", "之后", "期间",
                    "开始", "结束", "持续", "举行", "将在", "将于", "将"
                ],
                "confidence": 0.9,
                "bidirectional": False
            },
            # 空间关系
            "spatial": {
                "patterns": [
                    # 基础位置关系
                    "在", "位于", "处于", "坐落于", "座落于", "坐落在", "座落在",
                    # 方位关系
                    "上", "下", "左", "右", "前", "后", "内", "外",
                    "上面", "下面", "左边", "右边", "前面", "后面", "里面", "外面",
                    "上方", "下方", "左方", "右方", "前方", "后方", "内部", "外部",
                    # 距离关系
                    "附近", "旁边", "周围", "远离", "靠近", "紧邻", "毗邻",
                    "旁", "侧", "边", "角", "中", "间",
                    # 包含关系
                    "中的", "内的", "里的", "当中的", "之中的",
                    # 组合关系词
                    "位于...附近", "在...旁边", "靠近...方向"
                ],
                "confidence": 0.8,
                "bidirectional": True
            },
            # 部分-整体关系
            "part_of": {
                "patterns": [
                    "的一部分", "包含", "组成", "构成", "包括",
                    "属于", "从属于", "隶属于", "归属于",
                    "由...组成", "由...构成", "包含...在内"
                ],
                "confidence": 0.9,
                "bidirectional": False
            },
            # 功能关系
            "function": {
                "patterns": [
                    "用于", "用来", "用作", "作为", "充当",
                    "发挥...作用", "起...作用", "具有...功能"
                ],
                "confidence": 0.85,
                "bidirectional": False
            },
            # 状态关系
            "state": {
                "patterns": [
                    "处于", "保持", "维持", "呈现", "表现为",
                    "显示为", "变成", "成为", "转变为"
                ],
                "confidence": 0.8,
                "bidirectional": False
            },
            # 位置关系
            "located_in": {
                "patterns": [
                    "在", "位于", "地址", "地址是", "地址为",
                    "坐落于", "坐落在", "坐落于"
                ],
                "confidence": 0.9,
                "bidirectional": False
            }
        }
        
        # 英文关系模式
        self.en_relation_patterns = {
            "has_color": {
                "patterns": ["is", "color", "colored"],
                "confidence": 0.9,
                "bidirectional": False
            },
            "has_size": {
                "patterns": ["is", "size", "sized"],
                "confidence": 0.9,
                "bidirectional": False
            },
            "spatial": {
                "patterns": [
                    "in", "at", "on", "near",
                    "inside", "outside", "beside", "next to"
                ],
                "confidence": 0.8,
                "bidirectional": True
            }
        }
        
    def _get_entity_pairs(self, doc: Doc, entities: List[Entity]) -> List[Tuple[Entity, Entity]]:
        """获取可能存在关系的实体对。"""
        pairs = []
        
        # 预处理：确保实体的开始和结束位置在文档范围内
        valid_entities = []
        for entity in entities:
            if 0 <= entity.start < len(doc) and 0 <= entity.end <= len(doc):
                valid_entities.append(entity)
        
        # 获取所有句子的范围
        sentences = list(doc.sents)
        sent_ranges = [(s.start, s.end) for s in sentences]
        
        # 按句子组织实体
        sentence_entities = {}
        for entity in valid_entities:
            for i, (start, end) in enumerate(sent_ranges):
                if start <= entity.start and entity.end <= end:
                    if i not in sentence_entities:
                        sentence_entities[i] = []
                    sentence_entities[i].append(entity)
                    break
        
        # 在每个句子内生成实体对
        for sent_idx, sent_entities in sentence_entities.items():
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities:
                    if e1 == e2:
                        continue
                        
                    # 检查实体类型组合
                    is_valid_pair = False
                    
                    # 产品属性关系
                    if e1.label == "PRODUCT" and e2.label in ["COLOR", "SIZE", "PRICE"]:
                        is_valid_pair = True
                    # 产品之间的空间关系
                    elif e1.label == e2.label == "PRODUCT":
                        is_valid_pair = True
                    # 组织机构和地址关系
                    elif e1.label == "ORG" and e2.label == "ADDRESS":
                        is_valid_pair = True
                    # 事件和时间关系
                    elif e1.label == "EVENT" and e2.label == "TIME":
                        is_valid_pair = True
                    
                    if is_valid_pair:
                        # 检查是否已经有相同的实体对
                        pair_exists = False
                        for p1, p2 in pairs:
                            if (p1 == e1 and p2 == e2) or (p1 == e2 and p2 == e1):
                                pair_exists = True
                                break
                        
                        if not pair_exists:
                            pairs.append((e1, e2))
        
        return pairs
        
        return pairs
    
    def _extract_relation_type(self, doc: Doc, subject: Entity, object: Entity) -> Tuple[str, float]:
        """提取两个实体之间的关系类型及其置信度分数。"""
        # 检测语言
        is_chinese = any(ord(c) > 127 for c in doc.text)
        relation_patterns = self.zh_relation_patterns if is_chinese else self.en_relation_patterns
        
        # 获取实体间的文本
        start = min(subject.start, object.start)
        end = max(subject.end, object.end)
        between_text = doc.text[start:end]
        
        # 根据实体类型确定可能的关系类型
        possible_relations = []
        
        # 产品属性关系
        if subject.label == "PRODUCT":
            if object.label == "COLOR":
                possible_relations.append(("has_color", relation_patterns["has_color"]))
            elif object.label == "SIZE":
                possible_relations.append(("has_size", relation_patterns["has_size"]))
            elif object.label == "PRICE":
                possible_relations.append(("has_price", relation_patterns["has_price"]))
        
        # 空间关系
        if subject.label == object.label == "PRODUCT":
            possible_relations.append(("spatial", relation_patterns["spatial"]))
        
        # 位置关系
        if subject.label == "ORG" and object.label == "ADDRESS":
            possible_relations.append(("located_in", relation_patterns["located_in"]))
        
        # 时间关系
        if subject.label == "EVENT" and object.label == "TIME":
            possible_relations.append(("temporal", relation_patterns["temporal"]))
        
        max_confidence = 0.0
        best_relation = None
        
        # 检查每个可能的关系
        for rel_type, rel_info in possible_relations:
            # 检查实体间的文本是否包含模式
            for pattern in rel_info["patterns"]:
                if pattern in doc.text:
                    confidence = rel_info["confidence"]
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_relation = rel_type
                    break
        
        # 如果没有找到关系，但实体类型匹配，给予默认关系
        if best_relation is None:
            if subject.label == "PRODUCT":
                if object.label == "COLOR":
                    best_relation = "has_color"
                    max_confidence = 0.7
                elif object.label == "SIZE":
                    best_relation = "has_size"
                    max_confidence = 0.7
                elif object.label == "PRICE":
                    best_relation = "has_price"
                    max_confidence = 0.7
            elif subject.label == object.label == "PRODUCT":
                best_relation = "spatial"
                max_confidence = 0.6
            elif subject.label == "ORG" and object.label == "ADDRESS":
                best_relation = "located_in"
                max_confidence = 0.7
            elif subject.label == "EVENT" and object.label == "TIME":
                best_relation = "temporal"
                max_confidence = 0.7
        
        return best_relation, max_confidence
        if not best_relation:
            if subject.label == "PRODUCT" and object.label == "COLOR":
                best_relation = "has_color"
                max_confidence = 0.7
            elif subject.label == "PRODUCT" and object.label == "SIZE":
                best_relation = "has_size"
                max_confidence = 0.7
            elif subject.label == "PRODUCT" and object.label == "PRICE":
                best_relation = "has_price"
                max_confidence = 0.7
            elif subject.label == "ORG" and object.label == "ADDRESS":
                best_relation = "located_in"
                max_confidence = 0.7
            elif subject.label == "EVENT" and object.label == "TIME":
                best_relation = "temporal"
                max_confidence = 0.7
            elif subject.label == object.label == "PRODUCT":
                best_relation = "spatial"
                max_confidence = 0.7
        
        return best_relation, max_confidence

    
    def _calculate_relation_confidence(self, source: Entity, target: Entity, doc: Doc) -> float:
        """计算关系的置信度分数。
        
        考虑以下因素：
        1. 基础置信度
        2. 实体置信度
        3. 距离权重
        4. 语法依存关系
        5. 实体类型兼容性
        6. 上下文相关性
        7. 句子结构分析
        """
        # 1. 基础置信度
        base_confidence = 0.7
        
        # 2. 实体置信度
        entity_confidence = (source.confidence + target.confidence) / 2
        
        # 3. 距离权重
        distance = abs(source.start - target.start)
        # 使用更平滑的距离衰减函数
        distance_weight = 1.0 / (1.0 + 0.1 * distance)
        
        # 4. 语法依存分析
        syntactic_weight = self._calculate_syntactic_weight(doc, source, target)
        
        # 5. 实体类型兼容性
        type_compatibility = self._evaluate_type_compatibility(source.label, target.label)
        
        # 6. 上下文相关性
        context_weight = self._calculate_context_weight(doc, source, target)
        
        # 7. 句子结构分析
        structure_weight = self._calculate_structure_weight(doc, source, target)
        
        # 计算加权平均
        weights = {
            'base': (base_confidence, 0.15),
            'entity': (entity_confidence, 0.20),
            'distance': (distance_weight, 0.15),
            'syntactic': (syntactic_weight, 0.15),
            'type': (type_compatibility, 0.15),
            'context': (context_weight, 0.10),
            'structure': (structure_weight, 0.10)
        }
        
        confidence = sum(value * weight for value, weight in weights.values())
        
        # 应用sigmoid函数使置信度更平滑
        confidence = 1 / (1 + np.exp(-5 * (confidence - 0.5)))
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_syntactic_weight(self, doc: Doc, source: Entity, target: Entity) -> float:
        """计算语法依存关系权重"""
        weight = 1.0
        
        # 检查是否共享相同的头节点
        source_head = doc[source.start].head
        target_head = doc[target.start].head
        
        if source_head == target_head:
            weight *= 1.2
        
        # 检查依存路径长度
        path_length = self._calculate_dependency_path_length(doc, source, target)
        if path_length:
            weight *= 1.0 / (1.0 + 0.1 * path_length)
        
        return weight
    
    def _calculate_context_weight(self, doc: Doc, source: Entity, target: Entity) -> float:
        """计算上下文相关性权重"""
        # 设置上下文窗口大小
        window_size = 5
        
        # 获取两个实体的上下文窗口
        source_start = max(0, source.start - window_size)
        source_end = min(len(doc), source.end + window_size)
        target_start = max(0, target.start - window_size)
        target_end = min(len(doc), target.end + window_size)
        
        # 提取上下文中的关键词
        source_context = set(token.text for token in doc[source_start:source_end])
        target_context = set(token.text for token in doc[target_start:target_end])
        
        # 计算Jaccard相似度
        intersection = len(source_context & target_context)
        union = len(source_context | target_context)
        
        if union == 0:
            return 0.8
        
        similarity = intersection / union
        return 0.8 + (similarity * 0.2)
    
    def _calculate_structure_weight(self, doc: Doc, source: Entity, target: Entity) -> float:
        """计算句子结构权重"""
        weight = 1.0
        
        # 检查是否在同一个句子中
        if self._in_same_sentence(doc, source, target):
            weight *= 1.2
        
        # 检查是否有直接的语法关系
        source_token = doc[source.start]
        target_token = doc[target.start]
        
        if source_token.head == target_token or target_token.head == source_token:
            weight *= 1.3
        
        return weight
    
    def _calculate_dependency_path_length(self, doc: Doc, source: Entity, target: Entity) -> int:
        """计算两个实体之间的依存路径长度"""
        # 获取实体的根节点
        source_token = doc[source.start]
        target_token = doc[target.start]
        
        # 如果实体直接相连
        if source_token.head == target_token or target_token.head == source_token:
            return 1
        
        # 获取从源实体到根节点的路径
        source_path = []
        current = source_token
        while current.head != current:
            source_path.append(current)
            current = current.head
        source_path.append(current)
        
        # 获取从目标实体到根节点的路径
        target_path = []
        current = target_token
        while current.head != current:
            target_path.append(current)
            current = current.head
        target_path.append(current)
        
        # 找到最近的公共祖先
        common_ancestor = None
        for token in source_path:
            if token in target_path:
                common_ancestor = token
                break
        
        if not common_ancestor:
            return None
        
        # 计算路径长度
        source_distance = source_path.index(common_ancestor)
        target_distance = target_path.index(common_ancestor)
        
        return source_distance + target_distance
    
    def _evaluate_type_compatibility(self, source_type: str, target_type: str) -> float:
        """评估两个实体类型之间的兼容性。"""
        # 定义高兼容性的实体类型对
        high_compatibility = {
            ("PRODUCT", "COLOR"): 0.9,
            ("PRODUCT", "SIZE"): 0.9,
            ("PRODUCT", "PRICE"): 0.9,
            ("PRODUCT", "MATERIAL"): 0.9,
            ("ORG", "ADDRESS"): 0.8,
            ("PERSON", "ORG"): 0.8,
            ("EVENT", "TIME"): 0.9,
            ("EVENT", "ADDRESS"): 0.8
        }
        
        # 检查是否存在预定义的兼容性分数
        type_pair = (source_type, target_type)
        reverse_pair = (target_type, source_type)
        
        if type_pair in high_compatibility:
            return high_compatibility[type_pair]
        elif reverse_pair in high_compatibility:
            return high_compatibility[reverse_pair]
        
        # 默认兼容性分数
        return 0.5
    
    def extract_relations(self, doc: Doc, entities: List[Entity]) -> List[Relation]:
        """提取实体之间的双向关系。"""
        relations = []
        pairs = self._get_entity_pairs(doc, entities)
        
        for subject, object in pairs:
            # 检查是否在同一个句子中
            if not self._in_same_sentence(doc, subject, object):
                continue
            
            # 正向关系
            relation_type, confidence = self._extract_relation_type(doc, subject, object)
            
            if relation_type and confidence > 0:
                # 计算最终置信度
                final_confidence = confidence * self._calculate_relation_confidence(subject, object, doc)
                
                # 创建关系对象
                relation = Relation(
                    subject=subject,
                    predicate=relation_type,
                    object=object,
                    confidence=final_confidence,
                    metadata={"direction": "forward"}
                )
                relations.append(relation)
                
                # 对于空间关系，添加反向关系
                if relation_type == "spatial":
                    reverse_relation = Relation(
                        subject=object,
                        predicate=relation_type,
                        object=subject,
                        confidence=final_confidence,
                        metadata={"direction": "reverse"}
                    )
                    relations.append(reverse_relation)
                
                # 如果是双向关系，尝试提取反向关系
                if relation_type in self.zh_relation_patterns and self.zh_relation_patterns[relation_type].get("bidirectional", False):
                    reverse_type, reverse_confidence = self._extract_relation_type(doc, object, subject)
                    if reverse_type and reverse_confidence > 0:
                        reverse_final_confidence = reverse_confidence * self._calculate_relation_confidence(object, subject, doc)
                        reverse_relation = Relation(
                            subject=object,
                            predicate=reverse_type,
                            object=subject,
                            confidence=reverse_final_confidence,
                            metadata={"direction": "reverse"}
                        )
                        relations.append(reverse_relation)
        
        return relations

    
    def _in_same_sentence(self, doc: Doc, entity1: Entity, entity2: Entity) -> bool:
        """检查两个实体是否在同一个句子中。"""
        for sent in doc.sents:
            if (entity1.start >= sent.start and entity1.end <= sent.end and
                entity2.start >= sent.start and entity2.end <= sent.end):
                return True
        return False
    
    def _calculate_syntactic_distance(self, doc: Doc, entity1: Entity, entity2: Entity) -> int:
        """计算两个实体之间的语法距离。"""
        # 获取实体的根节点
        root1 = doc[entity1.start].head
        root2 = doc[entity2.start].head
        
        # 如果有直接的依存关系
        if root1 == root2:
            return 1
        
        # 计算依存树中的距离
        distance = 0
        current = doc[entity1.start]
        while current != root1:
            current = current.head
            distance += 1
        
        current = doc[entity2.start]
        while current != root2:
            current = current.head
            distance += 1
        
        return distance
