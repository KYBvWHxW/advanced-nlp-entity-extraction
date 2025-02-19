"""
Advanced entity extraction tool using spaCy and custom rules.
"""

import spacy
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import re

@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    attributes: Dict[str, str] = None

class EntityExtractor:
    """Advanced entity extraction using spaCy and custom rules."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: Name of spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        self.custom_patterns = self._load_custom_patterns()
        
    def _load_custom_patterns(self) -> Dict[str, List[Dict]]:
        """Load custom entity extraction patterns."""
        return {
            "product": [
                {"pattern": [{"LOWER": {"REGEX": ".*(?:shirt|t-shirt|tshirt).*"}}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": {"REGEX": ".*(?:jeans|pants|trousers).*"}}], "label": "PRODUCT_CLOTHING"},
                {"pattern": [{"LOWER": {"REGEX": ".*(?:phone|headphones|electronics).*"}}], "label": "PRODUCT_ELECTRONICS"},
                {"pattern": [{"LOWER": {"REGEX": ".*(?:book|guide|manual).*"}}], "label": "PRODUCT_MEDIA"}
            ],
            "attribute": [
                {"pattern": [{"LOWER": "blue"}], "label": "COLOR"},
                {"pattern": [{"LOWER": "black"}], "label": "COLOR"},
                {"pattern": [{"LOWER": {"REGEX": ".*(?:color|size|brand).*"}}], "label": "PRODUCT_ATTRIBUTE"},
                {"pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["dollars", "$", "usd"]}}], "label": "PRICE"}
            ],
            "action": [
                {"pattern": [{"LEMMA": {"IN": ["search", "find", "look", "browse"]}}], "label": "ACTION_SEARCH"},
                {"pattern": [{"LEMMA": {"IN": ["buy", "purchase", "order", "get"]}}], "label": "ACTION_PURCHASE"},
                {"pattern": [{"LEMMA": {"IN": ["compare", "contrast", "versus"]}}], "label": "ACTION_COMPARE"}
            ]
        }
        
    def add_custom_pattern(self, category: str, pattern: Dict, label: str):
        """
        Add a custom entity extraction pattern.
        
        Args:
            category: Pattern category (product, attribute, action)
            pattern: spaCy pattern dict
            label: Entity label for the pattern
        """
        if category not in self.custom_patterns:
            self.custom_patterns[category] = []
        self.custom_patterns[category].append({"pattern": pattern, "label": label})
        
    def _apply_custom_patterns(self, doc) -> List[Entity]:
        """Apply custom patterns to extract entities."""
        entities = []
        
        for category, patterns in self.custom_patterns.items():
            for pattern in patterns:
                matcher = spacy.matcher.Matcher(self.nlp.vocab)
                matcher.add(pattern["label"], [pattern["pattern"]])
                matches = matcher(doc)
                
                for match_id, start, end in matches:
                    span = doc[start:end]
                    entities.append(Entity(
                        text=span.text,
                        label=pattern["label"],
                        start=start,
                        end=end,
                        confidence=0.9 if category == "product" else 0.8,
                        attributes={"category": category}
                    ))
                    
        return entities
        
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using both spaCy and custom patterns.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities
        """
        doc = self.nlp(text)
        
        # Get spaCy entities
        spacy_entities = [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start,
                end=ent.end,
                confidence=0.95,
                attributes={"source": "spacy"}
            )
            for ent in doc.ents
        ]
        
        # Get custom pattern entities
        custom_entities = self._apply_custom_patterns(doc)
        
        # Merge and deduplicate entities
        all_entities = spacy_entities + custom_entities
        unique_entities = self._deduplicate_entities(all_entities)
        
        return unique_entities
        
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping ones with higher confidence."""
        sorted_entities = sorted(entities, key=lambda x: (-x.confidence, -len(x.text)))
        final_entities = []
        used_spans = set()
        
        for entity in sorted_entities:
            span = (entity.start, entity.end)
            overlapping = False
            
            for used_span in used_spans:
                if (span[0] <= used_span[1] and span[1] >= used_span[0]):
                    overlapping = True
                    break
                    
            if not overlapping:
                final_entities.append(entity)
                used_spans.add(span)
                
        return sorted(final_entities, key=lambda x: x.start)
        
    def extract_relations(self, text: str) -> List[Dict]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            
        Returns:
            List of relation dictionaries
        """
        doc = self.nlp(text)
        entities = self.extract_entities(text)
        relations = []
        
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"]:
                head = token.head
                
                # Find entities containing the tokens
                subject = next((e for e in entities if e.start <= token.i <= e.end), None)
                object_ = next((e for e in entities if e.start <= head.i <= e.end), None)
                
                if subject and object_:
                    relations.append({
                        "subject": subject,
                        "predicate": head.lemma_,
                        "object": object_,
                        "confidence": 0.8
                    })
                    
        return relations
