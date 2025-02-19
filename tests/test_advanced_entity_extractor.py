"""
Tests for the advanced entity extractor.
"""

import unittest
from datetime import datetime
from src.tools.advanced_entity_extractor import (
    AdvancedEntityExtractor,
    Entity,
    Relation,
    AttributeNormalizer
)

class TestAdvancedEntityExtractor(unittest.TestCase):
    """Test cases for AdvancedEntityExtractor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test cases."""
        cls.extractor = AdvancedEntityExtractor()
        cls.normalizer = AttributeNormalizer()
        
    def test_entity_extraction(self):
        """Test basic entity extraction."""
        text = "I want to buy a blue cotton t-shirt for $29.99"
        results = self.extractor.extract_entities_and_relations(text)
        
        entities = results["entities"]
        self.assertTrue(any(e.label == "PRODUCT_CLOTHING" for e in entities))
        self.assertTrue(any(e.label == "COLOR" for e in entities))
        self.assertTrue(any(e.label == "PRICE" for e in entities))
        
    def test_relation_extraction(self):
        """Test relation extraction between entities."""
        text = "The blue shirt costs $29.99"
        results = self.extractor.extract_entities_and_relations(text)
        
        relations = results["relations"]
        
        # Test basic relation
        self.assertTrue(any(
            r.subject.label == "PRODUCT_CLOTHING" and
            r.object.label == "PRICE" and
            r.predicate == "has_price"
            for r in relations
        ))
        
        # Test confidence score
        price_relations = [r for r in relations if r.object.label == "PRICE"]
        self.assertTrue(all(0 <= r.confidence <= 1.0 for r in price_relations))
        
        # Test metadata
        self.assertTrue(all(
            "distance" in r.metadata and
            "text_between" in r.metadata and
            "direction" in r.metadata
            for r in relations
        ))
        
    def test_attribute_normalization(self):
        """Test attribute normalization."""
        # Test color normalization
        self.assertEqual(
            self.normalizer.normalize_color("navy"),
            "blue"
        )
        
        # Test size normalization
        self.assertEqual(
            self.normalizer.normalize_size("extra large"),
            "XL"
        )
        
        # Test price normalization
        self.assertEqual(
            self.normalizer.normalize_price("$29.99"),
            29.99
        )
        
    def test_complex_entity_extraction(self):
        """Test extraction with complex relationships."""
        text = """
        I'm looking for a navy blue cotton t-shirt, size medium,
        that costs less than $30. It should be similar to the red shirt
        I saw yesterday for $25.99.
        """
        results = self.extractor.extract_entities_and_relations(text)
        
        entities = results["entities"]
        relations = results["relations"]
        
        # Check entity extraction
        self.assertTrue(any(
            e.label == "PRODUCT_CLOTHING" and "shirt" in e.text.lower()
            for e in entities
        ))
        
        # Check color normalization
        color_entities = [e for e in entities if e.label == "COLOR"]
        self.assertTrue(any(
            e.normalized_value == "blue" and e.text.lower() == "navy blue"
            for e in color_entities
        ))
        
        # Check semantic relations
        self.assertTrue(any(
            r.subject.label == "PRODUCT_CLOTHING" and
            r.object.label == "COLOR" and
            r.predicate == "has_color"
            for r in relations
        ))
        
        # Check comparison relations
        self.assertTrue(any(
            r.predicate == "similar_to" and
            "shirt" in r.subject.text.lower() and
            "shirt" in r.object.text.lower()
            for r in relations
        ))
        
        # Check confidence scores
        self.assertTrue(all(0 <= r.confidence <= 1.0 for r in relations))
        
        # Check bidirectional relations
        self.assertTrue(any(
            r.metadata.get("direction") == "forward" for r in relations
        ))
        self.assertTrue(any(
            r.metadata.get("direction") == "reverse" for r in relations
        ))
        
    def test_entity_deduplication(self):
        """Test entity deduplication."""
        text = "The blue t-shirt is a cotton blue shirt"
        results = self.extractor.extract_entities_and_relations(text)
        
        color_entities = [e for e in results["entities"] if e.label == "COLOR"]
        # Should only have one "blue" color entity
        self.assertEqual(
            len([e for e in color_entities if e.normalized_value == "blue"]),
            1
        )
        
    def test_distribution_analysis(self):
        """Test entity distribution analysis."""
        text = """
        Looking for a blue shirt and a red shirt.
        Also interested in a green jacket and black pants.
        """
        results = self.extractor.extract_entities_and_relations(text)
        
        distribution = self.extractor.analyze_entity_distribution(results["entities"])
        
        self.assertTrue("label_counts" in distribution)
        self.assertTrue("normalized_counts" in distribution)
        self.assertTrue(distribution["label_counts"]["COLOR"] >= 4)
        
    def test_date_normalization(self):
        """Test date normalization."""
        dates = [
            "2025-02-19",
            "2025-02-19T16:51:14",
        ]
        
        for date_str in dates:
            normalized = self.normalizer.normalize_date(date_str)
            self.assertIsNotNone(normalized)
            self.assertIsInstance(normalized, datetime)
    
    def test_semantic_relations(self):
        """Test semantic relation types."""
        # Test composition relations
        text = "The shirt is made of cotton and contains polyester fibers"
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        self.assertTrue(any(
            r.predicate == "composed_of" and
            r.confidence > 0.7
            for r in relations
        ))
        
        # Test spatial relations
        text = "The blue shirt is near the black pants on the shelf"
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        self.assertTrue(any(
            r.predicate in ["near", "on_top_of"] and
            r.confidence > 0.6
            for r in relations
        ))
        
        # Test temporal relations
        text = "I bought the red shirt before getting the blue one"
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        self.assertTrue(any(
            r.predicate == "before" and
            r.confidence > 0.6
            for r in relations
        ))
    
    def test_relation_confidence(self):
        """Test relation confidence scoring."""
        text = "The cotton shirt contains polyester. The blue pants are similar to the black jeans."
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        # Test composition relation confidence
        composition_relations = [r for r in relations if r.predicate == "composed_of"]
        self.assertTrue(all(r.confidence > 0.7 for r in composition_relations))
        
        # Test comparison relation confidence
        comparison_relations = [r for r in relations if r.predicate == "similar_to"]
        self.assertTrue(all(r.confidence > 0.6 for r in comparison_relations))
        
        # Test confidence components
        for relation in relations:
            self.assertIn("distance", relation.metadata)
            self.assertIn("text_between", relation.metadata)
            self.assertIn("direction", relation.metadata)
    
    def test_bidirectional_relations(self):
        """Test bidirectional relation extraction."""
        text = "The blue shirt matches the black pants. The cotton shirt contains polyester."
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        # Get all unique entity pairs
        entity_pairs = set()
        for r in relations:
            pair = tuple(sorted([id(r.subject), id(r.object)]))
            entity_pairs.add(pair)
        
        # Check for bidirectional relations
        for pair in entity_pairs:
            pair_relations = [r for r in relations if 
                            tuple(sorted([id(r.subject), id(r.object)])) == pair]
            
            # Each pair should have at least one relation
            self.assertGreater(len(pair_relations), 0)
            
            # If multiple relations exist for a pair, they should have different directions
            if len(pair_relations) > 1:
                directions = {r.metadata["direction"] for r in pair_relations}
                self.assertEqual(len(directions), len(pair_relations))
    
    def test_relation_deduplication(self):
        """Test relation deduplication."""
        text = "The blue cotton shirt contains polyester. The shirt is made of cotton and polyester."
        results = self.extractor.extract_entities_and_relations(text)
        relations = results["relations"]
        
        # Get composition relations
        composition_relations = [r for r in relations if r.predicate in ["composed_of", "contains"]]
        
        # Check for duplicate relations
        relation_pairs = set()
        for relation in composition_relations:
            pair = (id(relation.subject), id(relation.object))
            self.assertNotIn(pair, relation_pairs, "Duplicate relation found")
            relation_pairs.add(pair)
            
if __name__ == '__main__':
    unittest.main()
