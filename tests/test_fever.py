"""
Tests for Fever task.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tasks.fever import Fever, VerificationResult
from src.model import ReActModel
from src.tools.wikipedia_tool import WikipediaTool

class TestFever(unittest.TestCase):
    """Test cases for Fever class."""
    
    def setUp(self):
        """Set up test cases."""
        self.mock_model = MagicMock(spec=ReActModel)
        self.mock_wiki_tool = MagicMock(spec=WikipediaTool)
        self.task = Fever(self.mock_model, self.mock_wiki_tool)
        
    def test_format_prompt(self):
        """Test prompt formatting."""
        claim = "Paris is the capital of France."
        evidence = [
            {"title": "Paris", "content": "Paris is the capital and largest city of France."}
        ]
        history = [
            {
                "thought": "I should verify this claim.",
                "action": "Search[Paris]",
                "observation": "Found information about Paris."
            }
        ]
        
        prompt = self.task.format_prompt(claim, evidence, history)
        
        self.assertIn(claim, prompt)
        self.assertIn("Paris is the capital and largest city of France.", prompt)
        self.assertIn("I should verify this claim.", prompt)
        self.assertIn("Search[Paris]", prompt)
        
    def test_extract_entities(self):
        """Test entity extraction from claim."""
        claim = "The Eiffel Tower is located in Paris, France."
        entities = self.task.extract_entities(claim)
        
        self.assertIn("Eiffel", entities)
        self.assertIn("Tower", entities)
        self.assertIn("Paris", entities)
        self.assertIn("France", entities)
        self.assertNotIn("is", entities)
        self.assertNotIn("in", entities)
        
    def test_execute_search_action(self):
        """Test executing search action."""
        action = "Search[Paris]"
        mock_results = [{"title": "Paris", "content": "Capital of France"}]
        self.mock_wiki_tool.search.return_value = mock_results
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "search")
        self.assertEqual(result["query"], "Paris")
        self.assertEqual(result["results"], mock_results)
        
    def test_execute_compare_action(self):
        """Test executing compare action."""
        action = "Compare[Paris is in France|Paris is the capital of France]"
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "compare")
        self.assertEqual(result["claim"], "Paris is in France")
        self.assertEqual(result["evidence"], "Paris is the capital of France")
        
    def test_execute_invalid_action(self):
        """Test executing invalid action."""
        action = "InvalidAction[test]"
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "error")
        self.assertIn("Unknown action format", result["message"])
        
    def test_determine_verdict_supports(self):
        """Test verdict determination for supporting evidence."""
        thought = "The evidence clearly supports the claim."
        verdict = self.task.determine_verdict(thought)
        self.assertEqual(verdict, VerificationResult.SUPPORTS)
        
    def test_determine_verdict_refutes(self):
        """Test verdict determination for refuting evidence."""
        thought = "The evidence contradicts the claim."
        verdict = self.task.determine_verdict(thought)
        self.assertEqual(verdict, VerificationResult.REFUTES)
        
    def test_determine_verdict_not_enough_info(self):
        """Test verdict determination for insufficient evidence."""
        thought = "There isn't enough information to verify the claim."
        verdict = self.task.determine_verdict(thought)
        self.assertEqual(verdict, VerificationResult.NOT_ENOUGH_INFO)
        
    def test_verify_with_supporting_evidence(self):
        """Test claim verification with supporting evidence."""
        claim = "Paris is the capital of France."
        
        # Mock search results
        self.mock_wiki_tool.search.return_value = [
            {"title": "Paris", "content": "Paris is the capital of France."}
        ]
        
        # Mock model to perform search then conclude
        self.mock_model.react_step.side_effect = [
            {
                "thought": "I should search for information about Paris.",
                "action": "Search[Paris]"
            },
            {
                "thought": "The evidence supports the claim that Paris is the capital of France.",
                "action": ""
            }
        ]
        
        result = self.task.verify(claim)
        
        self.assertEqual(result["claim"], claim)
        self.assertEqual(result["verdict"], VerificationResult.SUPPORTS.value)
        self.assertEqual(result["confidence"], "HIGH")
        self.assertEqual(len(result["reasoning_chain"]), 1)
        
    def test_verify_with_no_evidence(self):
        """Test claim verification with no evidence."""
        claim = "XYZ is the capital of ABC."
        
        # Mock empty search results
        self.mock_wiki_tool.search.return_value = []
        
        # Mock model to conclude without evidence
        self.mock_model.react_step.return_value = {
            "thought": "There isn't enough information to verify this claim.",
            "action": ""
        }
        
        result = self.task.verify(claim)
        
        self.assertEqual(result["claim"], claim)
        self.assertEqual(result["verdict"], VerificationResult.NOT_ENOUGH_INFO.value)
        self.assertEqual(result["confidence"], "LOW")
        self.assertEqual(len(result["evidence"]), 0)

if __name__ == '__main__':
    unittest.main()
