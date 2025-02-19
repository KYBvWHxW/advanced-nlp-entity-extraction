"""
Tests for HotpotQA task.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tasks.hotpot_qa import HotpotQA
from src.model import ReActModel
from src.tools.wikipedia_tool import WikipediaTool

class TestHotpotQA(unittest.TestCase):
    """Test cases for HotpotQA class."""
    
    def setUp(self):
        """Set up test cases."""
        self.mock_model = MagicMock(spec=ReActModel)
        self.mock_wiki_tool = MagicMock(spec=WikipediaTool)
        self.task = HotpotQA(self.mock_model, self.mock_wiki_tool)
        
    def test_format_prompt(self):
        """Test prompt formatting."""
        question = "What is the capital of France?"
        context = [
            {"title": "France", "content": "France is a country in Europe."}
        ]
        
        prompt = self.task.format_prompt(question, context)
        
        self.assertIn(question, prompt)
        self.assertIn("France", prompt)
        self.assertIn("France is a country in Europe.", prompt)
        
    def test_parse_model_output_with_action(self):
        """Test parsing model output with action."""
        output = {
            "thought": "I should search Wikipedia",
            "action": "Search[France]"
        }
        
        thought, action = self.task.parse_model_output(output)
        
        self.assertEqual(thought, "I should search Wikipedia")
        self.assertEqual(action, "Search[France]")
        
    def test_parse_model_output_without_action(self):
        """Test parsing model output without action."""
        output = {
            "thought": "The capital of France is Paris.",
            "action": ""
        }
        
        thought, action = self.task.parse_model_output(output)
        
        self.assertEqual(thought, "The capital of France is Paris.")
        self.assertIsNone(action)
        
    def test_execute_search_action(self):
        """Test executing search action."""
        action = "Search[France]"
        mock_results = [{"title": "France", "summary": "A country in Europe"}]
        self.mock_wiki_tool.search.return_value = mock_results
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "search")
        self.assertEqual(result["query"], "France")
        self.assertEqual(result["results"], mock_results)
        self.mock_wiki_tool.search.assert_called_once_with("France")
        
    def test_execute_lookup_action(self):
        """Test executing lookup action."""
        action = "Lookup[France]"
        mock_content = {
            "title": "France",
            "content": "France is a country in Europe",
            "url": "http://example.com"
        }
        self.mock_wiki_tool.get_page_content.return_value = mock_content
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "lookup")
        self.assertEqual(result["title"], "France")
        self.assertEqual(result["content"], mock_content)
        self.mock_wiki_tool.get_page_content.assert_called_once_with("France")
        
    def test_execute_invalid_action(self):
        """Test executing invalid action."""
        action = "InvalidAction[test]"
        
        result = self.task.execute_action(action)
        
        self.assertEqual(result["type"], "error")
        self.assertIn("Unknown action format", result["message"])
        
    def test_solve_with_immediate_answer(self):
        """Test solving question with immediate answer."""
        question = "What is the capital of France?"
        self.mock_model.react_step.return_value = {
            "thought": "The capital of France is Paris.",
            "action": ""
        }
        
        result = self.task.solve(question)
        
        self.assertEqual(result["question"], question)
        self.assertEqual(len(result["thoughts"]), 1)
        self.assertEqual(len(result["actions"]), 0)
        self.assertEqual(result["answer"], "The capital of France is Paris.")
        
    def test_solve_with_multiple_steps(self):
        """Test solving question with multiple steps."""
        question = "What is the capital of France?"
        
        # Mock model to perform search then answer
        self.mock_model.react_step.side_effect = [
            {
                "thought": "I should search for France",
                "action": "Search[France]"
            },
            {
                "thought": "The capital of France is Paris.",
                "action": ""
            }
        ]
        
        # Mock wiki tool search results
        self.mock_wiki_tool.search.return_value = [
            {"title": "France", "summary": "Capital: Paris"}
        ]
        
        result = self.task.solve(question)
        
        self.assertEqual(result["question"], question)
        self.assertEqual(len(result["thoughts"]), 2)
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["answer"], "The capital of France is Paris.")

if __name__ == '__main__':
    unittest.main()
