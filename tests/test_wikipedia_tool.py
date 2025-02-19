"""
Tests for Wikipedia tool.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.wikipedia_tool import WikipediaTool

class TestWikipediaTool(unittest.TestCase):
    """Test cases for WikipediaTool class."""
    
    def setUp(self):
        """Set up test cases."""
        self.mock_wiki = MagicMock()
        with patch('wikipediaapi.Wikipedia', return_value=self.mock_wiki):
            self.tool = WikipediaTool()
        
    def test_search_existing_page(self):
        """Test searching for an existing page."""
        # Mock page
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Test Page"
        mock_page.summary = "Test summary"
        
        self.mock_wiki.page.return_value = mock_page
        
        # Perform search
        results = self.tool.search("test query")
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Test Page")
        self.assertTrue(results[0]["summary"].startswith("Test summary"))
        
    def test_search_nonexistent_page(self):
        """Test searching for a non-existent page."""
        # Mock page
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        
        self.mock_wiki.page.return_value = mock_page
        
        # Perform search
        results = self.tool.search("nonexistent")
        
        # Verify results
        self.assertEqual(len(results), 0)
        
    def test_get_page_content_existing(self):
        """Test getting content of an existing page."""
        # Mock page
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.title = "Test Page"
        mock_page.text = "Page content"
        mock_page.fullurl = "http://test.url"
        
        self.mock_wiki.page.return_value = mock_page
        
        # Get page content
        content = self.tool.get_page_content("Test Page")
        
        # Verify content
        self.assertIsNotNone(content)
        self.assertEqual(content["title"], "Test Page")
        self.assertEqual(content["content"], "Page content")
        self.assertEqual(content["url"], "http://test.url")
        
    def test_get_page_content_nonexistent(self):
        """Test getting content of a non-existent page."""
        # Mock page
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        
        self.mock_wiki.page.return_value = mock_page
        
        # Get page content
        content = self.tool.get_page_content("Nonexistent")
        
        # Verify content
        self.assertIsNone(content)

if __name__ == '__main__':
    unittest.main()
