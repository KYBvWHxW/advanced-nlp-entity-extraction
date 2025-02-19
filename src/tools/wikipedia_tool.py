"""
Wikipedia API tool for ReAct model.
"""

import json
from typing import Dict, List, Optional

import wikipediaapi

class WikipediaTool:
    """
    Tool for interacting with Wikipedia API.
    """
    
    def __init__(self, config_path: str = "config/model_config.json"):
        """
        Initialize Wikipedia tool.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            config = json.load(f)
            
        self.wiki = wikipediaapi.Wikipedia(
            config["wikipedia_api"]["user_agent"],
            config["wikipedia_api"]["language"]
        )
        
    def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Search Wikipedia for pages matching query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing page titles and summaries
        """
        # Note: wikipediaapi doesn't support direct search
        # This is a simplified implementation
        page = self.wiki.page(query)
        
        if not page.exists():
            return []
            
        return [{
            "title": str(page.title),
            "summary": str(page.summary[:500]) + "..."  # Truncate long summaries
        }]
        
    def get_page_content(self, title: str) -> Optional[Dict[str, str]]:
        """
        Get full content of a Wikipedia page.
        
        Args:
            title: Title of the Wikipedia page
            
        Returns:
            Dictionary containing page title and content, or None if page doesn't exist
        """
        page = self.wiki.page(title)
        
        if not page.exists():
            return None
            
        return {
            "title": str(page.title),
            "content": str(page.text),
            "url": str(page.fullurl)
        }
        
    def get_page_section(self, title: str, section: str) -> Optional[Dict[str, str]]:
        """
        Get content of a specific section from a Wikipedia page.
        
        Args:
            title: Title of the Wikipedia page
            section: Name of the section to retrieve
            
        Returns:
            Dictionary containing section title and content, or None if not found
        """
        page = self.wiki.page(title)
        
        if not page.exists():
            return None
            
        for s in page.sections:
            if str(s.title).lower() == section.lower():
                return {
                    "title": str(s.title),
                    "content": str(s.text)
                }
                
        return None
