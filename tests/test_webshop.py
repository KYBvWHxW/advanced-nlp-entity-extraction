"""
Tests for WebShop task.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tasks.webshop import WebShop, ActionType, ShopState, Product
from src.model import ReActModel

class TestWebShop(unittest.TestCase):
    """Test cases for WebShop class."""
    
    def setUp(self):
        """Set up test cases."""
        self.mock_model = MagicMock(spec=ReActModel)
        
        # Create temporary product database
        self.temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False)
        json.dump({
            "products": [
                {
                    "id": "1",
                    "name": "Blue Cotton T-Shirt",
                    "category": "PRODUCT_CLOTHING",
                    "price": "19.99",
                    "attributes": {
                        "color": "blue",
                        "size": "M",
                        "brand": "TestBrand"
                    },
                    "description": "Comfortable cotton t-shirt",
                    "rating": "4.5",
                    "reviews": [
                        {"text": "Great shirt!", "rating": 5}
                    ]
                },
                {
                    "id": "2",
                    "name": "Black Jeans",
                    "category": "PRODUCT_CLOTHING",
                    "price": "49.99",
                    "attributes": {
                        "color": "black",
                        "size": "32",
                        "brand": "TestBrand"
                    },
                    "description": "Classic black jeans",
                    "rating": "4.0",
                    "reviews": [
                        {"text": "Good fit", "rating": 4}
                    ]
                }
            ]
        }, self.temp_db)
        self.temp_db.close()
        
        self.shop = WebShop(self.mock_model, self.temp_db.name)
        
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_db.name)
        
    def test_format_prompt(self):
        """Test prompt formatting."""
        state = ShopState(
            current_page="search_results",
            search_results=[self.shop.products[0]],
            cart_items=[],
            filters_applied={"color": "blue"},
            sort_by="price_low",
            total_cost=0.0,
            last_action="search(query='t-shirt')"
        )
        
        prompt = self.shop.format_prompt("Buy a blue t-shirt", state)
        
        self.assertIn("Buy a blue t-shirt", prompt)
        self.assertIn("Blue Cotton T-Shirt", prompt)
        self.assertIn("color: blue", prompt)
        self.assertIn("search(query='t-shirt')", prompt)
        
    def test_search_products(self):
        """Test product search functionality."""
        results = self.shop._search_products("blue t-shirt")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Blue Cotton T-Shirt")
        
    def test_apply_filters(self):
        """Test filter application."""
        filters = {
            "price_max": "30.00",
            "brand": "TestBrand"
        }
        
        filtered = self.shop._apply_filters(self.shop.products, filters)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "Blue Cotton T-Shirt")
        
    def test_sort_products(self):
        """Test product sorting."""
        # Test price low to high
        sorted_low = self.shop._sort_products(self.shop.products, "price_low")
        self.assertEqual(sorted_low[0].name, "Blue Cotton T-Shirt")
        
        # Test price high to low
        sorted_high = self.shop._sort_products(self.shop.products, "price_high")
        self.assertEqual(sorted_high[0].name, "Black Jeans")
        
    def test_execute_action_search(self):
        """Test search action execution."""
        self.shop.state = ShopState(
            current_page="home",
            search_results=[],
            cart_items=[],
            filters_applied={},
            sort_by=None,
            total_cost=0.0
        )
        
        new_state = self.shop.execute_action(
            ActionType.SEARCH,
            {"query": "blue t-shirt"}
        )
        
        self.assertEqual(len(new_state.search_results), 1)
        self.assertEqual(new_state.current_page, "search_results")
        
    def test_execute_action_add_to_cart(self):
        """Test add to cart action execution."""
        self.shop.state = ShopState(
            current_page="product_1",
            search_results=[self.shop.products[0]],
            cart_items=[],
            filters_applied={},
            sort_by=None,
            total_cost=0.0
        )
        
        new_state = self.shop.execute_action(
            ActionType.ADD_TO_CART,
            {"product_id": "1"}
        )
        
        self.assertEqual(len(new_state.cart_items), 1)
        self.assertEqual(new_state.total_cost, 19.99)
        
    def test_solve_successful_purchase(self):
        """Test successful purchase flow."""
        task = "Buy a blue t-shirt"
        
        # Mock model to perform sequence of actions
        self.mock_model.react_step.side_effect = [
            {
                "thought": "I should search for blue t-shirts",
                "action": "search({\"query\": \"blue t-shirt\"})"
            },
            {
                "thought": "Found a good t-shirt, let's view it",
                "action": "view({\"product_id\": \"1\"})"
            },
            {
                "thought": "This t-shirt matches the requirements",
                "action": "add_to_cart({\"product_id\": \"1\"})"
            },
            {
                "thought": "Ready to complete the purchase",
                "action": "checkout({})"
            }
        ]
        
        result = self.shop.solve(task, max_steps=5)
        
        self.assertEqual(len(result["trajectory"]), 4)
        self.assertEqual(result["trajectory"][0]["action"], "search({\"query\": \"blue t-shirt\"})")
        self.assertEqual(result["trajectory"][1]["action"], "view({\"product_id\": \"1\"})")
        self.assertEqual(result["trajectory"][2]["action"], "add_to_cart({\"product_id\": \"1\"})")
        self.assertEqual(result["trajectory"][3]["action"], "checkout({})")
        self.assertEqual(result["final_state"].total_cost, 19.99)
        
    def test_solve_with_invalid_action(self):
        """Test solving with invalid action."""
        task = "Buy a blue t-shirt"
        
        # Mock model to generate invalid action
        self.mock_model.react_step.return_value = {
            "thought": "I should teleport",
            "action": "teleport({\"destination\": \"store\"})"
        }
        
        result = self.shop.solve(task, max_steps=1)
        
        self.assertFalse(result["success"])
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertIn("error", result["trajectory"][0])

if __name__ == '__main__':
    unittest.main()
