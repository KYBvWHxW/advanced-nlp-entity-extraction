"""
WebShop task implementation using ReAct approach.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import json
from enum import Enum

from ..model import ReActModel
from ..tools.entity_extractor import EntityExtractor

class ActionType(Enum):
    """Types of actions available in WebShop."""
    SEARCH = "search"
    FILTER = "filter"
    SORT = "sort"
    VIEW = "view"
    COMPARE = "compare"
    ADD_TO_CART = "add_to_cart"
    CHECKOUT = "checkout"

@dataclass
class Product:
    """Represents a product in the WebShop."""
    id: str
    name: str
    category: str
    price: float
    attributes: Dict[str, str]
    description: str
    rating: float
    reviews: List[Dict]

@dataclass
class ShopState:
    """Represents the current state of the WebShop session."""
    current_page: str
    search_results: List[Product]
    cart_items: List[Product]
    filters_applied: Dict[str, str]
    sort_by: Optional[str]
    total_cost: float
    last_action: Optional[str] = None

class WebShop:
    """
    Implementation of WebShop task using ReAct approach.
    """
    
    def __init__(self, model: ReActModel, product_db_path: str):
        """
        Initialize WebShop task.
        
        Args:
            model: ReAct model instance
            product_db_path: Path to product database JSON
        """
        self.model = model
        self.entity_extractor = EntityExtractor()
        self.products = self._load_products(product_db_path)
        self.state = None
        
    def _load_products(self, db_path: str) -> List[Product]:
        """Load product database."""
        with open(db_path, 'r') as f:
            data = json.load(f)
            
        return [
            Product(
                id=p['id'],
                name=p['name'],
                category=p['category'],
                price=float(p['price']),
                attributes=p['attributes'],
                description=p['description'],
                rating=float(p['rating']),
                reviews=p['reviews']
            )
            for p in data['products']
        ]
        
    def format_prompt(self, task: str, state: ShopState) -> str:
        """Format prompt for the model."""
        prompt = f"Task: {task}\n\n"
        prompt += "Current state:\n"
        prompt += f"Page: {state.current_page}\n"
        
        if state.search_results:
            prompt += "\nSearch results:\n"
            for i, product in enumerate(state.search_results[:5], 1):
                prompt += f"{i}. {product.name} (${product.price:.2f})\n"
                
        if state.filters_applied:
            prompt += "\nActive filters:\n"
            for k, v in state.filters_applied.items():
                prompt += f"- {k}: {v}\n"
                
        if state.cart_items:
            prompt += "\nCart:\n"
            for item in state.cart_items:
                prompt += f"- {item.name} (${item.price:.2f})\n"
            prompt += f"Total: ${state.total_cost:.2f}\n"
            
        if state.last_action:
            prompt += f"\nLast action: {state.last_action}\n"
            
        prompt += "\nAvailable actions:\n"
        for action in ActionType:
            prompt += f"- {action.value}\n"
            
        prompt += "\nWhat action should I take next? Think step-by-step:\n"
        return prompt
        
    def _search_products(self, query: str) -> List[Product]:
        """Search products based on query."""
        # Extract entities from query
        entities = self.entity_extractor.extract_entities(query)
        
        # Build search criteria
        criteria = {
            "category": next((e.label for e in entities if "PRODUCT_" in e.label), None),
            "color": next((e.text for e in entities if e.label == "COLOR"), None)
        }
        
        # Filter products
        results = []
        for product in self.products:
            matches = True
            
            # Check category
            if criteria["category"] and criteria["category"] != product.category:
                matches = False
                
            # Check color
            if criteria["color"] and (
                "color" not in product.attributes or 
                product.attributes["color"].lower() != criteria["color"].lower()
            ):
                matches = False
                
            # Check name match
            if not any(term.lower() in product.name.lower() for term in query.split()):
                matches = False
                
            if matches:
                results.append(product)
                
        return results
        
    def _apply_filters(self, products: List[Product], filters: Dict[str, str]) -> List[Product]:
        """Apply filters to product list."""
        filtered = products
        
        for key, value in filters.items():
            if key == "price_min":
                filtered = [p for p in filtered if p.price >= float(value)]
            elif key == "price_max":
                filtered = [p for p in filtered if p.price <= float(value)]
            elif key == "rating_min":
                filtered = [p for p in filtered if p.rating >= float(value)]
            elif key in ["category", "brand"]:
                filtered = [p for p in filtered if p.attributes.get(key, "").upper() == value.upper()]
                
        return filtered
        
    def _sort_products(self, products: List[Product], sort_by: str) -> List[Product]:
        """Sort products by specified criterion."""
        if sort_by == "price_low":
            return sorted(products, key=lambda p: p.price)
        elif sort_by == "price_high":
            return sorted(products, key=lambda p: -p.price)
        elif sort_by == "rating":
            return sorted(products, key=lambda p: -p.rating)
        return products
        
    def execute_action(self, action_type: ActionType, params: Dict) -> ShopState:
        """
        Execute an action in the WebShop environment.
        
        Args:
            action_type: Type of action to execute
            params: Action parameters
            
        Returns:
            New shop state
        """
        if action_type == ActionType.SEARCH:
            self.state.search_results = self._search_products(params["query"])
            self.state.current_page = "search_results"
            
        elif action_type == ActionType.FILTER:
            self.state.filters_applied.update(params["filters"])
            self.state.search_results = self._apply_filters(
                self.state.search_results,
                self.state.filters_applied
            )
            
        elif action_type == ActionType.SORT:
            self.state.sort_by = params["sort_by"]
            self.state.search_results = self._sort_products(
                self.state.search_results,
                self.state.sort_by
            )
            
        elif action_type == ActionType.VIEW:
            product_id = params["product_id"]
            product = next(p for p in self.products if p.id == product_id)
            self.state.current_page = f"product_{product_id}"
            
        elif action_type == ActionType.ADD_TO_CART:
            product_id = params["product_id"]
            product = next(p for p in self.products if p.id == product_id)
            self.state.cart_items.append(product)
            self.state.total_cost = sum(float(p.price) for p in self.state.cart_items)
            
        elif action_type == ActionType.CHECKOUT:
            if not self.state.cart_items:
                raise ValueError("Cannot checkout with empty cart")
            self.state.current_page = "checkout"
            
        self.state.last_action = f"{action_type.value}({json.dumps(params)})"
        return self.state
        
    def solve(self, task: str, max_steps: int = 50) -> Dict:
        """
        Solve a WebShop task using ReAct approach.
        
        Args:
            task: Shopping task description
            max_steps: Maximum number of steps to take
            
        Returns:
            Dictionary containing solution trajectory and results
        """
        # Initialize shop state
        self.state = ShopState(
            current_page="home",
            search_results=[],
            cart_items=[],
            filters_applied={},
            sort_by=None,
            total_cost=0.0
        )
        
        trajectory = []
        success = False
        
        try:
            for step in range(max_steps):
                # Get model output
                prompt = self.format_prompt(task, self.state)
                output = self.model.react_step(prompt)
                
                thought = output.get("thought", "").strip()
                action_str = output.get("action", "").strip()
                
                # Record step
                step_record = {
                    "state": self.state,
                    "thought": thought,
                    "action": action_str
                }
                
                # Parse and execute action
                try:
                    action_type = ActionType(action_str.split("(")[0])
                    params = json.loads(action_str[action_str.index("("):].replace("'", '"'))
                    
                    self.state = self.execute_action(action_type, params)
                    step_record["result"] = "success"
                    
                    # Check if task is complete
                    if (self.state.current_page == "checkout" and 
                        self.state.cart_items and
                        "buy" in task.lower()):
                        success = True
                        break
                        
                except (ValueError, json.JSONDecodeError) as e:
                    step_record["error"] = str(e)
                    
                trajectory.append(step_record)
                
        except StopIteration:
            # Handle case where mock runs out of responses
            pass
            
        return {
            "task": task,
            "success": success,
            "trajectory": trajectory,
            "final_state": self.state,
            "steps_taken": len(trajectory)
        }
