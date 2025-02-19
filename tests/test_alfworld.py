"""
Tests for ALFWorld task.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tasks.alfworld import ALFWorld, ActionType, GameState
from src.model import ReActModel

@dataclass
class MockEnv:
    """Mock ALFWorld environment for testing."""
    def __init__(self):
        self.valid_actions = [
            "goto[kitchen]",
            "pickup[apple]",
            "put[apple|table]",
            "look[around]"
        ]
        self.state = "You are in the living room. There is an apple on the table."
        self.inventory = []
        self.done = False
        self.score = 0.0
        
    def reset(self):
        """Reset environment state."""
        self.state = "You are in the living room. There is an apple on the table."
        self.inventory = []
        self.done = False
        self.score = 0.0
        return self.state
        
    def step(self, action):
        """Take a step in the environment."""
        if action == "goto[kitchen]":
            self.state = "You are in the kitchen."
            self.score += 0.1
        elif action == "pickup[apple]":
            self.state = "You picked up the apple."
            self.inventory.append("apple")
            self.score += 0.2
        elif action == "put[apple|table]":
            if "apple" in self.inventory:
                self.state = "You put the apple on the table."
                self.inventory.remove("apple")
                self.score += 0.3
                self.done = True
            else:
                self.state = "You don't have an apple."
        elif action == "look[around]":
            self.state = "You see a table and some chairs."
            
        return (
            self.state,
            self.score,
            self.done,
            {
                "inventory": self.inventory,
                "valid_actions": self.valid_actions
            }
        )

class TestALFWorld(unittest.TestCase):
    """Test cases for ALFWorld class."""
    
    def setUp(self):
        """Set up test cases."""
        self.mock_model = MagicMock(spec=ReActModel)
        self.task = ALFWorld(self.mock_model)
        self.env = MockEnv()
        
    def test_format_prompt(self):
        """Test prompt formatting."""
        task = "Put the apple on the table."
        state = GameState(
            observation="You are in the living room.",
            inventory=["apple"],
            valid_actions={"goto[kitchen]", "put[apple|table]"},
            previous_action="pickup[apple]"
        )
        
        prompt = self.task.format_prompt(task, state)
        
        self.assertIn(task, prompt)
        self.assertIn("You are in the living room.", prompt)
        self.assertIn("apple", prompt)
        self.assertIn("goto[kitchen]", prompt)
        self.assertIn("put[apple|table]", prompt)
        self.assertIn("pickup[apple]", prompt)
        
    def test_parse_valid_action(self):
        """Test parsing valid action string."""
        action_str = "goto[kitchen]"
        
        action = self.task.parse_action(action_str)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], ActionType.GOTO)
        self.assertEqual(action["params"], ["kitchen"])
        
    def test_parse_invalid_action(self):
        """Test parsing invalid action string."""
        action_str = "invalid_action"
        
        action = self.task.parse_action(action_str)
        
        self.assertIsNone(action)
        
    def test_validate_valid_action(self):
        """Test validating valid action."""
        action = {
            "type": ActionType.GOTO,
            "params": ["kitchen"]
        }
        valid_actions = {"goto[kitchen]", "pickup[apple]"}
        
        is_valid = self.task.validate_action(action, valid_actions)
        
        self.assertTrue(is_valid)
        
    def test_validate_invalid_action(self):
        """Test validating invalid action."""
        action = {
            "type": ActionType.GOTO,
            "params": ["bedroom"]
        }
        valid_actions = {"goto[kitchen]", "pickup[apple]"}
        
        is_valid = self.task.validate_action(action, valid_actions)
        
        self.assertFalse(is_valid)
        
    def test_solve_successful_task(self):
        """Test solving task successfully."""
        task = "Put the apple on the table."
        
        # Mock model to perform sequence of actions
        self.mock_model.react_step.side_effect = [
            {
                "thought": "I need to pick up the apple first.",
                "action": "pickup[apple]"
            },
            {
                "thought": "Now I can put it on the table.",
                "action": "put[apple|table]"
            }
        ]
        
        result = self.task.solve(task, self.env, max_steps=5)
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertGreater(result["total_reward"], 0)
        
    def test_solve_with_invalid_action(self):
        """Test solving task with invalid action."""
        task = "Put the apple on the table."
        
        # Mock model to generate invalid action
        self.mock_model.react_step.return_value = {
            "thought": "I should teleport.",
            "action": "teleport[kitchen]"
        }
        
        result = self.task.solve(task, self.env, max_steps=1)
        
        self.assertFalse(result["success"])
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(result["total_reward"], 0)
        self.assertIn("error", result["trajectory"][0])
        
    def test_solve_with_no_action(self):
        """Test solving task with no action."""
        task = "Put the apple on the table."
        
        # Mock model to generate no action
        self.mock_model.react_step.return_value = {
            "thought": "I don't know what to do.",
            "action": ""
        }
        
        result = self.task.solve(task, self.env, max_steps=1)
        
        self.assertFalse(result["success"])
        self.assertEqual(len(result["trajectory"]), 0)
        self.assertEqual(result["total_reward"], 0)

if __name__ == '__main__':
    unittest.main()
