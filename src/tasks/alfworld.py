"""
ALFWorld task implementation using ReAct approach.
ALFWorld is a text-based game environment for learning household tasks.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ..model import ReActModel

class ActionType(Enum):
    """Types of actions available in ALFWorld."""
    GOTO = "goto"
    PICKUP = "pickup"
    PUT = "put"
    OPEN = "open"
    CLOSE = "close"
    TOGGLE = "toggle"
    USE = "use"
    LOOK = "look"
    INVENTORY = "inventory"

@dataclass
class GameState:
    """Represents the current state of the ALFWorld game."""
    observation: str
    inventory: List[str]
    valid_actions: Set[str]
    previous_action: Optional[str] = None
    score: float = 0.0
    done: bool = False

class ALFWorld:
    """
    Implementation of ALFWorld task using ReAct approach.
    """
    
    def __init__(self, model: ReActModel):
        """
        Initialize ALFWorld task.
        
        Args:
            model: ReAct model instance
        """
        self.model = model
        self.state = None
        
    def format_prompt(self, task: str, state: GameState) -> str:
        """
        Format prompt for the model.
        
        Args:
            task: Task description
            state: Current game state
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Task: {task}\n\n"
        prompt += "Current state:\n"
        prompt += f"Observation: {state.observation}\n"
        
        if state.inventory:
            prompt += f"Inventory: {', '.join(state.inventory)}\n"
        else:
            prompt += "Inventory: empty\n"
            
        if state.previous_action:
            prompt += f"Previous action: {state.previous_action}\n"
            
        prompt += "\nValid actions:\n"
        for action in state.valid_actions:
            prompt += f"- {action}\n"
            
        prompt += "\nWhat action should I take next? Think step-by-step:\n"
        return prompt
        
    def parse_action(self, action_str: str) -> Optional[Dict[str, str]]:
        """
        Parse action string into structured format.
        
        Args:
            action_str: Action string to parse
            
        Returns:
            Dictionary containing action type and parameters, or None if invalid
        """
        try:
            # Expected format: "ActionType[param1|param2|...]"
            action_type = action_str[:action_str.index("[")]
            params_str = action_str[action_str.index("[")+1:action_str.index("]")]
            params = [p.strip() for p in params_str.split("|")]
            
            return {
                "type": ActionType(action_type.lower()),
                "params": params
            }
        except (ValueError, KeyError):
            return None
            
    def validate_action(self, action: Dict[str, str], valid_actions: Set[str]) -> bool:
        """
        Validate if an action is currently valid.
        
        Args:
            action: Parsed action dictionary
            valid_actions: Set of valid action strings
            
        Returns:
            True if action is valid, False otherwise
        """
        action_str = f"{action['type'].value}[{('|').join(action['params'])}]"
        return action_str in valid_actions
        
    def execute_action(self, action: Dict[str, str], env) -> GameState:
        """
        Execute an action in the environment.
        
        Args:
            action: Parsed action dictionary
            env: ALFWorld environment instance
            
        Returns:
            New game state
        """
        # Convert action to environment format
        action_str = f"{action['type'].value}[{('|').join(action['params'])}]"
        
        # Execute in environment
        obs, score, done, info = env.step(action_str)
        
        # Update game state
        return GameState(
            observation=obs,
            inventory=info.get("inventory", []),
            valid_actions=set(info.get("valid_actions", [])),
            previous_action=action_str,
            score=score,
            done=done
        )
        
    def solve(self, task: str, env, max_steps: int = 50) -> Dict[str, any]:
        """
        Solve an ALFWorld task using ReAct approach.
        
        Args:
            task: Task description
            env: ALFWorld environment instance
            max_steps: Maximum number of steps to take
            
        Returns:
            Dictionary containing solution trajectory and results
        """
        # Initialize environment
        obs = env.reset()
        self.state = GameState(
            observation=obs,
            inventory=[],
            valid_actions=set(env.valid_actions)
        )
        
        trajectory = []
        total_reward = 0
        
        for step in range(max_steps):
            # Get model output
            prompt = self.format_prompt(task, self.state)
            output = self.model.react_step(prompt)
            
            thought = output.get("thought", "").strip()
            action_str = output.get("action", "").strip()
            
            # Record step
            step_record = {
                "observation": self.state.observation,
                "thought": thought,
                "action": action_str
            }
            
            # Parse and validate action
            if not action_str:
                break
                
            action = self.parse_action(action_str)
            if not action or not self.validate_action(action, self.state.valid_actions):
                step_record["error"] = "Invalid action"
                trajectory.append(step_record)
                continue
                
            # Execute action
            self.state = self.execute_action(action, env)
            total_reward += self.state.score
            
            step_record["result"] = self.state.observation
            trajectory.append(step_record)
            
            if self.state.done:
                break
                
        return {
            "task": task,
            "success": self.state.done and total_reward > 0,
            "trajectory": trajectory,
            "total_reward": total_reward,
            "steps_taken": len(trajectory)
        }
