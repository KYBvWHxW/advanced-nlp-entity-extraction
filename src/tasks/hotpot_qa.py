"""
HotpotQA task implementation using ReAct approach.
"""

from typing import Dict, List, Optional, Tuple

from ..model import ReActModel
from ..tools.wikipedia_tool import WikipediaTool

class HotpotQA:
    """
    Implementation of HotpotQA task using ReAct approach.
    """
    
    def __init__(self, model: ReActModel, wiki_tool: WikipediaTool):
        """
        Initialize HotpotQA task.
        
        Args:
            model: ReAct model instance
            wiki_tool: Wikipedia tool instance
        """
        self.model = model
        self.wiki_tool = wiki_tool
        
    def format_prompt(self, question: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Format prompt for the model.
        
        Args:
            question: Question to answer
            context: Optional list of context documents
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Question: {question}\n\n"
        
        if context:
            prompt += "Context:\n"
            for doc in context:
                prompt += f"- {doc['title']}: {doc['content']}\n"
                
        prompt += "\nLet's approach this step-by-step:\n"
        return prompt
        
    def parse_model_output(self, output: Dict[str, str]) -> Tuple[str, Optional[str]]:
        """
        Parse model output into thought and action.
        
        Args:
            output: Model output dictionary
            
        Returns:
            Tuple of (thought, action)
        """
        thought = output.get("thought", "").strip()
        action = output.get("action", "").strip()
        
        return thought, action if action else None
        
    def execute_action(self, action: str) -> Dict[str, str]:
        """
        Execute an action using available tools.
        
        Args:
            action: Action string to execute
            
        Returns:
            Action results
        """
        # Parse action string
        # Expected format: "Search[query]" or "Lookup[title]"
        if action.startswith("Search[") and action.endswith("]"):
            query = action[7:-1]  # Remove "Search[" and "]"
            results = self.wiki_tool.search(query)
            return {
                "type": "search",
                "query": query,
                "results": results
            }
        elif action.startswith("Lookup[") and action.endswith("]"):
            title = action[7:-1]  # Remove "Lookup[" and "]"
            content = self.wiki_tool.get_page_content(title)
            return {
                "type": "lookup",
                "title": title,
                "content": content
            }
        else:
            return {
                "type": "error",
                "message": f"Unknown action format: {action}"
            }
            
    def solve(self, question: str, max_steps: int = 5) -> Dict[str, str]:
        """
        Solve a HotpotQA question using ReAct approach.
        
        Args:
            question: Question to answer
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dictionary containing answer and reasoning chain
        """
        context = []
        thoughts = []
        actions = []
        observations = []
        
        # Initial prompt
        current_prompt = self.format_prompt(question)
        
        for step in range(max_steps):
            # Get model output
            output = self.model.react_step(current_prompt)
            thought, action = self.parse_model_output(output)
            
            # Record step
            thoughts.append(thought)
            
            # If no action, assume we have the answer
            if not action:
                break
                
            # Execute action and get observation
            actions.append(action)
            result = self.execute_action(action)
            observation = f"Action: {action}\nResult: {result}"
            observations.append(observation)
            
            # Update prompt with new information
            current_prompt = self.format_prompt(
                question,
                context + [{"title": "Observation", "content": observation}]
            )
            
        return {
            "question": question,
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "answer": thoughts[-1] if thoughts else "Unable to answer"
        }
