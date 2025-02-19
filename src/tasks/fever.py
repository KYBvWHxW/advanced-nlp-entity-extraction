"""
Fever (Fact Extraction and VERification) task implementation using ReAct approach.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..model import ReActModel
from ..tools.wikipedia_tool import WikipediaTool

class VerificationResult(Enum):
    """Possible verification results."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"

class Fever:
    """
    Implementation of Fever task using ReAct approach.
    """
    
    def __init__(self, model: ReActModel, wiki_tool: WikipediaTool):
        """
        Initialize Fever task.
        
        Args:
            model: ReAct model instance
            wiki_tool: Wikipedia tool instance
        """
        self.model = model
        self.wiki_tool = wiki_tool
        
    def format_prompt(
        self,
        claim: str,
        evidence: Optional[List[Dict[str, str]]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format prompt for the model.
        
        Args:
            claim: Claim to verify
            evidence: Optional list of evidence documents
            history: Optional list of previous reasoning steps
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Claim to verify: {claim}\n\n"
        
        if evidence:
            prompt += "Evidence:\n"
            for doc in evidence:
                prompt += f"- {doc['title']}: {doc['content']}\n"
        
        if history:
            prompt += "\nPrevious steps:\n"
            for step in history:
                prompt += f"- Thought: {step.get('thought', '')}\n"
                if step.get('action'):
                    prompt += f"  Action: {step['action']}\n"
                if step.get('observation'):
                    prompt += f"  Observation: {step['observation']}\n"
                    
        prompt += "\nLet's verify this claim step-by-step:\n"
        return prompt
        
    def extract_entities(self, claim: str) -> List[str]:
        """
        Extract key entities from claim for searching.
        
        Args:
            claim: Claim string
            
        Returns:
            List of entity strings
        """
        # TODO: Implement more sophisticated entity extraction
        # For now, just split on spaces and remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = claim.split()
        return [w for w in words if w.lower() not in common_words]
        
    def execute_action(self, action: str) -> Dict[str, any]:
        """
        Execute an action using available tools.
        
        Args:
            action: Action string to execute
            
        Returns:
            Action results
        """
        if action.startswith("Search[") and action.endswith("]"):
            query = action[7:-1]
            results = self.wiki_tool.search(query)
            return {
                "type": "search",
                "query": query,
                "results": results
            }
        elif action.startswith("Lookup[") and action.endswith("]"):
            title = action[7:-1]
            content = self.wiki_tool.get_page_content(title)
            return {
                "type": "lookup",
                "title": title,
                "content": content
            }
        elif action.startswith("Compare[") and action.endswith("]"):
            # Special action for comparing claim with evidence
            try:
                claim, evidence = action[8:-1].split("|")
                return {
                    "type": "compare",
                    "claim": claim.strip(),
                    "evidence": evidence.strip()
                }
            except ValueError:
                return {
                    "type": "error",
                    "message": "Invalid compare action format"
                }
        else:
            return {
                "type": "error",
                "message": f"Unknown action format: {action}"
            }
            
    def determine_verdict(self, final_thought: str) -> VerificationResult:
        """
        Determine verification result from final thought.
        
        Args:
            final_thought: Final reasoning thought
            
        Returns:
            VerificationResult enum value
        """
        thought_lower = final_thought.lower()
        
        if "support" in thought_lower or "proves" in thought_lower or "correct" in thought_lower:
            return VerificationResult.SUPPORTS
        elif "refute" in thought_lower or "contradict" in thought_lower or "incorrect" in thought_lower:
            return VerificationResult.REFUTES
        else:
            return VerificationResult.NOT_ENOUGH_INFO
            
    def verify(self, claim: str, max_steps: int = 5) -> Dict[str, any]:
        """
        Verify a claim using ReAct approach.
        
        Args:
            claim: Claim to verify
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dictionary containing verification result and reasoning chain
        """
        evidence = []
        history = []
        
        # Initial search for entities
        entities = self.extract_entities(claim)
        initial_results = []
        for entity in entities[:2]:  # Limit initial search to top 2 entities
            results = self.wiki_tool.search(entity)
            initial_results.extend(results)
        
        if initial_results:
            evidence.extend(initial_results)
        
        current_prompt = self.format_prompt(claim, evidence, history)
        
        for step in range(max_steps):
            # Get model output
            output = self.model.react_step(current_prompt)
            thought = output.get("thought", "").strip()
            action = output.get("action", "").strip()
            
            # Record step
            step_record = {"thought": thought}
            
            # If no action, we've reached a conclusion
            if not action:
                break
                
            # Execute action and record results
            step_record["action"] = action
            result = self.execute_action(action)
            step_record["observation"] = str(result)
            
            # Update evidence if new information found
            if result["type"] in ["search", "lookup"] and result.get("results"):
                evidence.extend(result["results"])
                
            history.append(step_record)
            current_prompt = self.format_prompt(claim, evidence, history)
            
        # Determine final verdict
        verdict = self.determine_verdict(thought)
        
        return {
            "claim": claim,
            "verdict": verdict.value,
            "confidence": "HIGH" if len(evidence) > 0 else "LOW",
            "evidence": evidence,
            "reasoning_chain": history,
            "final_thought": thought
        }
