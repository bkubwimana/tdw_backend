"""
VLM Adapter

Bridges physicai's VLMPlanner with CoELA's prompt template patterns.
Builds structured context from spatial memory + agent state,
sends to VLM, returns a plan string compatible with CoELA's agent loop.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List

import numpy as np

PHYSICAI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PHYSICAI_ROOT not in sys.path:
    sys.path.insert(0, PHYSICAI_ROOT)

logger = logging.getLogger(__name__)


class VLMAdapter:
    """
    Wraps physicai VLMPlanner to produce CoELA-compatible plan strings.

    Instead of CSV templates + LLM text parsing, this uses structured
    context injection into the VLM and parses the structured output.
    """

    def __init__(self, vlm_url: Optional[str] = None):
        self._planner = None
        self._vlm_url = vlm_url
        self._init_planner()

    def _init_planner(self):
        try:
            from simulator.vla.vlm_planner import VLMPlanner
            self._planner = VLMPlanner(url=self._vlm_url)
            logger.info("VLMPlanner connected")
        except Exception as e:
            logger.warning(f"VLMPlanner init failed: {e}")

    @property
    def available(self) -> bool:
        return self._planner is not None

    def plan(self, rgb_image: np.ndarray, context: Dict[str, Any],
             available_actions: List[str]) -> Optional[str]:
        """
        Generate a plan using VLM.

        Args:
            rgb_image: Current camera view (H,W,3)
            context: Agent state context dict
            available_actions: List of valid action strings

        Returns:
            Selected action string from available_actions, or None on failure
        """
        if not self.available:
            return None

        instruction = self._build_instruction(context, available_actions)
        try:
            result = self._planner.plan(
                image=rgb_image,
                instruction=instruction,
                context=context
            )
            return self._match_action(result, available_actions)
        except Exception as e:
            logger.warning(f"VLM plan failed: {e}")
            return None

    def _build_instruction(self, ctx: Dict[str, Any], actions: List[str]) -> str:
        """Build a structured instruction from context (replaces CSV template)."""
        parts = [
            f"Goal: {ctx.get('goal', 'Transport objects to bed.')}",
            f"Room: {ctx.get('current_room', 'unknown')}",
            f"Holding: {ctx.get('holding_desc', 'nothing')}",
            f"Progress: {ctx.get('satisfied_count', 0)} objects transported",
            f"Step: {ctx.get('step', 0)}/{ctx.get('max_steps', 3000)}",
        ]
        if ctx.get('visible_targets'):
            parts.append(f"Visible targets: {ctx['visible_targets']}")
        if ctx.get('visible_containers'):
            parts.append(f"Visible containers: {ctx['visible_containers']}")

        parts.append("\nAvailable actions:")
        for i, a in enumerate(actions):
            parts.append(f"  {chr(ord('A') + i)}. {a}")
        parts.append("\nChoose the best action.")
        return "\n".join(parts)

    def _match_action(self, result, actions: List[str]) -> Optional[str]:
        """Match VLM output to one of the available actions."""
        text = str(result).strip()
        # Direct option letter match
        for i, action in enumerate(actions):
            opt = chr(ord('A') + i)
            if opt in text or action.lower() in text.lower():
                return action
        # Keyword match
        for action in actions:
            keywords = action.split()[:3]
            if all(k.lower() in text.lower() for k in keywords):
                return action
        return actions[0] if actions else None
