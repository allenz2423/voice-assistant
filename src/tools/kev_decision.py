"""Compatibility imports for the retired local-Kev proof of concept."""
from src.tools.jev_decision import JevDecisionClient, KevActionDecision

KevDecisionClient = JevDecisionClient

__all__ = ["JevDecisionClient", "KevDecisionClient", "KevActionDecision"]
