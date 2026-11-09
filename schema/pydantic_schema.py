from pydantic import BaseModel, Field
from typing import Literal, List

# ==============================================
# ORCHESTRATION & ROUTING SCHEMA
# ==============================================

class OrchestratorPlan(BaseModel):
    """
    Structure for the orchestrator's output, including routing decision and search plan.
    """
    routing: Literal["internal", "external", "hybrid"] = Field(
        ...,
        description="'internal' for CNES documents; 'external' for general physics, space history, or competitors; 'hybrid' for comparisons."
    )
    search_plan: str = Field(
        ...,
        description="Step-by-step query plan in English."
    )


# ==============================================
# QUALITY CONTROL SCHEMA
# ==============================================

class CriticOutput(BaseModel):
    """
    Structure for the critic's evaluation of the analyst's draft answer.
    Evaluates factual accuracy, completeness, and adherence to the context.
    """
    is_faithful: bool = Field(
        ..., # this field is required
        description="True ONLY if every single claim/number exists word for word or strictly implicit in the context."
    )
    answers_all_questions: bool = Field(
        ..., # this field is required
        description="True if all parts of the user question (example: diameter base AND top) are taken into account."
    )
    missing_points: list[str] = Field(
        default_factory=list, 
        description="List of specific questions left unanswered by the draft."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, 
        description="List of statements in the draft not backed by the context."
    )
    
    @property
    def is_valid(self) -> bool:
        """Returns True if the draft is both faithful and complete."""
        return self.is_faithful and self.answers_all_questions

    @property
    def feedback_message(self) -> str:
        """Generates a concise feedback message for the analyst."""
        issues = []
        if self.unsupported_claims:
            issues.append(f"Unsupported claims: {', '.join(self.unsupported_claims)}")
        if self.missing_points:
            issues.append(f"Missing points: {', '.join(self.missing_points)}")
        return " | ".join(issues) if issues else "No issues detected."


# ==============================================
# SELF-REFLECTION SCHEMA
# ==============================================

class ReflectionOutput(BaseModel):
    """Diagnostic output from the self-reflection node, guiding the next action."""
    root_cause: str = Field(
        description="Why did the draft fail? (example: 'missing data in docs', 'analyst misinterpreted page 12')"
    )
    action_type: Literal["re_search", "re_write"] = Field(
        ..., # this field is required
        description="'re_search' if context lacks data, 're_write' if data exists but analyst made a mistake"
    )
    new_search_queries: list[str] = Field(
        default_factory=list,
        description="Refined search queries if action_type is 're_search'"
    )
    critique_for_analyst: str = Field(
        default="",
        description="Instructions for analyst if action_type is 're_write'"
    )

