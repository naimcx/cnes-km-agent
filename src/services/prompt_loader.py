from src.config import PROMPTS_DIR, SKILLS_DIR

def get_node_prompt(agent_name: str) -> str:
    """Loads the content of a Markdown prompt file."""
    file_path = PROMPTS_DIR / f"{agent_name}.md"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file for agent '{agent_name}' not found at {file_path}")
    return file_path.read_text(encoding="utf-8")

def get_skill_prompt(skill_name: str) -> str:
    """Loads the content of a Markdown skill file."""
    file_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file for skill '{skill_name}' not found at {file_path}")
    return file_path.read_text(encoding="utf-8")