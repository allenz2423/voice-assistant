import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.skills.manager import SkillManager
from src.llm.brain import AdamBrain
from src.llm.tools import ADAM_TOOLS


def test_create_and_delete_skill(tmp_path):
    custom_dir = tmp_path / "custom_skills"
    sm = SkillManager(custom_skills_dir=custom_dir)

    # 1. Create a new skill
    skill_path = sm.create_or_update_skill(
        skill_id="obs_recording",
        content="Use ffmpeg or obs-cli to record display :0.",
        description="Records the active screen using OBS or ffmpeg.",
    )
    assert skill_path.exists()
    assert skill_path.name == "obs_recording.md"
    assert "obs_recording" in [s["id"] for s in sm.list_skills()]

    # Verify content
    loaded = sm.load_skill("obs_recording")
    assert loaded is not None
    assert "Records the active screen using OBS" in loaded
    assert "Use ffmpeg or obs-cli" in loaded

    # 2. Overwrite / update skill
    updated_path = sm.create_or_update_skill(
        skill_id="obs_recording",
        content="Updated: Use wf-recorder for Wayland recording.",
        description="Updated screen recording guide.",
    )
    assert updated_path == skill_path
    loaded_updated = sm.load_skill("obs_recording")
    assert "wf-recorder" in loaded_updated

    # 3. Delete skill
    deleted = sm.delete_skill("obs_recording")
    assert deleted is True
    assert not skill_path.exists()
    assert sm.load_skill("obs_recording") is None


def test_invalid_skill_id_raises(tmp_path):
    sm = SkillManager(custom_skills_dir=tmp_path)
    with pytest.raises(ValueError):
        sm.create_or_update_skill(skill_id="   ", content="test")
    with pytest.raises(ValueError):
        sm.create_or_update_skill(skill_id="///", content="test")


def test_automatic_skill_matching(tmp_path):
    custom_dir = tmp_path / "skills"
    sm = SkillManager(custom_skills_dir=custom_dir)

    # Create two custom skills
    sm.create_or_update_skill(
        skill_id="git_release",
        content="# Skill: Git Release\n\nRun git tag and git push origin --tags.",
        description="Procedure for tagging and publishing a git release.",
    )
    sm.create_or_update_skill(
        skill_id="docker_cleanup",
        content="# Skill: Docker Cleanup\n\nRun docker system prune -af to free disk space.",
        description="Safely clean up unused docker containers and images.",
    )

    # Match git query
    matches = sm.match_skills("how do I publish a git release tag?")
    assert len(matches) == 1
    assert matches[0][0] == "git_release"
    assert "git tag" in matches[0][1]

    # Match docker query
    docker_matches = sm.match_skills("free up disk space by cleaning docker images")
    assert len(docker_matches) == 1
    assert docker_matches[0][0] == "docker_cleanup"
    assert "docker system prune" in docker_matches[0][1]

    # Unrelated query should not match
    no_matches = sm.match_skills("what is the weather today in Tokyo?")
    assert len(no_matches) == 0

    # Formatted context check
    context = sm.get_matched_skill_context("clean up docker containers")
    assert context is not None
    assert "=== SPECIALIZED SKILL GUIDANCE (DOCKER_CLEANUP) ===" in context
    assert "docker system prune" in context


def test_skill_matching_uses_declared_triggers_not_procedure_body(tmp_path):
    sm = SkillManager(custom_skills_dir=tmp_path / "skills")
    sm.create_or_update_skill(
        skill_id="shenzhen_level_helper",
        description="Navigate Shenzhen I/O levels.",
        content=(
            "## When to Use\n"
            "- Trigger: solve or navigate a Shenzhen I/O level.\n\n"
            "## Procedure\n"
            "Explain entropy and then inspect the puzzle screen."
        ),
    )

    assert sm.match_skills("Can you help me navigate a Shenzhen I/O level?")
    assert sm.match_skills("Explain entropy in plain language.") == []


def test_skill_match_rejects_incidental_shared_word_but_keeps_trigger_match(tmp_path):
    sm = SkillManager(custom_skills_dir=tmp_path / "skills")
    sm.create_or_update_skill(
        skill_id="open_up_work_email",
        description="Open the user's work webmail in the default browser.",
        content=(
            "## When to Use\n"
            "- Trigger: open my work email, check my work inbox, or open Outlook.\n\n"
            "## Procedure\nOpen the configured mailbox."
        ),
    )

    assert sm.match_skills("What times did I work this week?") == []
    matches = sm.match_skills("Can you open my work email?")
    assert matches and matches[0][0] == "open_up_work_email"


def test_startup_loaded_skills_excluded_from_auto_match(tmp_path):
    sm = SkillManager(custom_skills_dir=tmp_path)
    # computer_use is loaded by default at startup, so it shouldn't match as a specialized add-on
    matches = sm.match_skills("computer control click mouse on screen")
    matched_ids = [m[0] for m in matches]
    assert "computer_use" not in matched_ids


@pytest.mark.asyncio
async def test_brain_create_skill_tool_execution(tmp_path):
    custom_dir = tmp_path / "custom_skills"
    config = SimpleNamespace(
        llm=SimpleNamespace(
            provider="local",
            local_model="qwen",
            cloud_model="",
            ollama_host="",
            temperature=0.2,
            num_ctx=4096,
        ),
        execution=SimpleNamespace(downloads_dir="~/Downloads", workspace_dir="~/workspace"),
    )
    brain = AdamBrain(
        config=config,
        supervisor=MagicMock(),
        probe=MagicMock(),
        confirmation_mgr=MagicMock(),
        tts_engine=MagicMock(),
    )
    brain.skill_manager = SkillManager(custom_skills_dir=custom_dir)

    # Verify create_skill tool is in get_tools()
    tool_names = [t.name for t in brain.get_tools()]
    assert "create_skill" in tool_names

    # Execute create_skill via brain
    brain._skill_creation_authorized = True  # This test represents an explicit user request.
    res = await brain._execute_tool(
        "create_skill",
        {
            "skill_name": "kubernetes_status",
            "content": "Use kubectl get pods -A to check cluster state.",
            "description": "Inspect Kubernetes cluster pods.",
        },
    )
    assert "created at" in res
    assert (custom_dir / "kubernetes_status.md").exists()

    # Verify context matches
    matched = brain.skill_manager.get_matched_skill_context("check kubernetes pods")
    assert matched is not None
    assert "kubectl get pods" in matched
