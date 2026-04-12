"""Tests for cognitive engine — forgetting, ontology, confidence, priming."""

from datetime import datetime, timezone, timedelta

from jarvis.cognitive.forgetting import (
    compute_importance,
    compute_recency,
    find_stale_memories,
    should_prune,
)
from jarvis.cognitive.ontology import (
    get_decay_half_life,
    get_default_importance,
    infer_subtype,
)
from jarvis.cognitive.confidence import update_confidence, is_uncertain
from jarvis.cognitive.priming import extract_context_signals, auto_detect_context
from jarvis.models.memory import Memory, MemoryType, MemorySubtype


def _make_memory(days_old: float = 0, **kwargs) -> Memory:
    defaults = dict(
        type=MemoryType.EPISODIC,
        content="Test",
        source="manual",
        importance=0.5,
        access_count=0,
        decay_half_life_days=7.0,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_old),
    )
    defaults.update(kwargs)
    return Memory(**defaults)


class TestForgetting:
    def test_fresh_memory_full_importance(self):
        mem = _make_memory(days_old=0)
        imp = compute_importance(mem)
        assert abs(imp - 0.5) < 0.01

    def test_half_life_decay(self):
        mem = _make_memory(days_old=7, decay_half_life_days=7.0)
        imp = compute_importance(mem)
        # After one half-life, should be roughly half
        assert 0.2 < imp < 0.3

    def test_access_reinforcement(self):
        mem_no_access = _make_memory(days_old=7, access_count=0)
        mem_accessed = _make_memory(days_old=7, access_count=10)
        imp_no = compute_importance(mem_no_access)
        imp_yes = compute_importance(mem_accessed)
        assert imp_yes > imp_no

    def test_procedural_command_never_decays(self):
        mem = _make_memory(
            days_old=365,
            type=MemoryType.PROCEDURAL,
            decay_half_life_days=99999.0,
        )
        imp = compute_importance(mem)
        assert imp > 0.49  # barely decayed

    def test_should_prune(self):
        old_unaccessed = _make_memory(days_old=60, importance=0.1)
        assert should_prune(old_unaccessed)

    def test_recency_score(self):
        recent = _make_memory(days_old=0)
        old = _make_memory(days_old=30)
        assert compute_recency(recent) > compute_recency(old)

    def test_find_stale(self):
        memories = [
            _make_memory(days_old=0, importance=0.5),
            _make_memory(days_old=100, importance=0.05),
        ]
        stale = find_stale_memories(memories, threshold=0.1)
        assert len(stale) == 1


class TestOntology:
    def test_debug_session_half_life(self):
        hl = get_decay_half_life(MemoryType.EPISODIC, MemorySubtype.DEBUG_SESSION)
        assert hl == 7.0

    def test_architecture_half_life(self):
        hl = get_decay_half_life(MemoryType.SEMANTIC, MemorySubtype.ARCHITECTURE)
        assert hl == 180.0

    def test_command_never_decays(self):
        hl = get_decay_half_life(MemoryType.PROCEDURAL, MemorySubtype.COMMAND)
        assert hl == 99999.0

    def test_default_importance(self):
        assert get_default_importance(MemorySubtype.INCIDENT) == 0.7
        assert get_default_importance(MemorySubtype.COMMAND) == 0.8

    def test_infer_subtype_debug(self):
        st = infer_subtype(MemoryType.EPISODIC, "Fixed bug in billing", "git")
        assert st == MemorySubtype.DEBUG_SESSION

    def test_infer_subtype_command(self):
        st = infer_subtype(MemoryType.PROCEDURAL, "run `pytest -v`", "manual")
        assert st == MemorySubtype.COMMAND

    def test_infer_subtype_architecture(self):
        st = infer_subtype(MemoryType.SEMANTIC, "Clean Architecture pattern used", "manual")
        assert st == MemorySubtype.ARCHITECTURE


class TestConfidence:
    def test_confirming_evidence_increases(self):
        mem = _make_memory(confidence=0.5)
        new_conf = update_confidence(mem, evidence_strength=0.8, is_confirming=True)
        assert new_conf > 0.5

    def test_contradictory_evidence_decreases(self):
        mem = _make_memory(confidence=0.8)
        new_conf = update_confidence(mem, evidence_strength=0.8, is_confirming=False)
        assert new_conf < 0.8

    def test_confidence_bounds(self):
        mem = _make_memory(confidence=0.99)
        new_conf = update_confidence(mem, evidence_strength=1.0, is_confirming=True)
        assert new_conf <= 0.99

        mem2 = _make_memory(confidence=0.05)
        new_conf2 = update_confidence(mem2, evidence_strength=1.0, is_confirming=False)
        assert new_conf2 >= 0.05

    def test_is_uncertain(self):
        mem = _make_memory(confidence=0.3)
        assert is_uncertain(mem)
        mem2 = _make_memory(confidence=0.7)
        assert not is_uncertain(mem2)


class TestPriming:
    def test_extract_service_from_path(self):
        signals = extract_context_signals(file_path="services/billing/app/adapters/repo.py")
        assert "billing" in signals["services"]

    def test_extract_ticket_from_branch(self):
        signals = extract_context_signals(git_branch="gc-blo-1774")
        assert "BLO-1774" in signals["tickets"]

    def test_extract_multiple_signals(self):
        signals = extract_context_signals(
            file_path="services/cloud-scanner/modules/commitment/main.py",
            git_branch="gc-blo-1490",
            recent_errors=["Permission denied: timeout"],
        )
        assert "cloud-scanner" in signals["services"]
        assert "BLO-1490" in signals["tickets"]
        assert "timeout" in signals["keywords"] or "permission" in signals["keywords"]

    def test_empty_context(self):
        signals = extract_context_signals()
        assert signals["services"] == []
        assert signals["tickets"] == []

    def test_auto_detect_git_context(self):
        # Test with ground-control (known git repo with commits)
        ctx = auto_detect_context("/Users/darshanthevarmahalingam/Desktop/ground-control")
        assert ctx["git_branch"] is not None
        assert isinstance(ctx["recent_commit_messages"], list)

    def test_auto_detect_nonexistent_dir(self):
        ctx = auto_detect_context("/nonexistent/path")
        assert ctx["git_branch"] is None

    def test_commit_message_keywords_extracted(self):
        signals = extract_context_signals(
            mentioned_text="fix SEPA webhook bug and update billing tests"
        )
        assert "fix" in signals["keywords"] or "update" in signals["keywords"]

    def test_modified_files_detect_services(self):
        signals = extract_context_signals(
            auto_context={"git_branch": "main", "modified_files": ["services/billing/app.py"], "recent_commit_messages": []}
        )
        assert "billing" in signals["services"]
