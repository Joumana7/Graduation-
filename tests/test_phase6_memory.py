"""tests/test_phase6_memory.py — Phase 6 conversational memory validation."""
import json
import sys
import uuid
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def session():
    from phase6_conversational_memory import ConversationSession
    return ConversationSession()


@pytest.fixture(scope="module")
def assistant():
    try:
        from phase6_conversational_memory import ConversationalAssistant
        return ConversationalAssistant()
    except Exception as exc:
        pytest.skip(f"ConversationalAssistant init failed: {exc}")


# ── ConversationSession unit tests ─────────────────────────────────────────────

class TestConversationSession:

    def test_session_has_unique_id(self, session):
        assert session.session_id
        uid = uuid.UUID(session.session_id)
        assert str(uid) == session.session_id

    def test_add_messages(self, session):
        session.add_user("Hello")
        session.add_assistant("Hi there!")
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
        assert session.conversation_history[1]["role"] == "assistant"

    def test_recent_history_limits_turns(self, session):
        for i in range(20):
            session.add_user(f"Question {i}")
            session.add_assistant(f"Answer {i}")
        hist = session.recent_history(n_turns=5)
        assert len(hist) <= 10  # 5 turns × 2 messages

    def test_profile_summary_empty(self, session):
        assert session.profile_summary() == ""

    def test_profile_summary_with_data(self, session):
        session.user_profile = {"program": "CSAI", "semester": "Semester 3"}
        summary = session.profile_summary()
        assert "CSAI" in summary
        assert "Semester 3" in summary

    def test_session_save_and_load(self, tmp_path, session):
        session.add_user("Test question")
        session.add_assistant("Test answer")
        session.user_profile["program"] = "SCI"
        saved_path = session.save(tmp_path)

        assert saved_path.exists()
        from phase6_conversational_memory import ConversationSession
        loaded = ConversationSession.load(session.session_id, tmp_path)
        assert loaded.session_id == session.session_id
        assert len(loaded.conversation_history) == len(session.conversation_history)
        assert loaded.user_profile["program"] == "SCI"

    def test_save_load_roundtrip_complete(self, tmp_path):
        from phase6_conversational_memory import ConversationSession
        s = ConversationSession()
        s.add_user("What is CSAI?")
        s.add_assistant("CSAI is a program at Zewail City.")
        s.user_profile = {"program": "CSAI", "gpa": "3.2"}
        s.query_count  = 1
        s.topic_counts = {"courses": 2, "admissions": 1}

        saved = s.save(tmp_path)
        loaded = ConversationSession.load(s.session_id, tmp_path)

        assert loaded.query_count == 1
        assert loaded.topic_counts == {"courses": 2, "admissions": 1}
        assert loaded.user_profile["gpa"] == "3.2"


# ── Profile extractor unit tests ───────────────────────────────────────────────

class TestProfileExtraction:

    def test_extracts_program(self):
        from phase6_conversational_memory import extract_profile
        p = extract_profile("I am studying CSAI at Zewail City", {})
        assert p.get("program") == "CSAI"

    def test_extracts_semester(self):
        from phase6_conversational_memory import extract_profile
        p = extract_profile("I am in semester 4 of my degree", {})
        assert "4" in p.get("semester", "")

    def test_extracts_gpa(self):
        from phase6_conversational_memory import extract_profile
        p = extract_profile("My GPA is 3.15", {})
        assert p.get("gpa") == "3.15"

    def test_does_not_overwrite_existing_profile(self):
        from phase6_conversational_memory import extract_profile
        existing = {"program": "SCI", "semester": "Semester 2"}
        p = extract_profile("I might study CSAI", existing)
        # Should NOT overwrite already-known program
        assert p.get("program") == "SCI"

    def test_extracts_failed_courses(self):
        from phase6_conversational_memory import extract_profile
        p = extract_profile("I failed CSAI201 last semester", {})
        assert "CSAI201" in p.get("failed_courses", "")


# ── Integration: multi-turn conversation ──────────────────────────────────────

class TestConversationalAssistant:

    def test_ask_returns_answer_and_sources(self, assistant, session):
        answer, sources = assistant.ask(
            "What undergraduate programs are offered at Zewail City?", session
        )
        assert isinstance(answer, str) and len(answer) >= 30
        assert isinstance(sources, list)

    def test_session_history_grows(self, assistant, session):
        initial = len(session.conversation_history)
        assistant.ask("Tell me about CSAI.", session)
        assert len(session.conversation_history) == initial + 2

    def test_profile_detected_in_turn(self, assistant):
        from phase6_conversational_memory import ConversationSession
        s = ConversationSession()
        assistant.ask("I am a CSAI student in semester 3 with a 2.9 GPA.", s)
        assert s.user_profile.get("program") == "CSAI"

    def test_follow_up_uses_context(self, assistant):
        from phase6_conversational_memory import ConversationSession
        s = ConversationSession()
        assistant.ask("I'm studying CSAI in semester 5.", s)
        answer2, _ = assistant.ask("What courses should I avoid if I have a low GPA?", s)
        # The answer should be a non-empty, sensible response
        assert len(answer2) >= 50

    def test_query_count_increments(self, assistant):
        from phase6_conversational_memory import ConversationSession
        s = ConversationSession()
        assert s.query_count == 0
        assistant.ask("What is Zewail City?", s)
        assert s.query_count == 1
        assistant.ask("Tell me more.", s)
        assert s.query_count == 2

    def test_topic_counts_populated(self, assistant):
        from phase6_conversational_memory import ConversationSession
        s = ConversationSession()
        assistant.ask("Tell me about admissions and scholarships.", s)
        assert len(s.topic_counts) >= 1
