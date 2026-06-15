#!/usr/bin/env python3
"""
tests/test_phase8_advisor.py — Academic Advisor Engine Tests
═══════════════════════════════════════════════════════════════════════════════
Comprehensive test suite for phase8_advisor_engine.py.

Test coverage:
  Unit tests (no API key required):
    - StudentProfile creation and serialisation
    - Profile extraction from natural language (Engine 2)
    - Intent routing (IntentRouter)
    - Prerequisite graph operations (Engines 3 + 4)
    - Eligibility analysis (Engine 5)
    - Risk flag generation (Engine 9)
    - Graduation estimate (Engine 7)

  Integration scenarios (require OPENAI_API_KEY + ChromaDB):
    Scenario A: Student with failed courses
    Scenario B: Student near graduation
    Scenario C: Student with low GPA
    Scenario D: Student with missing prerequisites
    Scenario E: General campus question (passes through to RAG)

Run unit tests only:
    pytest tests/test_phase8_advisor.py -m "not integration" -v

Run all tests (requires API key):
    pytest tests/test_phase8_advisor.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase8_advisor_engine import (
    StudentProfile,
    CurriculumGraph,
    IntentRouter,
    update_profile,
    _GPA_PROBATION,
    _GPA_WARNING,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_courses_json(tmp_path: Path) -> Path:
    """
    Minimal curriculum JSON for testing — CSAI prerequisite chain.

    Chain: CSAI 101 → CSAI 201 → CSAI 251 → CSAI 253 (Machine Learning)
                                             → CSAI 261 → CSAI 371 (DSP)
    """
    data = {
        "version": "1.0",
        "built_at": "2026-06-14T00:00:00+00:00",
        "total_courses": 8,
        "courses": {
            "CSAI 101": {"name": "Introduction to Computer Science", "credits": 3, "prerequisites": []},
            "CSAI 102": {"name": "Computing for Engineering",        "credits": 3, "prerequisites": []},
            "CSAI 201": {"name": "Data Structures",                  "credits": 3, "prerequisites": ["CSAI 101"]},
            "CSAI 251": {"name": "Algorithms",                       "credits": 3, "prerequisites": ["CSAI 201"]},
            "CSAI 253": {"name": "Machine Learning",                 "credits": 3, "prerequisites": ["CSAI 251"]},
            "CSAI 261": {"name": "Signals and Systems",              "credits": 3, "prerequisites": ["CSAI 101"]},
            "CSAI 371": {"name": "Digital Signal Processing",        "credits": 3, "prerequisites": ["CSAI 261"]},
            "MATH 201": {"name": "Calculus I",                       "credits": 3, "prerequisites": []},
        },
    }
    p = tmp_path / "courses.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def graph(sample_courses_json: Path) -> CurriculumGraph:
    return CurriculumGraph(sample_courses_json)


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


@pytest.fixture
def empty_profile() -> StudentProfile:
    return StudentProfile()


@pytest.fixture
def csai_profile() -> StudentProfile:
    """Typical CSAI semester 5 student."""
    return StudentProfile(
        major="CS",
        school="CSAI",
        semester=5,
        gpa=2.8,
        completed_courses=["CSAI 101", "CSAI 102", "CSAI 201", "MATH 201"],
        failed_courses=["CSAI 261"],
        current_courses=[],
        completed_credits=45,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — StudentProfile
# ══════════════════════════════════════════════════════════════════════════════

class TestStudentProfile:

    def test_default_values(self, empty_profile: StudentProfile) -> None:
        assert empty_profile.major == ""
        assert empty_profile.school == ""
        assert empty_profile.semester == 0
        assert empty_profile.gpa == 0.0
        assert empty_profile.completed_courses == []
        assert empty_profile.failed_courses == []

    def test_has_academic_context_empty(self, empty_profile: StudentProfile) -> None:
        assert not empty_profile.has_academic_context()

    def test_has_academic_context_with_school(self) -> None:
        p = StudentProfile(school="CSAI")
        assert p.has_academic_context()

    def test_has_academic_context_with_courses(self) -> None:
        p = StudentProfile(completed_courses=["CSAI 101"])
        assert p.has_academic_context()

    def test_is_ready_for_planning(self, csai_profile: StudentProfile) -> None:
        assert csai_profile.is_ready_for_planning()

    def test_is_not_ready_for_planning_no_school(self) -> None:
        p = StudentProfile(gpa=3.0)
        assert not p.is_ready_for_planning()

    def test_serialisation_round_trip(self, csai_profile: StudentProfile) -> None:
        d = csai_profile.to_dict()
        restored = StudentProfile.from_dict(d)
        assert restored.major    == csai_profile.major
        assert restored.school   == csai_profile.school
        assert restored.semester == csai_profile.semester
        assert restored.gpa      == csai_profile.gpa
        assert restored.completed_courses == csai_profile.completed_courses
        assert restored.failed_courses    == csai_profile.failed_courses

    def test_from_dict_back_compat_string_courses(self) -> None:
        """Old sessions stored completed_courses as comma-separated strings."""
        old_dict = {
            "major":             "CS",
            "school":            "CSAI",
            "semester":          3,
            "gpa":               3.1,
            "completed_courses": "CSAI 101, CSAI 102",   # old format
            "failed_courses":    "",
            "current_courses":   "",
            "completed_credits": 0,
            "preferences":       {},
        }
        p = StudentProfile.from_dict(old_dict)
        assert p.completed_courses == ["CSAI 101", "CSAI 102"]
        assert p.failed_courses    == []

    def test_gpa_probation_flag_in_summary(self) -> None:
        p = StudentProfile(gpa=1.8, school="CSAI")
        summary = p.summary_for_prompt()
        assert "PROBATION" in summary

    def test_gpa_warning_flag_in_summary(self) -> None:
        p = StudentProfile(gpa=2.3, school="CSAI")
        summary = p.summary_for_prompt()
        assert "LOW GPA" in summary

    def test_total_credits_by_school(self) -> None:
        assert StudentProfile(school="CSAI").total_credits_needed() == 132
        assert StudentProfile(school="BUS").total_credits_needed()  == 114
        assert StudentProfile(school="ENGR").total_credits_needed() == 140

    def test_remaining_credits_from_completed(self) -> None:
        p = StudentProfile(school="CSAI", completed_credits=60)
        assert p.estimated_remaining_credits() == 72  # 132 - 60

    def test_remaining_credits_from_semester(self) -> None:
        p = StudentProfile(school="CSAI", semester=4)  # ~45 credits done
        remaining = p.estimated_remaining_credits()
        assert remaining == 132 - (3 * 15)  # semester 4 → 3 completed sems × 15 cr

    def test_sidebar_items_not_empty(self, csai_profile: StudentProfile) -> None:
        items = csai_profile.sidebar_items()
        labels = [label for label, _ in items]
        assert "Major"    in labels
        assert "School"   in labels
        assert "Semester" in labels
        assert "GPA"      in labels


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Profile extraction (Engine 2)
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileExtraction:

    def test_extract_school_csai(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I'm a CSAI student", empty_profile)
        assert p.school == "CSAI"

    def test_extract_major_computer_science(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I'm studying computer science", empty_profile)
        assert p.major == "CS"
        assert p.school == "CSAI"

    def test_extract_major_dsai(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I am in DSAI program", empty_profile)
        assert p.major == "DSAI"

    def test_extract_semester_number(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I'm in semester 5", empty_profile)
        assert p.semester == 5

    def test_extract_semester_ordinal(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I'm a 3rd semester student", empty_profile)
        assert p.semester == 3

    def test_extract_gpa(self, empty_profile: StudentProfile) -> None:
        p = update_profile("My GPA is 2.8", empty_profile)
        assert p.gpa == 2.8

    def test_extract_gpa_alternative_phrasing(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I have a 3.5 GPA", empty_profile)
        assert p.gpa == 3.5

    def test_extract_completed_courses_by_code(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I completed CSAI 101, CSAI 201", empty_profile)
        assert "CSAI 101" in p.completed_courses
        assert "CSAI 201" in p.completed_courses

    def test_extract_failed_courses_by_code(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I failed CSAI 261", empty_profile)
        assert "CSAI 261" in p.failed_courses

    def test_extract_failed_courses_by_name(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I failed Signals and Systems", empty_profile)
        assert len(p.failed_courses) > 0

    def test_extract_completed_credits(self, empty_profile: StudentProfile) -> None:
        p = update_profile("I have completed 60 credits", empty_profile)
        assert p.completed_credits == 60

    def test_full_profile_in_one_message(self, empty_profile: StudentProfile) -> None:
        msg = (
            "I'm CSAI. I'm currently in semester 5. My GPA is 2.8. "
            "I completed CSAI 101, CSAI 201, CSAI 251. "
            "I failed CSAI 261."
        )
        p = update_profile(msg, empty_profile)
        assert p.school   == "CSAI"
        assert p.semester == 5
        assert p.gpa      == 2.8
        assert "CSAI 101" in p.completed_courses
        assert "CSAI 261" in p.failed_courses

    def test_no_overwrite_existing_school(self) -> None:
        existing = StudentProfile(school="CSAI", major="CS")
        p = update_profile("What about BUS programs?", existing)
        # school and major should not be overwritten by a general question
        assert p.school == "CSAI"

    def test_gpa_update_allowed(self) -> None:
        existing = StudentProfile(gpa=2.8)
        p = update_profile("Actually my GPA is now 3.0", existing)
        assert p.gpa == 3.0

    def test_accumulate_courses_across_messages(self, empty_profile: StudentProfile) -> None:
        p1 = update_profile("I completed CSAI 101", empty_profile)
        p2 = update_profile("I also passed CSAI 201", p1)
        assert "CSAI 101" in p2.completed_courses
        assert "CSAI 201" in p2.completed_courses


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Intent Router
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentRouter:

    def test_planning_intent_explicit(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("What courses should I take next semester?", empty_profile) == "planning"

    def test_planning_intent_recommend(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("Recommend courses for me", empty_profile) == "planning"

    def test_graduation_intent(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        intent = router.classify("When will I graduate?", empty_profile)
        assert intent in ("graduation", "planning")

    def test_prereq_intent_can_i_take(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("Can I take CSAI 253?", empty_profile) == "prerequisite"

    def test_prereq_intent_explicit(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("What are the prerequisites for Machine Learning?", empty_profile) == "prerequisite"

    def test_risk_intent(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("Am I at risk of academic probation?", empty_profile) == "risk"

    def test_profile_intent_no_question(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("I am a CSAI student in semester 3 with a 2.9 GPA", empty_profile) == "profile"

    def test_general_intent_scholarship(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("What scholarships are available?", empty_profile) == "general"

    def test_general_intent_faculty(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("Who is the dean of CSAI?", empty_profile) == "general"

    def test_general_intent_admission(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("What are the admission requirements?", empty_profile) == "general"

    def test_plan_my_semester_explicit(self, router: IntentRouter, empty_profile: StudentProfile) -> None:
        assert router.classify("Plan my semester for me", empty_profile) == "planning"

    def test_credits_remaining_is_graduation(self, router: IntentRouter, csai_profile: StudentProfile) -> None:
        intent = router.classify("How many credits do I have remaining?", csai_profile)
        assert intent in ("graduation", "planning")


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — CurriculumGraph (Engines 3 + 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestCurriculumGraph:

    def test_loads_courses(self, graph: CurriculumGraph) -> None:
        assert graph.available
        assert graph.course_count == 8

    def test_empty_graph_when_file_missing(self) -> None:
        g = CurriculumGraph(Path("/nonexistent/courses.json"))
        assert not g.available
        assert g.course_count == 0

    def test_get_info_known_course(self, graph: CurriculumGraph) -> None:
        info = graph.get_info("CSAI 201")
        assert info is not None
        assert info["name"]    == "Data Structures"
        assert info["credits"] == 3

    def test_get_info_unknown_course(self, graph: CurriculumGraph) -> None:
        assert graph.get_info("CSAI 999") is None

    def test_get_prereqs_chain(self, graph: CurriculumGraph) -> None:
        assert graph.get_prereqs("CSAI 201") == ["CSAI 101"]
        assert graph.get_prereqs("CSAI 251") == ["CSAI 201"]
        assert graph.get_prereqs("CSAI 253") == ["CSAI 251"]

    def test_get_prereqs_no_prereqs(self, graph: CurriculumGraph) -> None:
        assert graph.get_prereqs("CSAI 101") == []
        assert graph.get_prereqs("MATH 201") == []

    def test_prereqs_met_true(self, graph: CurriculumGraph) -> None:
        done = {"CSAI 101"}
        assert graph.prereqs_met("CSAI 201", done)

    def test_prereqs_met_false(self, graph: CurriculumGraph) -> None:
        done = set()  # nothing completed
        assert not graph.prereqs_met("CSAI 201", done)

    def test_resolve_code_direct(self, graph: CurriculumGraph) -> None:
        assert graph.resolve_code("CSAI 201") == "CSAI 201"

    def test_resolve_code_by_name(self, graph: CurriculumGraph) -> None:
        resolved = graph.resolve_code("data structures")
        assert resolved == "CSAI 201"

    def test_resolve_code_unknown(self, graph: CurriculumGraph) -> None:
        assert graph.resolve_code("CSAI 999") is None

    def test_unlock_path_dsp(self, graph: CurriculumGraph) -> None:
        path = graph.unlock_path("CSAI 371")
        # DSP needs CSAI 261, which needs CSAI 101
        assert "CSAI 261" in path
        assert "CSAI 101" in path

    def test_courses_unlocked_by(self, graph: CurriculumGraph) -> None:
        # If student completes CSAI 101, they unlock CSAI 201 and CSAI 261
        done = set()
        new  = {"CSAI 101"}
        unlocked = graph.courses_unlocked_by(done, new)
        assert "CSAI 201" in unlocked
        assert "CSAI 261" in unlocked

    def test_self_referential_prereqs_not_present(self, graph: CurriculumGraph) -> None:
        for code, info in graph._courses.items():
            assert code not in info.get("prerequisites", []), \
                f"{code} lists itself as a prerequisite"


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Eligibility Engine (Engine 5)
# ══════════════════════════════════════════════════════════════════════════════

class TestEligibilityEngine:

    def test_eligible_courses_with_no_completed(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(completed=[], failed=[], current=[])
        eligible_codes = [e[0] for e in result["eligible"]]
        # Only no-prereq courses should be eligible
        assert "CSAI 101" in eligible_codes
        assert "CSAI 102" in eligible_codes
        assert "MATH 201" in eligible_codes
        assert "CSAI 201" not in eligible_codes   # needs CSAI 101

    def test_eligible_courses_after_csai_101(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["CSAI 101"], failed=[], current=[],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        assert "CSAI 201" in eligible_codes
        assert "CSAI 261" in eligible_codes
        assert "CSAI 251" not in eligible_codes   # needs CSAI 201

    def test_failed_course_appears_in_retake(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["CSAI 101"],
            failed=["CSAI 261"],
            current=[],
        )
        retake_codes = [r[0] for r in result["retake_eligible"]]
        assert "CSAI 261" in retake_codes

    def test_failed_prereq_blocks_dependent(self, graph: CurriculumGraph) -> None:
        """If CSAI 261 is failed, CSAI 371 (DSP) must be blocked."""
        result = graph.analyze_eligibility(
            completed=["CSAI 101", "CSAI 261"],   # appeared in completed list
            failed=["CSAI 261"],                   # but also failed → not effective
            current=[],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        retake_codes   = [r[0] for r in result["retake_eligible"]]
        blocked_codes  = [b[0] for b in result["blocked"]]

        assert "CSAI 371" not in eligible_codes, \
            "DSP should be blocked when Signals is failed"
        assert "CSAI 371" in blocked_codes

    def test_current_courses_excluded_from_eligible(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["CSAI 101"],
            failed=[],
            current=["CSAI 201"],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        assert "CSAI 201" not in eligible_codes   # currently taking

    def test_completed_courses_excluded_from_eligible(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["CSAI 101", "CSAI 201"],
            failed=[],
            current=[],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        assert "CSAI 101" not in eligible_codes
        assert "CSAI 201" not in eligible_codes

    def test_blocked_course_shows_missing_prereqs(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=[],  # nothing done
            failed=[],
            current=[],
        )
        blocked = {b[0]: b[3] for b in result["blocked"]}
        assert "CSAI 201" in blocked
        assert "CSAI 101" in blocked["CSAI 201"]

    def test_completed_codes_set_correct(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["CSAI 101", "CSAI 201"],
            failed=["CSAI 201"],   # CSAI 201 is in both → net not done
            current=[],
        )
        done = result["completed_codes"]
        assert "CSAI 101" in done
        assert "CSAI 201" not in done   # failed overrides completion

    def test_name_resolution_in_completed(self, graph: CurriculumGraph) -> None:
        result = graph.analyze_eligibility(
            completed=["introduction to computer science"],  # by name
            failed=[],
            current=[],
        )
        done = result["completed_codes"]
        assert "CSAI 101" in done

    # ── Scenario A: Student with failed courses ────────────────────────────────

    def test_scenario_a_failed_signals(self, graph: CurriculumGraph) -> None:
        """Student failed Signals and Systems — DSP must be blocked."""
        result = graph.analyze_eligibility(
            completed=["CSAI 101", "CSAI 102", "CSAI 201", "CSAI 251", "MATH 201"],
            failed=["CSAI 261"],
            current=[],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        blocked_codes  = [b[0] for b in result["blocked"]]
        retake_codes   = [r[0] for r in result["retake_eligible"]]

        assert "CSAI 371" not in eligible_codes, "DSP blocked because Signals failed"
        assert "CSAI 371" in blocked_codes
        assert "CSAI 261" in retake_codes, "Signals eligible for retake"
        assert "CSAI 253" in eligible_codes, "ML eligible (prereq chain: 101→201→251→253)"

    # ── Scenario D: Student with missing prerequisites ─────────────────────────

    def test_scenario_d_missing_prereqs(self, graph: CurriculumGraph) -> None:
        """Student skipped CSAI 201 — Algorithms should be blocked."""
        result = graph.analyze_eligibility(
            completed=["CSAI 101"],  # skipped CSAI 201
            failed=[],
            current=[],
        )
        eligible_codes = [e[0] for e in result["eligible"]]
        blocked_codes  = [b[0] for b in result["blocked"]]

        assert "CSAI 201" in eligible_codes,  "Data Structures now eligible"
        assert "CSAI 251" not in eligible_codes, "Algorithms blocked (needs 201)"
        assert "CSAI 251" in blocked_codes


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Risk Engine (Engine 9)
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskEngine:
    """Test risk flag generation indirectly via AdvisorEngine._risk_flags."""

    @pytest.fixture
    def engine(self, sample_courses_json: Path):
        """AdvisorEngine with a mock RAG (no API calls)."""
        from phase8_advisor_engine import AdvisorEngine
        import types

        mock_rag = types.SimpleNamespace(
            _chat_model="gpt-4o",
            retrieve=lambda q, top_k=6: ([], ""),
        )
        # Temporarily override courses file
        import phase8_advisor_engine as m8
        original = m8.COURSES_FILE
        m8.COURSES_FILE = sample_courses_json
        engine = AdvisorEngine.__new__(AdvisorEngine)
        engine._rag    = mock_rag
        engine._graph  = CurriculumGraph(sample_courses_json)
        from phase8_advisor_engine import IntentRouter
        engine._router = IntentRouter()
        m8.COURSES_FILE = original
        return engine

    def test_probation_risk_low_gpa(self, engine) -> None:
        p = StudentProfile(school="CSAI", gpa=1.5)
        flags = engine._risk_flags(p)
        assert "PROBATION" in flags.upper() or "HIGH" in flags.upper()

    def test_warning_risk_medium_gpa(self, engine) -> None:
        p = StudentProfile(school="CSAI", gpa=2.3)
        flags = engine._risk_flags(p)
        assert "MEDIUM" in flags.upper() or "WARNING" in flags.upper()

    def test_no_risk_good_gpa(self, engine) -> None:
        p = StudentProfile(school="CSAI", gpa=3.2)
        flags = engine._risk_flags(p)
        assert "No major risk" in flags

    def test_retake_flag_when_failed(self, engine) -> None:
        p = StudentProfile(school="CSAI", gpa=3.0, failed_courses=["CSAI 261"])
        flags = engine._risk_flags(p)
        assert "RETAKE" in flags.upper() or "CSAI 261" in flags

    # ── Scenario C: Student with low GPA ──────────────────────────────────────

    def test_scenario_c_low_gpa(self, engine) -> None:
        """GPA 1.8 → probation risk flagged, heavy load warning expected."""
        p = StudentProfile(school="CSAI", gpa=1.8, semester=3)
        flags = engine._risk_flags(p)
        assert "1.8" in flags or "PROBATION" in flags.upper() or "HIGH" in flags.upper()


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Graduation Engine (Engine 7)
# ══════════════════════════════════════════════════════════════════════════════

class TestGraduationEngine:
    """Test graduation estimate via AdvisorEngine._graduation_estimate."""

    @pytest.fixture
    def engine(self, sample_courses_json: Path):
        from phase8_advisor_engine import AdvisorEngine
        import types

        mock_rag = types.SimpleNamespace(
            _chat_model="gpt-4o",
            retrieve=lambda q, top_k=6: ([], ""),
        )
        engine = AdvisorEngine.__new__(AdvisorEngine)
        engine._rag    = mock_rag
        engine._graph  = CurriculumGraph(sample_courses_json)
        from phase8_advisor_engine import IntentRouter
        engine._router = IntentRouter()
        return engine

    def test_graduation_estimate_contains_remaining(self, engine) -> None:
        p = StudentProfile(school="CSAI", completed_credits=60)
        text = engine._graduation_estimate(p)
        assert "72" in text  # 132 - 60

    def test_graduation_estimate_semesters_present(self, engine) -> None:
        p = StudentProfile(school="CSAI", completed_credits=90)
        text = engine._graduation_estimate(p)
        assert "Semesters" in text

    # ── Scenario B: Student near graduation ───────────────────────────────────

    def test_scenario_b_near_graduation(self, engine) -> None:
        """Student with 120/132 credits — should show ~1 semester remaining."""
        p = StudentProfile(school="CSAI", completed_credits=120, semester=7)
        text = engine._graduation_estimate(p)
        assert "12" in text   # 132 - 120 = 12 credits remaining

    def test_graduation_no_delay_with_failed_course(self, engine) -> None:
        """Failed courses should add warning to graduation estimate."""
        p = StudentProfile(school="CSAI", completed_credits=90, failed_courses=["CSAI 261"])
        text = engine._graduation_estimate(p)
        assert "failed" in text.lower() or "retake" in text.lower() or "repeat" in text.lower()


# ══════════════════════════════════════════════════════════════════════════════
#  Unit tests — Profile acknowledgement (Engine 11 route)
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileAcknowledgement:

    @pytest.fixture
    def engine(self, sample_courses_json: Path):
        from phase8_advisor_engine import AdvisorEngine
        import types

        mock_rag = types.SimpleNamespace(
            _chat_model="gpt-4o",
            retrieve=lambda q, top_k=6: ([], ""),
        )
        engine = AdvisorEngine.__new__(AdvisorEngine)
        engine._rag    = mock_rag
        engine._graph  = CurriculumGraph(sample_courses_json)
        from phase8_advisor_engine import IntentRouter
        engine._router = IntentRouter()
        return engine

    def test_ack_empty_profile_asks_for_info(self, engine) -> None:
        p = StudentProfile()
        ack = engine._ack_profile(p)
        assert "program" in ack.lower() or "school" in ack.lower() or "major" in ack.lower()

    def test_ack_partial_profile_shows_fields(self, engine) -> None:
        p = StudentProfile(school="CSAI", semester=3, gpa=2.9)
        ack = engine._ack_profile(p)
        assert "CSAI"  in ack
        assert "3"     in ack
        assert "2.9"   in ack

    def test_advise_profile_intent_returns_ack(self, engine) -> None:
        p = StudentProfile(school="CSAI", semester=5)
        msg = "I am a CSAI student in semester 5 with GPA 3.0"
        answer, chunks = engine.advise(msg, p, [])
        assert answer is not None
        assert chunks == []  # no RAG chunks for profile acknowledgement

    def test_advise_general_returns_none(self, engine) -> None:
        p = StudentProfile()
        msg = "What scholarships are available?"
        answer, chunks = engine.advise(msg, p, [])
        assert answer is None  # signals caller to use standard RAG


# ══════════════════════════════════════════════════════════════════════════════
#  Integration tests (require OPENAI_API_KEY + ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegrationScenarios:
    """
    End-to-end tests through ConversationalAssistant.

    These hit the real OpenAI API and ChromaDB.
    Run with:  pytest tests/test_phase8_advisor.py -m integration -v
    """

    @pytest.fixture(scope="class")
    def assistant(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        from phase6_conversational_memory import ConversationalAssistant
        return ConversationalAssistant()

    @pytest.fixture
    def session(self):
        from phase6_conversational_memory import ConversationSession
        return ConversationSession()

    # ── Scenario A: Student with failed courses ────────────────────────────────

    def test_scenario_a_failed_courses(self, assistant, session) -> None:
        """Advisor correctly identifies Signals as failed and blocks DSP."""
        q = (
            "I'm CSAI, semester 5, GPA 2.8. "
            "I completed CSAI 101, CSAI 201, CSAI 251. "
            "I failed CSAI 261 (Signals and Systems). "
            "What should I take next semester?"
        )
        answer, chunks = assistant.ask(q, session)
        assert answer, "No answer returned"
        low = answer.lower()
        assert "csai 261" in low or "signals" in low, \
            "Advisor should mention the failed Signals course"
        # DSP should appear as blocked or warned
        if "csai 371" in low or "digital signal" in low:
            assert "block" in low or "prereq" in low or "required" in low, \
                "DSP mentioned but not flagged as blocked"

    # ── Scenario B: Student near graduation ───────────────────────────────────

    def test_scenario_b_near_graduation(self, assistant, session) -> None:
        """Student in semester 7 with ~120 credits — advisor shows short roadmap."""
        q = (
            "I'm in CSAI (Computer Science), semester 7. "
            "I have completed 120 credits. GPA is 3.4. "
            "When will I graduate and what do I have left?"
        )
        answer, chunks = assistant.ask(q, session)
        assert answer, "No answer returned"
        low = answer.lower()
        assert "credit" in low or "semester" in low or "graduat" in low, \
            "Graduation response should mention credits or semesters"

    # ── Scenario C: Student with low GPA ──────────────────────────────────────

    def test_scenario_c_low_gpa(self, assistant, session) -> None:
        """GPA 1.7 → advisor should warn about probation and recommend light load."""
        q = (
            "I'm a CSAI student in semester 4. My GPA is 1.7. "
            "I completed CSAI 101, CSAI 102. "
            "I failed CSAI 201 (Data Structures). "
            "What should I do next semester?"
        )
        answer, chunks = assistant.ask(q, session)
        assert answer, "No answer returned"
        low = answer.lower()
        assert any(w in low for w in ("probation", "risk", "warning", "gpa", "12")), \
            "Low GPA should trigger a risk warning"

    # ── Scenario D: Student with missing prerequisites ─────────────────────────

    def test_scenario_d_missing_prerequisites(self, assistant, session) -> None:
        """Advisor should block ML because Algorithms prerequisite is missing."""
        q = (
            "I'm CSAI CS, semester 3. GPA 3.0. "
            "I completed CSAI 101, CSAI 102, MATH 201. "
            "Can I take Machine Learning (CSAI 253) next semester?"
        )
        answer, chunks = assistant.ask(q, session)
        assert answer, "No answer returned"
        low = answer.lower()
        # Should mention that ML requires intermediate courses
        assert any(w in low for w in ("prerequisite", "prereq", "require", "need", "block", "first")), \
            "Advisor should flag missing prerequisites for ML"

    # ── Scenario E: General campus question passes through to RAG ─────────────

    def test_scenario_e_general_question(self, assistant, session) -> None:
        """General question should still be answered via RAG."""
        q = "What scholarships are available for undergraduate students at Zewail City?"
        answer, chunks = assistant.ask(q, session)
        assert answer, "No answer returned"
        low = answer.lower()
        assert any(w in low for w in ("scholarship", "merit", "financial", "award", "funding")), \
            "Scholarship question should return relevant information"

    # ── Scenario: Same session handles both advisor and general questions ──────

    def test_scenario_unified_session(self, assistant, session) -> None:
        """The SAME conversation handles general AND advisor questions."""
        # First: general question
        q1 = "What is the academic probation policy at Zewail City?"
        a1, _ = assistant.ask(q1, session)
        assert a1

        # Second: profile share
        q2 = "By the way, I'm CSAI semester 3, GPA 2.9, completed CSAI 101 and CSAI 102."
        a2, _ = assistant.ask(q2, session)
        assert a2

        # Third: academic planning (uses remembered profile)
        q3 = "What courses should I take next semester?"
        a3, _ = assistant.ask(q3, session)
        assert a3
        low = a3.lower()
        # Should reference CSAI in some way
        assert "csai" in low or "data structure" in low or "semester" in low, \
            "Advisor should use remembered CSAI profile from earlier in conversation"

        # Verify profile was updated during the session
        profile = session.get_profile()
        assert profile.school == "CSAI" or profile.major != "", \
            "Profile should have been extracted from conversation"
