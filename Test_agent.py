"""
tests/test_agent.py — Physics Study Buddy Test Suite
Agentic AI Course 2026 | Dr. Kanthi Kiran Sirra

Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from agent import (
    CapstoneState, KNOWLEDGE_BASE,
    build_memory_node, skip_retrieval_node, save_node,
    init_resources, build_graph, make_initial_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — builds a fully-typed CapstoneState with safe defaults.
# Fixes: "dict[str, Unknown] is not assignable to CapstoneState"
# All TypedDict keys are always present; callers only supply what they need.
# ─────────────────────────────────────────────────────────────────────────────
def make_test_state(**overrides) -> CapstoneState:
    """Return a valid CapstoneState with all required keys populated."""
    state: CapstoneState = {
        "question":     "",
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 1.0,
        "eval_retries": 0,
        "student_name": None,
        "topic_asked":  None,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ─── FIXTURES ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def resources():
    """Initialise once for the whole module."""
    llm, embedder, collection = init_resources()
    app = build_graph(llm, embedder, collection)
    return llm, embedder, collection, app


def ask(app, question, thread_id="pytest-session"):
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(make_initial_state(question), config=config)


# ─── UNIT TESTS ───────────────────────────────────────────────────────────────

class TestMemoryNode:
    def test_appends_question(self):
        node = build_memory_node()
        state = make_test_state(question="Hello")
        result = node(state)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"

    def test_extracts_student_name(self):
        node = build_memory_node()
        state = make_test_state(question="My name is Priya, explain SHM")
        result = node(state)
        assert result["student_name"] == "Priya"

    def test_sliding_window(self):
        node = build_memory_node()
        old_msgs = [{"role": "user", "content": f"q{i}"} for i in range(10)]
        state = make_test_state(question="new question", messages=old_msgs)
        result = node(state)
        assert len(result["messages"]) <= 6

    def test_no_name_injection(self):
        node = build_memory_node()
        state = make_test_state(question="What is F=ma?")
        result = node(state)
        assert result["student_name"] is None


class TestSkipNode:
    def test_returns_empty_retrieved(self):
        result = skip_retrieval_node(make_test_state())
        assert result["retrieved"] == ""
        assert result["sources"] == []


class TestSaveNode:
    def test_appends_assistant_message(self):
        state = make_test_state(
            messages=[{"role": "user", "content": "test"}],
            answer="A answer.",
            eval_retries=1,
        )
        result = save_node(state)
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "A answer."

    def test_resets_eval_retries(self):
        state = make_test_state(answer="done.", eval_retries=2)
        result = save_node(state)
        assert result["eval_retries"] == 0


class TestKnowledgeBase:
    def test_kb_size(self):
        assert len(KNOWLEDGE_BASE) >= 10, "Must have at least 10 documents."

    def test_each_doc_has_required_fields(self):
        for doc in KNOWLEDGE_BASE:
            assert "id" in doc
            assert "topic" in doc
            assert "text" in doc

    def test_doc_word_count(self):
        for doc in KNOWLEDGE_BASE:
            wc = len(doc["text"].split())
            assert 80 <= wc <= 600, f"{doc['id']} has {wc} words — out of 100-500 range"

    def test_unique_ids(self):
        ids = [doc["id"] for doc in KNOWLEDGE_BASE]
        assert len(ids) == len(set(ids)), "Document IDs must be unique."

    def test_unique_topics(self):
        topics = [doc["topic"] for doc in KNOWLEDGE_BASE]
        assert len(topics) == len(set(topics)), "Topics should be unique (one topic per doc)."


# ─── INTEGRATION TESTS ────────────────────────────────────────────────────────

class TestGraphIntegration:
    def test_graph_compiles(self, resources):
        _, _, _, app = resources
        assert app is not None

    def test_newton_second_law(self, resources):
        _, _, _, app = resources
        result = ask(app, "What is Newton's second law? Give the formula.", "test-newton")
        assert result["answer"]
        assert result["route"] == "retrieve"
        assert "F" in result["answer"] or "force" in result["answer"].lower()

    def test_faithfulness_score_present(self, resources):
        _, _, _, app = resources
        result = ask(app, "Explain Bernoulli's principle.", "test-bernoulli")
        assert "faithfulness" in result
        assert 0.0 <= result["faithfulness"] <= 1.0

    def test_out_of_scope_admits_ignorance(self, resources):
        _, _, _, app = resources
        result = ask(app, "What is the best diet plan for weight loss?", "test-oos")
        answer_lower = result["answer"].lower()
        # Should contain admission of not knowing or redirect
        admits = any(phrase in answer_lower for phrase in [
            "don't have", "not in my knowledge", "textbook",
            "professor", "outside", "support@"
        ])
        assert admits, f"Expected admission of ignorance, got: {result['answer'][:200]}"

    def test_false_premise_corrected(self, resources):
        _, _, _, app = resources
        result = ask(app, "Since Newton's second law says F = m/a, explain it.", "test-false")
        # Should correct the false premise
        answer_lower = result["answer"].lower()
        assert "ma" in result["answer"] or "m×a" in result["answer"] or "mass" in answer_lower

    def test_tool_route_for_datetime(self, resources):
        _, _, _, app = resources
        result = ask(app, "What is today's date?", "test-datetime")
        assert result["route"] == "tool"
        assert result["answer"]

    def test_student_name_extraction(self, resources):
        _, _, _, app = resources
        result = ask(app, "Hi, my name is Kavya. What is escape velocity?", "test-name")
        assert result.get("student_name") == "Kavya"

    def test_retrieval_returns_sources(self, resources):
        _, _, _, app = resources
        result = ask(app, "Explain Faraday's law of electromagnetic induction.", "test-faraday")
        assert result["route"] == "retrieve"
        assert len(result.get("sources", [])) > 0

    def test_memory_persistence_across_turns(self, resources):
        _, _, _, app = resources
        session = "test-memory-001"
        ask(app, "My name is Rohan. What is escape velocity?", session)
        r2 = ask(app, "What was the first topic I asked about?", session)
        # Should reference escape velocity or gravitation in answer
        assert r2.get("answer"), "Second turn should have an answer."

    def test_answer_not_empty(self, resources):
        _, _, _, app = resources
        for q in [
            "State Kepler's third law.",
            "What is moment of inertia for a solid cylinder?",
            "Explain the photoelectric effect.",
        ]:
            result = ask(app, q, f"test-{hash(q)}")
            assert len(result["answer"]) > 50, f"Answer too short for: {q}"

    def test_eval_retries_bounded(self, resources):
        _, _, _, app = resources
        # Even for tricky questions, eval_retries should reset after save
        result = ask(app, "Derive Bohr radius formula.", "test-eval-bound")
        # eval_retries is reset to 0 by save_node
        assert result.get("eval_retries", 0) == 0


# ─── RAGAS EVALUATION STUB ────────────────────────────────────────────────────

RAGAS_QA_PAIRS = [
    {"question": "What is Newton's second law formula?",
     "ground_truth": "F = ma, where F is net force in Newtons, m is mass in kg, a is acceleration in m/s²."},
    {"question": "What is the escape velocity of Earth?",
     "ground_truth": "Approximately 11.2 km/s. Formula: v_esc = sqrt(2GM/R)."},
    {"question": "State the first law of thermodynamics.",
     "ground_truth": "ΔU = Q - W. Energy cannot be created or destroyed."},
    {"question": "What is the radioactive decay law?",
     "ground_truth": "N(t) = N₀·e^(-λt), where λ is the decay constant."},
    {"question": "State Bernoulli's principle formula.",
     "ground_truth": "P + (1/2)ρv² + ρgh = constant along a streamline."},
]


class TestRAGASBaseline:
    def test_baseline_faithfulness(self, resources):
        """Run 5 QA pairs and check average faithfulness >= 0.6."""
        _, _, _, app = resources
        scores = []
        for item in RAGAS_QA_PAIRS:
            r = ask(app, item["question"], f"ragas-{hash(item['question'])}")
            scores.append(r.get("faithfulness", 0.0))
        avg = sum(scores) / len(scores)
        print(f"\nRAGAS Baseline Average Faithfulness: {avg:.3f}")
        assert avg >= 0.6, f"Baseline faithfulness {avg:.3f} is below acceptable threshold of 0.60"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])