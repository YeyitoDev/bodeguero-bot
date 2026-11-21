from bodeguero.application.prompts import NO_ANSWER
from bodeguero.domain.models import Completion, Domain, ModelTier


def test_answer_is_grounded_and_cites_the_right_source(container):
    answer = container.answer_question.execute("¿Cuánto cuesta el aceite Primor?")

    assert answer.grounded
    assert answer.route.domain is Domain.INVENTORY
    assert answer.citations, "a grounded answer must cite at least one chunk"
    assert answer.citations[0].source == "catalogo/abarrotes.md"
    assert "9.90" in answer.text


def test_no_context_means_no_llm_call(container):
    class ExplodingLLM:
        def generate(self, system, user, tier):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called without context")

    container.answer_question._llm = ExplodingLLM()
    answer = container.answer_question.execute("zzzz qqqq xxyyzz")

    assert answer.text == NO_ANSWER
    assert not answer.grounded


def test_invented_citations_are_dropped(container):
    class LyingLLM:
        def generate(self, system, user, tier: ModelTier) -> Completion:
            return Completion(text="Cuesta S/ 1.00 [99]", model="liar")

    container.answer_question._llm = LyingLLM()
    answer = container.answer_question.execute("¿Cuánto vendí la primera semana de octubre?")

    assert answer.citations == []
    assert not answer.grounded


def test_misrouted_question_falls_back_to_all_domains(container):
    # "vendo" makes the router pick SALES, but the answer lives in the catalogue.
    answer = container.answer_question.execute("¿A cuánto vendo el aceite Primor?")

    assert answer.route.domain is Domain.SALES
    assert answer.grounded
    assert answer.citations[0].source == "catalogo/abarrotes.md"
