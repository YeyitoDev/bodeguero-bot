from bodeguero.application.prompts import NO_ANSWER
from bodeguero.application.routing import route_for
from bodeguero.domain.models import Completion, Domain, Intent, ModelTier


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


class _Router:
    def __init__(self, intent, spans=()):
        self._route = route_for(intent, 1.0, "test", list(spans))

    def route(self, question):
        return self._route


def test_inventory_lists_the_whole_catalogue(container):
    container.answer_question._router = _Router(Intent.INVENTORY)
    answer = container.answer_question.execute("dame un listado de mi inventario")

    lines = answer.text.splitlines()
    assert len([line for line in lines if not line.endswith(":")]) == 11
    assert lines[:2] == ["Abarrotes:", "Arroz Costeño 1 kg: S/ 4.80 · 20 bolsas [1]"]
    assert "Gaseosas:" in lines
    assert {c.source for c in answer.citations} == {"catalogo/abarrotes.md", "catalogo/bebidas.md"}


def test_category_stock_answers_every_product_in_it(container):
    container.answer_question._router = _Router(Intent.STOCK, [{"PRODUCTO": "gaseosas"}])
    lines = container.answer_question.execute("stock de gaseosas").text.splitlines()

    assert [line.split(":")[0] for line in lines] == ["Inca Kola 500 ml", "Coca-Cola 500 ml", "Inca Kola 1.5 litros"]


def test_several_products_answer_each_and_name_the_missing_ones(container):
    spans = [{"PRODUCTO": "frituras"}, {"PRODUCTO": "huevos"}, {"PRODUCTO": "turrón"}]
    container.answer_question._router = _Router(Intent.STOCK, spans)
    answer = container.answer_question.execute("frituras, huevos, turrón?")

    assert answer.text == "Huevos a granel: te quedan 180 huevos [1]\nNo tienes frituras, turrón en tu catálogo."
    assert answer.grounded
