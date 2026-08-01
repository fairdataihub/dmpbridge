"""Tests for dmpbridge.core.converter.to_structured().

Each test targets one branch of the label-handling state machine, so a
regression in how blocks get merged into sections/questions/answers fails
close to its cause instead of showing up as a mysterious evaluation-score
drift later.
"""
from dmpbridge.core.converter import to_structured


def _block(label: str, text: str) -> dict:
    return {"label": label, "text": text}


def _sections(blocks: list[dict]) -> list[dict]:
    return to_structured(blocks)["narrative"]["template"]["section"]


def _title(blocks: list[dict]) -> str:
    return to_structured(blocks)["narrative"]["template"]["title"]


def test_basic_document_structure():
    blocks = [
        _block("title", "My DMP"),
        _block("section.title", "Types of data"),
        _block("section.description", "Describe your data."),
        _block("question.text", "What data will you collect?"),
        _block("answer.text", "We will collect survey responses."),
    ]
    out = to_structured(blocks)
    template = out["narrative"]["template"]

    assert template["title"] == "My DMP"
    assert len(template["section"]) == 1
    section = template["section"][0]
    assert section["title"] == "Types of data"
    assert section["description"] == "Describe your data."
    assert len(section["question"]) == 1
    question = section["question"][0]
    assert question["text"] == "What data will you collect?"
    assert question["answer"]["json"]["answer"] == "We will collect survey responses."


def test_multiple_title_blocks_before_any_section_merge_with_space():
    blocks = [
        _block("title", "Data Management"),
        _block("title", "and Sharing Plan"),
        _block("section.title", "Roles"),
    ]
    assert _title(blocks) == "Data Management and Sharing Plan"


def test_title_after_sections_started_appends_to_current_answer():
    blocks = [
        _block("title", "Doc Title"),
        _block("section.title", "Sec 1"),
        _block("answer.text", "First answer."),
        _block("title", "Stray heading"),
    ]
    sections = _sections(blocks)
    answer = sections[0]["question"][0]["answer"]["json"]["answer"]
    assert answer == "First answer.\nStray heading"
    # The document title itself is unaffected by a later stray "title" block.
    assert _title(blocks) == "Doc Title"


def test_consecutive_question_text_blocks_merge_into_one_question():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("question.text", "Part A of the question."),
        _block("question.text", "Part B of the question."),
        _block("answer.text", "The answer."),
    ]
    section = _sections(blocks)[0]
    assert len(section["question"]) == 1
    assert section["question"][0]["text"] == "Part A of the question.\nPart B of the question."


def test_new_question_text_after_answer_starts_a_second_question():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("question.text", "Q1"),
        _block("answer.text", "A1"),
        _block("question.text", "Q2"),
        _block("answer.text", "A2"),
    ]
    questions = _sections(blocks)[0]["question"]
    assert len(questions) == 2
    assert questions[0]["text"] == "Q1"
    assert questions[0]["answer"]["json"]["answer"] == "A1"
    assert questions[1]["text"] == "Q2"
    assert questions[1]["answer"]["json"]["answer"] == "A2"


def test_answer_before_any_question_creates_implicit_empty_question():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("answer.text", "Orphan answer."),
    ]
    questions = _sections(blocks)[0]["question"]
    assert len(questions) == 1
    assert questions[0]["text"] == ""
    assert questions[0]["answer"]["json"]["answer"] == "Orphan answer."


def test_answer_text_before_any_section_creates_implicit_section_and_question():
    blocks = [_block("answer.text", "No section or question yet.")]
    sections = _sections(blocks)
    assert len(sections) == 1
    assert sections[0]["title"] == ""
    assert sections[0]["question"][0]["answer"]["json"]["answer"] == "No section or question yet."


def test_section_description_before_any_question_sets_leading_description():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("section.description", "Leading description."),
        _block("section.description", "Second line."),
    ]
    section = _sections(blocks)[0]
    assert section["description"] == "Leading description.\nSecond line."
    assert section["question"] == []


def test_section_description_with_open_question_no_answer_continues_question_text():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("question.text", "Start of question"),
        _block("section.description", "still part of the question"),
    ]
    question = _sections(blocks)[0]["question"][0]
    assert question["text"] == "Start of question\nstill part of the question"
    assert question["answer"]["json"]["answer"] == ""


def test_section_description_with_open_question_with_answer_continues_answer():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("question.text", "Q"),
        _block("answer.text", "A"),
        _block("section.description", "still part of the answer"),
    ]
    question = _sections(blocks)[0]["question"][0]
    assert question["answer"]["json"]["answer"] == "A\nstill part of the answer"


def test_empty_text_blocks_are_skipped():
    blocks = [
        _block("title", "Doc"),
        _block("section.title", "  "),
        _block("section.title", "Sec 1"),
    ]
    # The blank section.title block should not create an empty section.
    assert len(_sections(blocks)) == 1
    assert _sections(blocks)[0]["title"] == "Sec 1"


def test_missing_label_defaults_to_answer_text():
    blocks = [{"text": "unlabeled content"}]
    sections = _sections(blocks)
    assert sections[0]["question"][0]["answer"]["json"]["answer"] == "unlabeled content"


def test_no_blocks_returns_empty_template():
    out = to_structured([])
    template = out["narrative"]["template"]
    assert template["title"] == ""
    assert template["section"] == []


def test_section_order_and_question_order_increment_correctly():
    blocks = [
        _block("section.title", "Sec 1"),
        _block("question.text", "Q1"),
        _block("answer.text", "A1"),
        _block("question.text", "Q2"),
        _block("answer.text", "A2"),
        _block("section.title", "Sec 2"),
        _block("question.text", "Q1-of-sec2"),
    ]
    sections = _sections(blocks)
    assert sections[0]["order"] == 1
    assert sections[1]["order"] == 2
    assert [q["order"] for q in sections[0]["question"]] == [1, 2]
    # Question numbering resets per section.
    assert sections[1]["question"][0]["order"] == 1
