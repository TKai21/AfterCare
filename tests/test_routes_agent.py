import json

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text, update

from agent import fallback_answer
from app import create_app
from auth import create_doctor
from config import Settings
from druglabels import DEMO_LABELS, LabelClient, fixture_transport
from conftest import auth_headers
from guidance import embed_entry
from knowledge import KnowledgeStore
from llm import LLMClient, LLMError
from models import CareTeamQuestion, ChatMessage, DoctorGuidance, DrugLabel, User
from test_routes_documents import DOC, upload_text

# DOC paragraphs: [1] discharge date, [2] amoxicillin 500 mg, [3] fever above 101 F.
ITEMS = [
    {
        "category": "medication",
        "text": "Take amoxicillin 500 mg three times daily for 7 days.",
        "source_quote": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
        "details": {"name": "Amoxicillin"},
    },
]


def answer(text, citations, kind="answer", care_team_question=None):
    return {"content": json.dumps({
        "kind": kind, "answer": text, "citations": citations,
        "care_team_question": care_team_question,
    })}


async def confirmed(client, fake_llm, headers=None, text=DOC):
    h = headers or await auth_headers(client)
    resp = await upload_text(client, h, fake_llm, text=text, items=ITEMS)
    assert resp.status_code == 200, resp.text
    resp = await client.post("/api/documents/active/confirm", headers=h)
    assert resp.status_code == 200, resp.text
    return h


async def default_doctor(client) -> int:
    """The lowest id doctor, created when there is none yet."""
    async with client.app.state.session_factory() as session:
        found = await session.scalar(
            select(User.id).where(User.role == "doctor").order_by(User.id).limit(1)
        )
        if found is not None:
            return found
        return (await create_doctor(session, "dr.default", "doctor-password-1", "Dr. Default")).id


async def post_ask(client, h, question, doctor_id=None):
    if doctor_id is None:
        doctor_id = await default_doctor(client)
    return await client.post(
        "/api/ask", json={"question": question, "doctor_id": doctor_id}, headers=h
    )


async def get_conversation(client, h, doctor_id=None):
    if doctor_id is None:
        doctor_id = await default_doctor(client)
    return await client.get(f"/api/conversation?doctor_id={doctor_id}", headers=h)


async def care_team(client, h):
    return (await client.get("/api/care-team", headers=h)).json()["questions"]


async def test_ask_returns_cited_answer_and_saves_conversation(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    doctor_id = await default_doctor(client)
    resp = await post_ask(client, h, "  How often do I take amoxicillin?  ")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "kind": "answer",
        "answer": "Take amoxicillin 500 mg three times daily [2].",
        "sources": [{
            "number": 2, "kind": "document", "title": "Your discharge document", "url": None,
            "content": "Take amoxicillin 500 mg by mouth three times daily for 7 days.",
            "guidance_id": None,
        }],
        "flagged": False,
        "emergency": {"terms": [], "banner": False},
        "guidance_ids": [],
        "habit_rule_ids": [],
        "doctor_id": doctor_id,
        "doctor_name": "Dr. Default",
        "confidence": None,
    }
    assert fake_llm.calls[-1][0][-1]["content"].endswith("How often do I take amoxicillin?")

    messages = (await get_conversation(client, h)).json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "How often do I take amoxicillin?"),
        ("assistant", body["answer"]),
    ]
    assert messages[0]["payload"] is None
    assert messages[1]["payload"] == body
    assert all(m["created_at"] for m in messages)
    assert await care_team(client, h) == []


async def test_routes_ask_sets_banner_even_if_model_ignores(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    resp = await post_ask(client, h, "I have chest pain, should I still take amoxicillin?")
    body = resp.json()
    assert body["kind"] == "answer"
    assert body["emergency"] == {"terms": ["chest pain"], "banner": True}
    assert "A possible emergency was detected in the patient's message: chest pain" in (
        fake_llm.calls[-1][0][-1]["content"]
    )


async def test_ask_without_confirmed_doc_409(client, fake_llm):
    h = await auth_headers(client)
    assert (await post_ask(client, h, "Hello?")).status_code == 409
    assert (await upload_text(client, h, fake_llm, items=ITEMS)).status_code == 200
    resp = await post_ask(client, h, "Hello?")
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Confirm your discharge document first."


async def test_conversation_scoped_to_active_document(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    assert (await post_ask(client, h, "How often?")).status_code == 200
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    assert (await post_ask(client, h, "Can I swim?")).status_code == 200
    assert len((await get_conversation(client, h)).json()["messages"]) == 4

    await confirmed(client, fake_llm, headers=h, text=DOC + "\n\nRest at home.")
    assert (await get_conversation(client, h)).json() == {"messages": []}
    fake_llm.script(answer("Rest at home [4].", [4]))
    assert (await post_ask(client, h, "Can I go to work?")).status_code == 200
    messages = fake_llm.calls[-1][0]
    assert [m["role"] for m in messages] == ["system", "user"]  # no old-document history
    # The care-team list keeps the question from the old document.
    assert [q["question"] for q in await care_team(client, h)] == ["Can I swim?"]


async def test_history_sends_last_six_messages(client, fake_llm):
    h = await confirmed(client, fake_llm)
    words = ["alpha", "beta", "gamma", "delta"]  # number words would need grounding
    for word in words:
        fake_llm.script(answer(f"Answer {word} [2].", [2]))
        assert (await post_ask(client, h, f"Question {word}")).status_code == 200
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    await post_ask(client, h, "Last question")
    history = [m["content"] for m in fake_llm.calls[-1][0][1:-1]]
    assert history == [
        f"{prefix} {word}" + ("." if prefix == "Answer" else "")  # markers stripped
        for word in words[1:] for prefix in ("Question", "Answer")
    ]


async def test_not_covered_saves_exactly_one_care_team_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(
        {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "add_care_team_question", "arguments": json.dumps(
                {"question": "Can I swim with my IV site?", "reason": "Not in my document"})},
        }]},
        answer("I added this to your care-team questions.", [], kind="not_covered"),
    )
    body = (await post_ask(client, h, "Can I swim?")).json()
    assert body["kind"] == "not_covered" and body["flagged"] is True
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    await post_ask(client, h, "Can I fly?")
    questions = await care_team(client, h)
    assert [(q["question"], q["reason"], q["source"]) for q in questions] == [
        ("Can I fly?", "Not answered by the discharge document or approved sources.", "agent"),
        ("Can I swim with my IV site?", "Not in my document", "agent"),
    ]


async def test_validation_fallback_saves_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    bad = answer("Take amoxicillin 900 mg [2].", [2])
    fake_llm.script(bad, bad)  # the repair round fails too
    body = (await post_ask(client, h, "What is my dose?")).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Default")
    assert body["sources"] == [] and body["flagged"] is True
    questions = await care_team(client, h)
    assert [(q["question"], q["source"]) for q in questions] == [
        ("What is my dose?", "validation_fallback")
    ]


async def test_ask_blocks_volunteered_naproxen(client, fake_llm):
    h = await confirmed(client, fake_llm)
    bad = answer("I prefer naproxen first [2].", [2])
    fake_llm.script(bad, bad)  # the repair round names it again
    body = (await post_ask(client, h, "What can I take for pain?")).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Default")
    assert body["sources"] == [] and body["flagged"] is True
    assert "naproxen" in fake_llm.calls[-1][0][-1]["content"]
    questions = await care_team(client, h)
    assert [(q["question"], q["source"]) for q in questions] == [
        ("What can I take for pain?", "validation_fallback")
    ]


async def test_ask_allows_a_prescribed_drug_without_a_cited_paragraph_naming_it(client, fake_llm):
    # Amoxicillin is prescribed (item text and name) but the cited paragraph [3] is the
    # fever line; no resolved RxCUIs are needed, and a brand spelling is allowed too.
    # (The fever number stays in a sentence of its own: a sentence that names a
    # drug takes its numbers only from evidence that names it.)
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Call if your fever is above 101 F [3]. Keep taking amoxicillin [3].", [3]))
    body = (await post_ask(client, h, "When should I call?")).json()
    assert body["kind"] == "answer" and "amoxicillin" in body["answer"]
    fake_llm.script(answer("Call if your fever is above 101 F [3]. Keep taking Amoxil [3].", [3]))
    body = (await post_ask(client, h, "When should I call?")).json()
    assert body["kind"] == "answer" and "Amoxil" in body["answer"]


async def test_ask_validates_question_length(client, fake_llm):
    h = await confirmed(client, fake_llm)
    for question in ("", "   \n ", "x" * 2001):
        assert (await post_ask(client, h, question)).status_code == 422
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    assert (await post_ask(client, h, " " + "x" * 2000 + " ")).status_code == 200


async def test_ask_llm_error_502(client, fake_llm):
    h = await confirmed(client, fake_llm)

    async def broken(messages, tools=None, json_mode=False):
        raise LLMError("down")

    fake_llm.chat = broken
    resp = await post_ask(client, h, "How often?")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "The assistant could not answer right now. Please try again."
    assert (await get_conversation(client, h)).json() == {"messages": []}


@pytest_asyncio.fixture
async def app_and_client(fake_llm, tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production"
    )
    app = create_app(settings, llm=fake_llm, store=KnowledgeStore(tmp_path / "i.sqlite3", fake_llm),
                     label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app  # as the shared client fixture does
            yield app, c


async def test_ask_without_llm_503(app_and_client, fake_llm):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    app.state.llm = None
    resp = await post_ask(client, h, "How often?")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "The language model is not configured."


def test_create_app_without_model_has_no_llm_or_store():
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production"
    )
    app = create_app(settings, label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    assert app.state.llm is None and app.state.store is None


def test_create_app_builds_llm_client_and_store_from_settings(tmp_path):
    settings = Settings(
        database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production",
        llm_model="local-model", rag_db_path=str(tmp_path / "index.sqlite3"),
    )
    app = create_app(settings, label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    assert isinstance(app.state.llm, LLMClient)
    assert isinstance(app.state.store, KnowledgeStore)
    assert app.state.store.path == tmp_path / "index.sqlite3"
    assert app.state.foods.lookup("almonds")[0]["name"] == "Almonds"


# ---------- the emergency path never depends on the model (R19) ----------

EMERGENCY_ANSWER_COPY = (
    "If you are having this symptom now, call 911 or follow the emergency instructions "
    "in your discharge document."
)


def assert_emergency_payload(body, terms):
    assert body["kind"] == "emergency"
    assert body["answer"] == EMERGENCY_ANSWER_COPY
    assert body["sources"] == []
    assert body["emergency"] == {"terms": terms, "banner": True}


async def assert_saved(client, h, question, body):
    messages = (await get_conversation(client, h)).json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", question), ("assistant", EMERGENCY_ANSWER_COPY),
    ]
    assert messages[1]["payload"] == body


async def test_ask_without_llm_still_answers_emergency(app_and_client, fake_llm):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    app.state.llm = None
    question = "I have chest pain right now"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_emergency_payload(body, ["chest pain"])
    await assert_saved(client, h, question, body)


async def test_ask_llm_error_still_answers_emergency(client, fake_llm):
    h = await confirmed(client, fake_llm)

    async def broken(messages, tools=None, json_mode=False):
        raise LLMError("down")

    fake_llm.chat = broken
    question = "My chest pain is getting worse"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_emergency_payload(body, ["chest pain"])
    await assert_saved(client, h, question, body)


async def test_emergency_kind_with_uncited_number_becomes_emergency_answer(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Call 911 now. If you have a fever over 104 F, go to the ER.", [],
                           kind="emergency"))
    question = "I have chest pain and a fever"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert_emergency_payload(body, ["chest pain"])
    await assert_saved(client, h, question, body)


async def test_fallback_with_matched_terms_becomes_emergency_answer(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script({"content": "not json"})
    resp = await post_ask(client, h, "I have chest pain")
    assert resp.status_code == 200, resp.text
    assert_emergency_payload(resp.json(), ["chest pain"])


async def test_model_emergency_kind_without_terms_sets_banner(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    body = (await post_ask(client, h, "I feel really strange and scared")).json()
    assert body["kind"] == "emergency"
    assert body["emergency"] == {"terms": [], "banner": True}


async def test_emergency_kind_invalid_without_terms_is_emergency_fallback(client, fake_llm):
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Call the clinic at 555 0142 now.", [], kind="emergency"))
    body = (await post_ask(client, h, "I feel really strange and scared")).json()
    assert body["kind"] == "emergency"
    assert body["answer"] == EMERGENCY_ANSWER_COPY
    assert body["emergency"] == {"terms": [], "banner": True}


# ---------- the doctor clone ----------

MILK_ADVICE = "Yes, amoxicillin can be taken with milk or with food."


def tool(name, args, call_id="c1"):
    return {"content": None, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}


async def make_doctor(client, username="dr.ting", display_name="Dr. Ting Kai") -> int:
    async with client.app.state.session_factory() as session:
        return (await create_doctor(session, username, "doctor-password-1", display_name)).id


async def seed_guidance(client, fake_llm, doctor_id, **kw) -> DoctorGuidance:
    fields = {
        "doctor_id": doctor_id, "question": "Can I take amoxicillin with milk?",
        "answer": MILK_ADVICE, "scope": "general", "drug_tags": ["amoxicillin"],
        "condition_tags": [], **kw,
    }
    async with client.app.state.session_factory() as session:
        entry = DoctorGuidance(**fields)
        assert await embed_entry(fake_llm, entry)
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return entry


def tool_results(fake_llm, call_index):
    return [json.loads(m["content"]) for m in fake_llm.calls[call_index][0] if m["role"] == "tool"]


async def test_ask_cites_eligible_doctor_advice(client, fake_llm):
    h = await confirmed(client, fake_llm)
    entry = await seed_guidance(client, fake_llm, await make_doctor(client))
    date = f"{entry.updated_at:%b} {entry.updated_at.day}, {entry.updated_at.year}"
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "amoxicillin with milk"}),
        answer("Dr. Ting Kai advised that amoxicillin can be taken with milk [4].", [4]),
    )
    body = (await post_ask(client, h, "Can I take amoxicillin with milk?")).json()
    assert body["kind"] == "answer"
    assert body["sources"] == [{
        "number": 4, "kind": "doctor", "title": f"Dr. Ting Kai's advice, {date}", "url": None,
        "content": MILK_ADVICE, "guidance_id": entry.id,
    }]
    assert body["guidance_ids"] == [entry.id]
    assert body["doctor_name"] == "Dr. Ting Kai"
    assert "sent it to Dr. Ting Kai" in fake_llm.calls[-1][0][0]["content"]
    # The entry was prefetched for the question, so the tool hit reuses its number.
    first_user = [m for m in fake_llm.calls[-2][0] if m["role"] == "user"][-1]["content"]
    assert f"[4] My earlier advice, {date} (general advice)" in first_user
    (hits,) = tool_results(fake_llm, -1)
    assert hits["entries"][0]["question_answered"] == "Can I take amoxicillin with milk?"
    async with client.app.state.session_factory() as session:
        saved = await session.scalar(
            select(ChatMessage).where(ChatMessage.role == "assistant")
        )
        assert saved.payload["guidance_ids"] == [entry.id]
    # The doctor's guidance list counts the citation.
    login = await client.post("/api/login", json={"username": "dr.ting",
                                                   "password": "doctor-password-1"})
    dh = {"Authorization": f"Bearer {login.json()['token']}"}
    listed = (await client.get("/api/doctor/guidance", headers=dh)).json()["entries"]
    assert [(e["id"], e["cited_count"]) for e in listed] == [(entry.id, 1)]


async def test_ask_never_retrieves_ineligible_general_advice(client, fake_llm):
    h = await confirmed(client, fake_llm)  # the document never mentions colchicine or gout
    await seed_guidance(
        client, fake_llm, await make_doctor(client),
        question="Can I take amoxicillin with colchicine for gout?",
        answer="Yes, amoxicillin with colchicine is fine.",
        drug_tags=["colchicine"], condition_tags=["gout"],
    )
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "amoxicillin with colchicine for gout"}),
        answer("I could not find this.", [], kind="not_covered"),
    )
    body = (await post_ask(client, h, "Can I take amoxicillin with colchicine?")).json()
    assert tool_results(fake_llm, -1) == [{
        "query": "amoxicillin with colchicine for gout", "entries": [],
        "result": "no matching advice",
    }]
    assert body["guidance_ids"] == []


async def test_doctor_name_falls_back_to_username(client, fake_llm):
    h = await confirmed(client, fake_llm)
    await make_doctor(client, username="dr.a", display_name="Dr. A")
    blank = await make_doctor(client, username="dr.b", display_name="")
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    body = (await post_ask(client, h, "How much?", blank)).json()
    assert body["doctor_name"] == "dr.b"  # its display name is empty
    assert "sent it to dr.b" in fake_llm.calls[-1][0][0]["content"]


async def test_sources_disagree_adds_one_open_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    patient = await auth_user(client, "alice")
    doc_id = (await client.get("/api/documents/active", headers=h)).json()["id"]
    entry = await seed_guidance(
        client, fake_llm, await make_doctor(client), scope="patient", patient_id=patient,
        document_id=doc_id, drug_tags=[], answer="Take amoxicillin 2 times daily.",
        question="How often do I take amoxicillin?",
    )
    disagree = "Should amoxicillin be 3 times daily as in my document, or 2 times daily?"

    def turn():
        fake_llm.script(
            tool("search_doctor_guidance", {"query": "how often amoxicillin"}),
            tool("add_care_team_question", {"question": disagree, "reason": "Sources disagree"},
                 call_id="c2"),
            answer("Your document says to take amoxicillin 500 mg three times daily [2]. "
                   "Dr. Ting Kai's earlier advice differs [4], so I sent this to Dr. Ting Kai.",
                   [2, 4], care_team_question=disagree),
        )

    for _ in range(2):
        turn()
        body = (await post_ask(client, h, "How often do I take amoxicillin?")).json()
        assert body["kind"] == "answer" and body["flagged"] is True
        assert body["guidance_ids"] == [entry.id]
    questions = await care_team(client, h)
    assert [(q["question"], q["reason"]) for q in questions] == [(disagree, "Sources disagree")]


async def test_emergency_backstop_answers_when_guidance_search_fails(client, fake_llm):
    h = await confirmed(client, fake_llm)
    await seed_guidance(client, fake_llm, await make_doctor(client))

    async def broken(texts):
        raise LLMError("embedding down")

    fake_llm.embed = broken
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "amoxicillin chest pain"}),
        {"content": "not json"},
    )
    question = "I have chest pain, can I still take amoxicillin?"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    assert tool_results(fake_llm, -1) == [{"error": "Doctor guidance is not available right now."}]
    assert_emergency_payload(resp.json(), ["chest pain"])


async def auth_user(client, username) -> int:
    async with client.app.state.session_factory() as session:
        return await session.scalar(select(User.id).where(User.username == username))


async def broken_retrieve(session, *args, **kwargs):
    """A real database failure inside guidance retrieval."""
    await session.execute(text("SELECT * FROM no_such_table"))


async def test_guidance_database_error_keeps_the_emergency_answer(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    await make_doctor(client)
    monkeypatch.setattr("routes_agent.retrieve", broken_retrieve)
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "amoxicillin chest pain"}),
        {"content": "not json"},
    )
    question = "I have chest pain, can I still take amoxicillin?"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    assert tool_results(fake_llm, -1) == [{"error": "Doctor guidance is not available right now."}]
    body = resp.json()
    assert_emergency_payload(body, ["chest pain"])
    messages = (await get_conversation(client, h)).json()["messages"]
    assert [m["content"] for m in messages] == [question, EMERGENCY_ANSWER_COPY]


async def test_guidance_database_error_is_a_tool_error_and_turn_completes(
    client, fake_llm, monkeypatch
):
    h = await confirmed(client, fake_llm)
    await make_doctor(client)
    monkeypatch.setattr("routes_agent.retrieve", broken_retrieve)
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "swimming"}),
        tool("search_doctor_guidance", {"query": "swimming again"}, call_id="c2"),
        answer("I could not find this.", [], kind="not_covered"),
    )
    resp = await post_ask(client, h, "Can I swim?")
    assert resp.status_code == 200, resp.text
    unavailable = {"error": "Doctor guidance is not available right now."}
    assert tool_results(fake_llm, -1) == [unavailable, unavailable]
    assert [q["question"] for q in await care_team(client, h)] == ["Can I swim?"]


async def test_clone_never_sees_another_doctors_advice(client, fake_llm):
    h = await confirmed(client, fake_llm)
    first = await make_doctor(client, username="dr.a", display_name="Dr. A")
    second = await make_doctor(client, username="dr.b", display_name="Dr. B")
    await seed_guidance(client, fake_llm, second)
    fake_llm.script(
        tool("search_doctor_guidance", {"query": "amoxicillin with milk"}),
        answer("I could not find this.", [], kind="not_covered"),
    )
    body = (await post_ask(client, h, "Can I take amoxicillin with milk?", first)).json()
    first_user = [m for m in fake_llm.calls[-2][0] if m["role"] == "user"][-1]["content"]
    assert MILK_ADVICE not in first_user and "none matched this question" in first_user
    assert tool_results(fake_llm, -1) == [{
        "query": "amoxicillin with milk", "entries": [], "result": "no matching advice",
    }]
    assert body["doctor_name"] == "Dr. A" and body["guidance_ids"] == []


# ---------- one conversation per doctor ----------

async def test_ask_requires_a_doctor(client, fake_llm):
    h = await confirmed(client, fake_llm)
    patient = await auth_user(client, "alice")
    doctor = await default_doctor(client)
    calls_before = len(fake_llm.calls)
    resp = await client.post("/api/ask", json={"question": "How much?"}, headers=h)
    assert resp.status_code == 422
    for bad in (patient, doctor + 999):
        resp = await client.post("/api/ask", json={"question": "How much?", "doctor_id": bad},
                                 headers=h)
        assert resp.status_code == 422
        assert resp.json()["detail"] == "Choose one of the doctors."
    assert (await client.get("/api/conversation", headers=h)).status_code == 422
    assert (await get_conversation(client, h, patient)).status_code == 422
    assert len(fake_llm.calls) == calls_before  # no model work for a bad doctor


async def test_emergency_with_a_bad_doctor_still_gets_the_emergency_answer(client, fake_llm):
    # The backstop never depends on the doctor id: an emergency message gets the
    # fixed emergency answer, and nothing is saved under a doctor that does not exist.
    h = await confirmed(client, fake_llm)
    patient = await auth_user(client, "alice")
    doctor = await default_doctor(client)
    calls_before = len(fake_llm.calls)
    for body in ({"question": "I have chest pain", "doctor_id": patient},
                 {"question": "I have chest pain", "doctor_id": doctor + 999},
                 {"question": "I have chest pain"}):
        resp = await client.post("/api/ask", json=body, headers=h)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert_emergency_payload(payload, ["chest pain"])
        assert payload["doctor_id"] is None and payload["doctor_name"] is None
        assert payload["flagged"] is False
    assert len(fake_llm.calls) == calls_before
    async with client.app.state.session_factory() as session:
        assert list(await session.scalars(select(ChatMessage))) == []
    assert await care_team(client, h) == []


async def test_history_is_per_doctor(client, fake_llm):
    h = await confirmed(client, fake_llm)
    a = await make_doctor(client, username="dr.a", display_name="Dr. A")
    b = await make_doctor(client, username="dr.b", display_name="Dr. B")
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    assert (await post_ask(client, h, "Question for A", a)).status_code == 200
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    body = (await post_ask(client, h, "Question for B", b)).json()
    assert body["doctor_id"] == b
    assert [m["role"] for m in fake_llm.calls[-1][0]] == ["system", "user"]  # no A messages
    only_a = (await get_conversation(client, h, a)).json()["messages"]
    assert [(m["role"], m["content"]) for m in only_a] == [
        ("user", "Question for A"), ("assistant", "Take amoxicillin 500 mg [2].")
    ]
    only_b = (await get_conversation(client, h, b)).json()["messages"]
    assert only_b[0]["content"] == "Question for B" and len(only_b) == 2


async def test_not_covered_goes_to_that_doctor(client, fake_llm):
    h = await confirmed(client, fake_llm)
    await make_doctor(client, username="dr.a", display_name="Dr. A")
    b = await make_doctor(client, username="dr.b", display_name="Dr. B")
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    assert (await post_ask(client, h, "Can I swim?", b)).status_code == 200

    async def queue(username):
        login = await client.post("/api/login", json={"username": username,
                                                       "password": "doctor-password-1"})
        dh = {"Authorization": f"Bearer {login.json()['token']}"}
        return (await client.get("/api/doctor/questions", headers=dh)).json()["open"]

    assert await queue("dr.a") == []
    (mine,) = await queue("dr.b")
    assert mine["question"] == "Can I swim?" and mine["doctor_id"] == b
    # Doctor B sees the conversation with B's clone; nothing of any other doctor's.
    assert [m["content"] for m in mine["context"]["conversation"]] == [
        "Can I swim?", fallback_answer("Dr. B"),
    ]


async def test_queue_context_never_shows_another_doctors_conversation(client, fake_llm):
    h = await confirmed(client, fake_llm)
    a = await make_doctor(client, username="dr.a", display_name="Dr. A")
    b = await make_doctor(client, username="dr.b", display_name="Dr. B")
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    assert (await post_ask(client, h, "Private question for A", a)).status_code == 200
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    assert (await post_ask(client, h, "Can I swim?", b)).status_code == 200
    login = await client.post("/api/login", json={"username": "dr.a",
                                                   "password": "doctor-password-1"})
    ah = {"Authorization": f"Bearer {login.json()['token']}"}
    login = await client.post("/api/login", json={"username": "dr.b",
                                                   "password": "doctor-password-1"})
    bh = {"Authorization": f"Bearer {login.json()['token']}"}
    # B's question: B's conversation only, nothing from A's clone.
    (qb,) = (await client.get("/api/doctor/questions", headers=bh)).json()["open"]
    texts = [m["content"] for m in qb["context"]["conversation"]]
    assert "Private question for A" not in texts and "Can I swim?" in texts
    assert (await client.get("/api/doctor/questions", headers=ah)).json()["open"] == []
    # An unassigned question shows no conversation to anyone.
    async with client.app.state.session_factory() as session:
        from models import CareTeamQuestion
        doc_id = (await session.scalar(select(ChatMessage.document_id).limit(1)))
        owner = await auth_user(client, "alice")
        session.add(CareTeamQuestion(user_id=owner, document_id=doc_id, question="Loose?",
                                     reason="r", source="agent", doctor_id=None))
        await session.commit()
    for dh in (ah, bh):
        open_ = (await client.get("/api/doctor/questions", headers=dh)).json()["open"]
        (loose,) = [q for q in open_ if q["question"] == "Loose?"]
        assert loose["context"]["conversation"] == []


async def test_ask_uses_only_that_doctors_guidance(client, fake_llm):
    h = await confirmed(client, fake_llm)
    a = await make_doctor(client, username="dr.a", display_name="Dr. A")
    b = await make_doctor(client, username="dr.b", display_name="Dr. B")
    await seed_guidance(client, fake_llm, a, answer="A says milk is fine.")
    entry_b = await seed_guidance(client, fake_llm, b, answer="B says food is fine.")
    fake_llm.script(answer("B advised that food is fine [4].", [4]))
    body = (await post_ask(client, h, "Can I take amoxicillin with milk?", b)).json()
    first_user = [m for m in fake_llm.calls[-1][0] if m["role"] == "user"][-1]["content"]
    assert "B says food is fine." in first_user and "A says milk is fine." not in first_user
    assert body["guidance_ids"] == [entry_b.id]


async def test_prefetch_failure_still_answers_and_saves_the_turn(client, fake_llm, monkeypatch):
    # A database error in the prefetch rolls the session back before the agent
    # has read the document or items; the turn must still answer and be saved.
    h = await confirmed(client, fake_llm)
    await make_doctor(client)
    searched = []

    async def failing(session, *args, **kwargs):
        searched.append(args[-1])
        await broken_retrieve(session)

    monkeypatch.setattr("routes_agent.retrieve", failing)
    fake_llm.script(answer("Take amoxicillin 500 mg three times daily [2].", [2]))
    question = "How often do I take amoxicillin?"
    resp = await post_ask(client, h, question)
    assert resp.status_code == 200, resp.text
    assert searched == [question]  # the prefetch ran and failed
    body = resp.json()
    assert body["kind"] == "answer" and body["sources"][0]["number"] == 2
    first_user = [m for m in fake_llm.calls[-1][0] if m["role"] == "user"][-1]["content"]
    assert "Take amoxicillin 500 mg by mouth" in first_user  # the document reached the model
    messages = (await get_conversation(client, h)).json()["messages"]
    assert [m["content"] for m in messages] == [question, body["answer"]]


async def test_emergency_skips_the_guidance_prefetch(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    await make_doctor(client)
    searched = []

    async def spy(*args, **kwargs):
        searched.append(args[-2:])
        return []

    monkeypatch.setattr("routes_agent.retrieve", spy)
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    resp = await post_ask(client, h, "I have chest pain")
    assert resp.status_code == 200, resp.text
    assert searched == []  # no retrieval before the first model call
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    await post_ask(client, h, "How much amoxicillin?")
    assert len(searched) == 1  # an ordinary question is still prefetched


async def test_payload_names_the_doctor(client, fake_llm):
    h = await confirmed(client, fake_llm)
    await make_doctor(client, username="dr.a", display_name="Dr. A")
    b = await make_doctor(client, username="dr.b", display_name="Dr. B")
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    body = (await post_ask(client, h, "How much?", b)).json()
    assert (body["doctor_id"], body["doctor_name"]) == (b, "Dr. B")
    saved = (await get_conversation(client, h, b)).json()["messages"][1]
    assert saved["payload"]["doctor_id"] == b


class RecordingStore:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, limit=5, rxcuis=()):
        self.calls.append(frozenset(rxcuis))
        return []


async def test_ask_uses_active_document_rxcuis(client, fake_llm):
    stub = RecordingStore()
    client.app.state.store = stub
    # A confirmed document whose drug did not resolve unlocks no labels.
    h = await confirmed(client, fake_llm)
    fake_llm.script(
        tool("search_education_library", {"query": "amoxicillin"}),
        answer("Take amoxicillin 500 mg three times daily [2].", [2]),
    )
    assert (await post_ask(client, h, "What is amoxicillin?")).status_code == 200
    assert stub.calls == [frozenset()]
    stub.calls.clear()

    colchicine = {"category": "medication", "text": "Take colchicine 0.6 mg",
                  "source_quote": "Take colchicine 0.6 mg", "details": {"name": "Colchicine",
                                                                            "dose": "0.6 mg"}}
    fake_llm.script({"content": json.dumps({"discharge_date": "2026-03-02", "items": [colchicine]})})
    h2 = await auth_headers(client, "gout-patient")
    up = await client.post("/api/documents", data={"text": "Discharge date: 2026-03-02\n\nTake colchicine 0.6 mg"},
                           headers=h2)
    assert up.status_code == 200, up.text
    assert (await client.post("/api/documents/active/confirm", json={}, headers=h2)).status_code == 200
    for status, unlocked in (("pending", frozenset()), ("approved", frozenset({"2683"}))):
        async with client.app.state.session_factory() as session:
            await session.execute(update(DrugLabel).where(DrugLabel.rxcui == "2683").values(status=status))
            await session.commit()
        stub.calls.clear()
        fake_llm.script(
            tool("search_education_library", {"query": "colchicine"}),
            answer("Take colchicine 0.6 mg [2].", [2]),
        )
        assert (await post_ask(client, h2, "What is colchicine?")).status_code == 200
        assert stub.calls == [unlocked], status


async def seed_habit(client, entry, statement, quote, habit_type="ask", **kw):
    from models import GuidanceRule
    async with client.app.state.session_factory() as session:
        session.add(GuidanceRule(guidance_id=entry.id, doctor_id=entry.doctor_id, position=0,
                                 statement=statement, quote=quote, kind="habit",
                                 habit_type=habit_type, **kw))
        await session.commit()


def first_user_message(fake_llm, call_index=-1):
    """The user message of a main-model call (the last one by default; the reply
    judge's calls are skipped)."""
    import agent
    calls = [c for c in fake_llm.calls if c[0][0]["content"] != agent.REPLY_JUDGE_PROMPT]
    return [m for m in calls[call_index][0] if m["role"] == "user"][-1]["content"]


async def test_habit_of_another_patients_entry_reaches_this_patient(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await make_doctor(client)
    other = await seed_guidance(client, fake_llm, doctor_id, scope="patient", patient_id=None,
                                question="Q", answer="Take 500 mg twice a day.")
    await seed_habit(client, other, "Asks how the pain is", "How is the pain today?")
    fake_llm.script(answer("I do not know.", [], kind="not_covered", care_team_question="q"),
                    {"content": '{"reports_symptom": true, "ask": []}'})  # the ask-first picker finds none fitting
    await post_ask(client, h, "Can I eat rice?", doctor_id=doctor_id)
    text = first_user_message(fake_llm, -2)  # the clone's call; the picker's is last
    assert "How I usually respond" in text and "How is the pain today?" in text
    assert "500 mg twice" not in text


async def test_inactive_entry_habits_are_left_out(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await make_doctor(client)
    gone = await seed_guidance(client, fake_llm, doctor_id, active=False)
    await seed_habit(client, gone, "Asks about sleep", "How did you sleep?")
    fake_llm.script(answer("I do not know.", [], kind="not_covered", care_team_question="q"))
    await post_ask(client, h, "Can I eat rice?", doctor_id=doctor_id)
    assert "How did you sleep?" not in first_user_message(fake_llm)


async def test_emergency_loads_no_habits(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id = await make_doctor(client)
    loaded = []

    async def spy(*args, **kwargs):
        loaded.append(args)
        return []

    monkeypatch.setattr("routes_agent.doctor_habits", spy)
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    resp = await post_ask(client, h, "I have chest pain", doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    assert loaded == []


async def test_habit_load_failure_does_not_break_the_turn(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id = await make_doctor(client)

    async def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("routes_agent.doctor_habits", broken)
    fake_llm.script(answer("I do not know.", [], kind="not_covered", care_team_question="q"))
    resp = await post_ask(client, h, "Can I eat rice?", doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    assert "How I usually respond" not in first_user_message(fake_llm)


# ---------- ask-first turns: the waiting question (Task 11) ----------

ASK_QUESTIONS = "Do you have a fever? Have you thrown up?"
CLOSE_TEXT = "Keep your follow-up visit and call me if it changes."
NEEDS_DOCTOR = "Needs doctor after questions"
NOT_ANSWERED = "Patient did not answer the clone's questions"
ORIGINAL = "My incision is a bit red."
REPLY_MARK = "The patient is replying to my questions"


def final(kind, text, citations, **extra):
    body = {"kind": kind, "answer": text, "citations": citations, "care_team_question": None}
    body.update(extra)
    return {"content": json.dumps(body)}


class Reply(dict):
    """A main-model reply on an intake turn. The reply judge, not the model,
    reports the reply fields, so the reply carries the judge's answer that the
    scripting helper queues right after it."""
    judge: dict


def reply_final(kind, text, citations, replies=True, signs=(), concerns=False, close=5,
                again=False):
    """The main model's message (it no longer carries reply fields) plus the judge
    reading that follows it: `signs` are the judge's warning_signs. With `again`
    the judge left terms out and is asked once more about them; it leaves them
    out again."""
    message = Reply(final(kind, text, citations))
    message.judge = {"content": json.dumps({
        "replies_to_questions": replies, "warning_signs": list(signs),
        "other_concerns": concerns, "close": close})}
    message.again = message.judge if again else None
    return message


@pytest.fixture(autouse=True)
def judge_follows_the_reply(fake_llm):
    original = fake_llm.script

    def script(*messages):
        queued = []
        for message in messages:
            queued.append(message)
            if isinstance(message, Reply):
                queued.append(message.judge)
                if getattr(message, "again", None) is not None:
                    queued.append(message.again)
        original(*queued)

    fake_llm.script = script


def absent(*terms):
    return [{"term": t, "status": "absent"} for t in terms]


def gate_on(client, fake_llm, score=92):
    import confidence
    client.app.state.clone_threshold = 0.8
    fake_llm.route(confidence.VERIFY_PROMPT,
                   lambda m: {"content": json.dumps({"score": score, "unsupported": []})})


def gate_calls(fake_llm) -> int:
    import confidence
    return sum(1 for c in fake_llm.routed_calls if c[0][0]["content"] == confidence.VERIFY_PROMPT)


async def ask_first_doctor(client, fake_llm, username="dr.ting", display_name="Dr. Ting Kai"):
    """A doctor with one ask, one close and one warning habit. Evidence numbers:
    the document's paragraphs are [1]-[3], then ask [4], close [5], warning [6]."""
    doctor_id = await make_doctor(client, username, display_name)
    entry = await seed_guidance(client, fake_llm, doctor_id)
    await seed_habit(client, entry, "Asks about fever and vomiting", ASK_QUESTIONS)
    await seed_habit(client, entry, "Closes with the follow-up visit", CLOSE_TEXT, "close")
    await seed_habit(client, entry, "Wants to hear about fever or vomiting",
                     "Tell me about any fever or if you threw up.", "warning",
                     warning_terms=["fever", "threw up"])
    return doctor_id


async def start_clarify(client, fake_llm, h, doctor_id):
    fake_llm.script(final("clarify", ASK_QUESTIONS + " [4]", [4]))
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify", body
    return body


async def all_questions(client):
    from models import CareTeamQuestion
    async with client.app.state.session_factory() as session:
        return list(await session.scalars(select(CareTeamQuestion).order_by(CareTeamQuestion.id)))


async def clarify_message(client, doctor_id):
    async with client.app.state.session_factory() as session:
        return [m for m in await session.scalars(
            select(ChatMessage).where(ChatMessage.doctor_id == doctor_id).order_by(ChatMessage.id))
            if (m.payload or {}).get("kind") == "clarify"][-1]


async def test_clarify_creates_a_waiting_question(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)
    assert body["pending"] is True and body["flagged"] is False
    (q,) = await all_questions(client)
    assert body["waiting_question_id"] == q.id
    assert (q.status, q.question, q.asked, q.doctor_id, q.source) == (
        "waiting_on_patient", ORIGINAL, ASK_QUESTIONS, doctor_id, "agent")
    assert q.patient_reply is None and q.stopped_by is None
    assert gate_calls(fake_llm) == 0  # the confidence gate never scores a clarify
    # Not yet the care team's: the patient's own list leaves it out.
    assert await care_team(client, h) == []
    assert (await clarify_message(client, doctor_id)).payload["pending"] is True


async def test_clarify_with_a_flag_adds_no_second_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    fake_llm.script(
        tool("add_care_team_question", {"question": "Red incision", "reason": "Symptom"}),
        final("clarify", ASK_QUESTIONS + " [4]", [4]),
    )
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify" and body["flagged"] is False
    assert [q.status for q in await all_questions(client)] == ["waiting_on_patient"]


async def test_not_covered_first_turn_with_fitting_ask_habit_asks_instead_of_handing_off(
        client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    fake_llm.script(
        tool("add_care_team_question", {"question": "What should I do?", "reason": "Not covered"}),
        final("not_covered", "x", []),
        {"content": json.dumps({"reports_symptom": True, "ask": [4]})},
    )
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify" and body["pending"] is True and body["flagged"] is False
    assert ASK_QUESTIONS in body["answer"]
    assert [q.status for q in await all_questions(client)] == ["waiting_on_patient"]


async def test_first_turn_with_a_doctor_warning_term_is_handed_off_not_asked(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    fake_llm.script(final("not_covered", "x", []))  # no picker reply is scripted
    body = (await post_ask(client, h, "My incision is red and I have a fever.",
                           doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered"
    assert [q.status for q in await all_questions(client)] == ["open"]


async def test_benign_reply_closes_and_resolves(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    reply = "No fever, never threw up."
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and body["answer"] == CLOSE_TEXT + " [5]", body
    assert REPLY_MARK in first_user_message(fake_llm)
    assert gate_calls(fake_llm) == 1  # a close is scored like an answer
    (q,) = await all_questions(client)
    assert (q.status, q.answer, q.patient_reply) == ("resolved", CLOSE_TEXT, reply)
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


def judge_json(status="absent", close=5, concerns=False):
    return {"content": json.dumps({
        "replies_to_questions": True, "other_concerns": concerns, "close": close,
        "warning_signs": [{"term": t, "status": status} for t in ("fever", "threw up")]})}


@pytest.mark.parametrize("model", ["handoff", "fallback"])
async def test_reply_judge_close_resolves_after_the_model_hands_off(client, fake_llm, model):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    if model == "handoff":
        fake_llm.script(final("not_covered", "x", []), judge_json())
    else:  # the model's answer fails validation twice
        bad = final("answer", "Drink 9 liters [2].", [2])
        fake_llm.script(bad, bad, judge_json())
    reply = "No fever, never threw up."
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and body["answer"] == CLOSE_TEXT + " [5]", body
    (q,) = await all_questions(client)
    assert (q.status, q.answer, q.patient_reply) == ("resolved", CLOSE_TEXT, reply)
    assert q.intake == [{"asked": ASK_QUESTIONS, "reply": reply}]


async def test_the_judges_close_is_not_scored_by_the_gate(client, fake_llm):
    # The doctor's own closing words, built by code: a failing gate must not
    # turn them into a handoff (seen end to end: "Low confidence").
    gate_on(client, fake_llm, score=10)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(final("not_covered", "x", []), judge_json())
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and gate_calls(fake_llm) == 0
    # a close the model wrote itself is still scored
    h2 = await confirmed(client, fake_llm, headers=await auth_headers(client, "p.model"))
    await start_clarify(client, fake_llm, h2, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5], signs=absent("fever", "threw up")))
    body = (await post_ask(client, h2, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] != "close" and gate_calls(fake_llm) == 1


async def test_reply_judge_with_a_present_term_keeps_the_handoff(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(final("not_covered", "x", []), judge_json(status="unclear"))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered"
    (q,) = await all_questions(client)
    assert q.status == "open"


async def test_warning_reply_hands_off_with_summary(client, fake_llm):
    # Review Focus 1: the denied fever does not hide the vomiting.
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    reply = "no fever, but I threw up twice"
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=[{"term": "fever", "status": "absent"},
                                       {"term": "threw up", "status": "absent"}]))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["flagged"] is True
    assert body["answer"] == fallback_answer("Dr. Ting Kai") and body["sources"] == []
    assert gate_calls(fake_llm) == 0  # blocked before any scoring
    (q,) = await all_questions(client)  # never a second question
    assert (q.status, q.reason, q.question, q.asked, q.patient_reply, q.stopped_by) == (
        "open", NEEDS_DOCTOR, ORIGINAL, ASK_QUESTIONS, reply, "Warning sign: threw up")
    (seen,) = await care_team(client, h)
    assert (seen["asked"], seen["patient_reply"], seen["stopped_by"]) == (
        ASK_QUESTIONS, reply, "Warning sign: threw up")
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


async def test_low_confidence_close_hands_off(client, fake_llm):
    gate_on(client, fake_llm, score=30)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by) == ("open", NEEDS_DOCTOR, "Low confidence")


async def test_not_covered_reply_hands_off(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("not_covered", "I can't say.", [], close=None,
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "Nope, neither.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by) == ("open", NEEDS_DOCTOR, "Clone could not answer")


async def test_reply_answer_resolves_the_waiting_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    text = "Take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(reply_final("answer", text, [2], signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text
    (q,) = await all_questions(client)
    assert (q.status, q.answer) == ("resolved", "Take amoxicillin 500 mg three times daily.")


async def test_new_question_hands_off_the_symptom(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    text = "Take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(reply_final("answer", text, [2], replies=False,
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "How often do I take amoxicillin?",
                           doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text  # the new question is answered
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by, q.patient_reply) == (
        "open", NOT_ANSWERED, NOT_ANSWERED, "How often do I take amoxicillin?")
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


async def test_other_doctors_tab_is_not_the_reply(client, fake_llm):
    # Review Focus 4: the clarify belongs to one patient-doctor conversation.
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    other_id = await make_doctor(client, "dr.other", "Dr. Other")
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(answer("I could not find this.", [], kind="not_covered"))
    await post_ask(client, h, "No fever, never threw up.", doctor_id=other_id)
    assert REPLY_MARK not in first_user_message(fake_llm)
    waiting, other = await all_questions(client)
    assert (waiting.status, waiting.patient_reply) == ("waiting_on_patient", None)
    assert (other.doctor_id, other.status) == (other_id, "open")
    assert (await clarify_message(client, doctor_id)).payload["pending"] is True
    # The next message in the first doctor's tab is still the reply.
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and REPLY_MARK in first_user_message(fake_llm)


def test_settle_reply_with_emergency_terms_is_always_the_emergency_answer():
    # The rule itself: on a reply turn with emergency terms, any kind the model
    # gave other than "emergency" becomes the fixed emergency answer.
    from agent import AgentResult, EMERGENCY_ANSWER
    from routes_agent import settle_reply
    for kind in ("close", "answer", "not_covered", "out_of_scope"):
        result = AgentResult(kind=kind, answer="Keep your follow-up visit.", sources=[],
                             flagged=True, care_team_question="q", replies_to_questions=True)
        delivered, settle = settle_reply(result, ["chest pain"], None, True, "Dr. A")
        assert (delivered.kind, delivered.answer, delivered.flagged,
                delivered.care_team_question) == ("emergency", EMERGENCY_ANSWER, False, None), kind
        assert (settle.status, settle.reason, settle.stopped_by) == (
            "open", NEEDS_DOCTOR, "Emergency terms"), kind
    own = AgentResult(kind="emergency", answer="Call 911 now.", sources=[], flagged=False,
                      care_team_question=None)
    delivered, settle = settle_reply(own, ["chest pain"], None, True, "Dr. A")
    assert delivered.answer == "Call 911 now." and settle.stopped_by == "Emergency terms"


async def test_emergency_in_reply_gives_emergency_answer_and_opens_question(client, fake_llm):
    # Review Focus 5. The model answers with a valid non-emergency kind (the
    # document paragraph [2] is evidence even on an emergency turn): the server
    # still gives the fixed emergency answer.
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    reply = "No fever, but now I have chest pain"
    fake_llm.script(reply_final("answer", "Take amoxicillin 500 mg three times daily [2].", [2],
                                signs=absent("fever")))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "emergency" and body["answer"] == EMERGENCY_ANSWER_COPY
    assert body["emergency"]["banner"] is True
    assert gate_calls(fake_llm) == 0
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by, q.patient_reply) == (
        "open", NEEDS_DOCTOR, "Emergency terms", reply)
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


async def test_emergency_reply_without_a_model_still_opens_the_question(app_and_client, fake_llm):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    app.state.llm = None  # the model is not configured
    body = (await post_ask(client, h, "I have chest pain", doctor_id=doctor_id)).json()
    assert_emergency_payload(body, ["chest pain"])
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Emergency terms")


async def test_new_question_keeps_its_own_gate(client, fake_llm):
    gate_on(client, fake_llm, score=30)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("answer", "Take amoxicillin 500 mg three times daily [2].", [2],
                                replies=False, signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "How often do I take it?", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    waiting, new = await all_questions(client)
    assert (waiting.status, waiting.stopped_by) == ("open", NOT_ANSWERED)
    assert (new.question, new.status) == ("How often do I take it?", "open")


async def test_close_checks_every_warning_habit_not_the_deduplicated_list(client, fake_llm):
    # Two live warning habits with the same statement: doctor_habits keeps one,
    # but the close check must see both entries' terms.
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    async with client.app.state.session_factory() as session:
        entry = await session.scalar(
            select(DoctorGuidance).where(DoctorGuidance.doctor_id == doctor_id))
    # After the ask [4] and close [5] habits, so the evidence numbers hold.
    await seed_habit(client, entry, "Wants to hear about a fever", "Tell me about a fever.",
                     "warning", warning_terms=["fever"])
    await seed_habit(client, entry, "Wants to hear about a fever", "Tell me about chills.",
                     "warning", warning_terms=["chills"])
    await start_clarify(client, fake_llm, h, doctor_id)
    reply = "I have had some chills today"
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up", "chills")))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Warning sign: chills")


async def test_close_is_blocked_when_warning_terms_cannot_be_read(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)

    async def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("routes_agent.doctor_warning_terms", broken)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "The doctor's warning signs could not be loaded")


async def test_answer_reply_with_a_warning_sign_reaches_the_doctor(client, fake_llm):
    # An answer is delivered, but the close checks still decide the symptom's fate.
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    text = "Take amoxicillin 500 mg three times daily [2]."
    reply = "Yes I have a fever and I threw up three times."
    fake_llm.script(reply_final("answer", text, [2],
                                signs=[{"term": "fever", "status": "present"},
                                       {"term": "threw up", "status": "present"}]))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by, q.patient_reply, q.answer) == (
        "open", NEEDS_DOCTOR, "Warning sign: threw up", reply, None)


async def set_waiting(client, **values):
    from models import CareTeamQuestion
    async with client.app.state.session_factory() as session:
        await session.execute(update(CareTeamQuestion).values(**values))
        await session.commit()


async def test_reply_after_the_doctor_answered_is_never_lost(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await set_waiting(client, status="answered", answer="Please send a photo.")
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    reply = "I threw up all night."
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    old, new = await all_questions(client)
    assert (old.status, old.answer, old.patient_reply) == ("answered", "Please send a photo.", None)
    assert (new.status, new.doctor_id, new.question, new.asked, new.patient_reply,
            new.reason, new.stopped_by) == (
        "open", doctor_id, ORIGINAL, ASK_QUESTIONS, reply, NEEDS_DOCTOR, "Warning sign: threw up")
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


async def test_patient_cannot_resolve_a_waiting_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)
    resp = await client.post(f"/api/care-team/{body['waiting_question_id']}/resolve", headers=h)
    assert resp.status_code == 409
    (q,) = await all_questions(client)
    assert q.status == "waiting_on_patient"


async def test_a_late_close_never_overwrites_a_handoff(client, fake_llm):
    # Two replies that both read the pending clarify: only the first settles it.
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    await post_ask(client, h, "no fever, but I threw up twice", doctor_id=doctor_id)
    message = await clarify_message(client, doctor_id)
    async with client.app.state.session_factory() as session:  # the second turn's stale read
        await session.execute(update(ChatMessage).where(ChatMessage.id == message.id)
                              .values(payload={**message.payload, "pending": True}))
        await session.execute(delete(ChatMessage).where(ChatMessage.id > message.id))
        # ... which read the rounds before the first turn saved its reply
        await session.execute(update(CareTeamQuestion).values(intake=None))
        await session.commit()
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close"
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by, q.answer) == ("open", "Warning sign: threw up", None)


async def test_reply_gate_scores_the_original_question_and_the_reply(client, fake_llm):
    import confidence
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)
    (verify,) = [c for c in fake_llm.routed_calls if c[0][0]["content"] == confidence.VERIFY_PROMPT]
    scored = verify[0][1]["content"]
    assert ORIGINAL in scored and "No fever, never threw up." in scored


async def test_pending_lookup_failure_keeps_the_emergency_answer(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)

    async def broken(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("routes_agent.pending_clarify", broken)
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    resp = await post_ask(client, h, "I have chest pain", doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "emergency"


async def test_a_term_the_model_left_out_hands_off(client, fake_llm):
    reply = "No fever. I vomited twice though."
    for kind, text, cites in (("close", CLOSE_TEXT + " [5]", [5]),
                              ("answer", "Take amoxicillin 500 mg three times daily [2].", [2])):
        h = await confirmed(client, fake_llm, headers=await auth_headers(client, f"p.{kind}"))
        doctor_id = await ask_first_doctor(client, fake_llm, f"dr.{kind}", f"Dr. {kind}")
        await start_clarify(client, fake_llm, h, doctor_id)
        fake_llm.script(reply_final(kind, text, cites, signs=absent("fever"), again=True))
        body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
        q = [q for q in await all_questions(client) if q.doctor_id == doctor_id][0]
        assert (q.status, q.reason, q.stopped_by, q.patient_reply) == (
            "open", NEEDS_DOCTOR, "Warning sign: threw up", reply), kind
        if kind == "close":
            assert body["answer"] == fallback_answer(f"Dr. {kind}")
        else:
            assert body["answer"] == text  # the answer is delivered, the case is not settled


async def test_empty_warning_signs_block_the_close(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5], signs=[], again=True))
    body = (await post_ask(client, h, "It looks fine.", doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Warning sign: fever")


async def test_a_new_question_names_the_blocker_when_one_applies(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("answer", "Take amoxicillin 500 mg three times daily [2].", [2],
                                replies=False, signs=absent("fever"), again=True))
    await post_ask(client, h, "I threw up. How often do I take amoxicillin?", doctor_id=doctor_id)
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by) == ("open", NEEDS_DOCTOR, "Warning sign: threw up")


async def test_payload_marks_habit_sources(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)
    (source,) = body["sources"]
    assert source["kind"] == "habit"
    assert source["title"] == "Dr. Ting Kai's usual practice"
    assert source["content"] == ASK_QUESTIONS
    assert source["url"] is None


# ---------- final review fixes ----------

async def bare_doctor(client, fake_llm, username="dr.bare", display_name="Dr. Bare"):
    """A doctor with one ask and one close habit and no warning habit: [4] ask, [5] close."""
    doctor_id = await make_doctor(client, username, display_name)
    entry = await seed_guidance(client, fake_llm, doctor_id)
    await seed_habit(client, entry, "Asks about fever and vomiting", ASK_QUESTIONS)
    await seed_habit(client, entry, "Closes with the follow-up visit", CLOSE_TEXT, "close")
    return doctor_id


async def test_no_warning_signs_taught_never_closes(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await bare_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    reply = "Yes I have a fever and pus is coming out of the wound"
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5]))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Bare")
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by) == ("open", NEEDS_DOCTOR, "No warning signs taught")


async def test_other_concerns_hand_off(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up"), concerns=True))
    body = (await post_ask(client, h, "No fever, never threw up. My leg feels odd.",
                           doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Other concerns")
    assert "other_concerns" not in first_user_message(fake_llm)  # the judge's field, not the model's


async def test_free_text_close_never_resolves_with_the_gate_off(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    free = reply_final("close", "You are fine, no need to see anyone [5].", [5],
                       signs=absent("fever", "threw up"))
    fake_llm.script(free, free, {"content": "{}"})  # then an unclean judge
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    assert "only use the doctor's own closing words" in fake_llm.calls[-2][0][-1]["content"]
    (q,) = await all_questions(client)
    assert (q.status, q.answer) == ("open", None)


async def test_reply_turn_may_name_a_drug_from_the_original_question(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    fake_llm.script(final("clarify", ASK_QUESTIONS + " [4]", [4]))
    await post_ask(client, h, "I took Advil and my incision is red.", doctor_id=doctor_id)
    text = "Ibuprofen is not in your document; take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(reply_final("answer", text, [2], signs=absent("fever", "threw up")))
    before = len(fake_llm.calls)
    body = (await post_ask(client, h, "No fever, never threw up.", doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text, body
    assert len(fake_llm.calls) == before + 2  # the model once (no repair: "ibuprofen" is Advil's
    # ingredient) and the judge
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Drug not on the prescription: advil")


async def time_out_waiting(client):
    from datetime import timedelta
    from models import CareTeamQuestion
    from routes_doctor import utcnow
    async with client.app.state.session_factory() as session:
        await session.execute(update(CareTeamQuestion).values(
            created_at=utcnow() - timedelta(hours=30)))
        await session.execute(update(CareTeamQuestion).where(
            CareTeamQuestion.last_asked_at.is_not(None)).values(
            last_asked_at=utcnow() - timedelta(hours=30)))
        await session.commit()


async def doctor_inbox(client, username="dr.ting"):
    resp = await client.post("/api/login",
                             json={"username": username, "password": "doctor-password-1"})
    assert resp.status_code == 200, resp.text
    headers = {"Authorization": f"Bearer {resp.json()['token']}"}
    return (await client.get("/api/doctor/questions", headers=headers)).json()


async def test_a_late_reply_after_the_timeout_fills_the_same_card(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await time_out_waiting(client)
    inbox = await doctor_inbox(client)
    assert [q["stopped_by"] for q in inbox["open"]] == [NOT_ANSWERED]
    reply = "no fever, but I threw up twice"
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    await post_ask(client, h, reply, doctor_id=doctor_id)
    (q,) = await all_questions(client)  # never a second card
    assert (q.status, q.reason, q.stopped_by, q.patient_reply) == (
        "open", NEEDS_DOCTOR, "Warning sign: threw up", reply)


async def test_a_late_benign_close_after_the_timeout_resolves_the_same_card(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await time_out_waiting(client)
    await doctor_inbox(client)
    reply = "No fever, never threw up."
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "close"
    (q,) = await all_questions(client)
    assert (q.status, q.answer, q.patient_reply) == ("resolved", CLOSE_TEXT, reply)


async def test_payload_lists_cited_habit_rule_ids(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)
    from models import GuidanceRule
    async with client.app.state.session_factory() as session:
        ask_id = await session.scalar(select(GuidanceRule.id).where(
            GuidanceRule.habit_type == "ask"))
    assert body["habit_rule_ids"] == [ask_id] and body["guidance_ids"] == []
    message = await clarify_message(client, doctor_id)
    assert message.payload["habit_rule_ids"] == [ask_id]



# ---------- M4 reverted: a reply gets the unchanged backstop ----------

# A representative subset at route level (every probe string is checked at
# function level in test_safety): one reply per static label category, plus
# the reviewer cases that once closed a case.
REPLY_PROBES = [
    "No trouble breathing but he is not breathing",            # breathing
    "No chest pain, my chest hurts",                           # chest
    "No face swelling. My tongue is swollen",                  # allergy, throat
    "No numbness on one side, but my arm is numb",             # stroke signs
    "Not unconscious, but he is unresponsive",                 # fainting, unconscious
    "No heart attack, cardiac arrest",                         # heart
    "No fever, I can't keep fluids down",                      # vomiting, fluids, urine
    "No fever. I can't breathe.",
    "No chest pain, until an hour ago when it started",
    "No chest pain. I take that back.",
    "No seizure, but it happened again",                       # seizure
    "No heavy bleeding, until the bandage got soaked",         # bleeding
    "I don't want to die",                                     # self-harm
]


@pytest.mark.parametrize("reply", REPLY_PROBES)
async def test_a_reply_probe_without_a_model_is_the_emergency_answer(app_and_client, fake_llm,
                                                                     reply):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    app.state.llm = None
    resp = await post_ask(client, h, reply, doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "emergency" and resp.json()["answer"] == EMERGENCY_ANSWER_COPY


@pytest.mark.parametrize("reply", REPLY_PROBES)
async def test_a_reply_probe_with_a_closing_model_is_the_emergency_answer(client, fake_llm,
                                                                          reply):
    from safety import detect_emergency
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up",
                                             *detect_emergency(reply, []))))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "emergency" and body["answer"] == EMERGENCY_ANSWER_COPY, body
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Emergency terms")


async def test_history_leaves_out_fixed_server_answers(client, fake_llm):
    # A fallback in the history taught the model to send later basic questions
    # to the doctor too. The patient's question stays; the fixed sentence becomes a stub.
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("x", [], kind="not_covered"))
    first = await post_ask(client, h, "Question alpha")
    assert first.json()["answer"].startswith("I couldn't answer this")
    fake_llm.script(answer("Answer beta [2].", [2]))
    await post_ask(client, h, "Question beta")
    fake_llm.script(answer("Take amoxicillin 500 mg [2].", [2]))
    await post_ask(client, h, "Last question")
    sent = fake_llm.calls[-1][0][1:-1]
    # A stub stands in for the fixed answer: roles keep alternating and the
    # questions stay apart.
    assert [(m["role"], m["content"]) for m in sent] == [
        ("user", "Question alpha"), ("assistant", "(I sent that question to the care team.)"),
        ("user", "Question beta"), ("assistant", "Answer beta.")]
    # The patient's own conversation still shows the fixed sentence.
    shown = (await get_conversation(client, h)).json()["messages"]
    assert any(m["content"].startswith("I couldn't answer this") for m in shown)


# --- multi-round intake (spec 4.7) ---

OWN_Q = "When did it start?"
OWN_Q2 = "Has it changed since yesterday?"
REPLY1 = "No fever, never threw up."
REPLY2 = "It started two days ago."


def own_clarify(text=OWN_Q, cites=()):
    """The model's clarify with a question of its own (it cites nothing)."""
    return final("clarify", text, list(cites))


async def second_round(client, fake_llm, h, doctor_id, text=OWN_Q, cites=()):
    """Reply 1 (clean) and the clone's second round of questions."""
    fake_llm.script(own_clarify(text, cites), judge_message(signs=absent("fever", "threw up"), close=None))
    return (await post_ask(client, h, REPLY1, doctor_id=doctor_id)).json()


def judge_message(replies=True, signs=(), concerns=False, close=5):
    return {"content": json.dumps({"replies_to_questions": replies, "warning_signs": list(signs),
                                   "other_concerns": concerns, "close": close})}


async def test_a_reply_appends_a_round_and_a_new_clarify_updates_the_waiting_row(
        client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    first = await start_clarify(client, fake_llm, h, doctor_id)
    body = await second_round(client, fake_llm, h, doctor_id)
    assert body["kind"] == "clarify" and body["pending"] is True and body["flagged"] is False
    (q,) = await all_questions(client)  # the same card, no new row
    assert body["waiting_question_id"] == q.id == first["waiting_question_id"]
    assert (q.status, q.question, q.asked, q.patient_reply, q.reason, q.stopped_by) == (
        "waiting_on_patient", ORIGINAL, OWN_Q, None, "", None)
    assert q.intake == [{"asked": ASK_QUESTIONS, "reply": REPLY1}]
    assert q.last_asked_at is not None
    assert body["questions"] == [{"text": OWN_Q, "source": "clone", "habit_id": None}]
    assert body["opener"] is None and body["sources"] == []
    # the first clarify ended; the new one is the pending one
    messages = (await get_conversation(client, h, doctor_id)).json()["messages"]
    clarifies = [m for m in messages if (m["payload"] or {}).get("kind") == "clarify"]
    assert [m["payload"]["pending"] for m in clarifies] == [False, True]
    assert await care_team(client, h) == []  # still not the care team's
    # every round reaches the next turn's context
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, REPLY2, doctor_id=doctor_id)).json()
    text = first_user_message(fake_llm)
    assert ("Round 1 - I asked: " + ASK_QUESTIONS in text and "the patient replied: " + REPLY1 in text
            and "Round 2 - I asked: " + OWN_Q in text and "Round 2 of 3" in text)
    assert body["kind"] == "close"
    (q,) = await all_questions(client)
    assert (q.status, q.answer, q.patient_reply) == ("resolved", CLOSE_TEXT, REPLY2)
    assert q.intake == [{"asked": ASK_QUESTIONS, "reply": REPLY1}, {"asked": OWN_Q, "reply": REPLY2}]


async def test_the_clarify_payload_marks_each_question_as_the_doctors_or_the_clones(
        client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)
    assert [(x["text"], x["source"]) for x in body["questions"]] == [
        ("Do you have a fever?", "doctor"), ("Have you thrown up?", "doctor")]
    assert all(isinstance(x["habit_id"], int) for x in body["questions"])
    assert body["habit_rule_ids"] == [x["habit_id"] for x in body["questions"]][:1]
    # an opener, a habit question and a question of the clone's own
    h2 = await confirmed(client, fake_llm, headers=await auth_headers(client, "pat-two"))
    d2 = await ask_first_doctor(client, fake_llm, "dr.two", "Dr. Two")
    fake_llm.script(own_clarify("Thanks for letting me know. Have you thrown up? [4] " + OWN_Q,
                                cites=[4]))
    mixed = (await post_ask(client, h2, ORIGINAL, doctor_id=d2)).json()
    assert mixed["kind"] == "clarify", mixed
    assert mixed["opener"] == "Thanks for letting me know."
    assert [(x["text"], x["source"]) for x in mixed["questions"]] == [
        ("Have you thrown up?", "doctor"), (OWN_Q, "clone")]


async def test_a_warning_sign_only_in_round_2_blocks_the_close_and_the_answer(client, fake_llm):
    # Review Focus 1
    for kind, text, cites in (("close", CLOSE_TEXT + " [5]", [5]),
                              ("answer", "Take amoxicillin 500 mg three times daily [2].", [2])):
        h = await confirmed(client, fake_llm, headers=await auth_headers(client, f"pat-{kind}"))
        doctor_id = await ask_first_doctor(client, fake_llm, f"dr.{kind}", f"Dr. {kind}")
        await start_clarify(client, fake_llm, h, doctor_id)
        await second_round(client, fake_llm, h, doctor_id)
        gate_on(client, fake_llm)
        # the judge reads round 2 (and says the fever is absent in this reply) - but
        # round 1's "threw up" was never denied: the close_blocker sees every reply
        reply = "It started two days ago and I threw up twice."
        fake_llm.script(reply_final(kind, text, cites, signs=absent("fever", "threw up")))
        body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
        assert body["answer"] == fallback_answer(f"Dr. {kind}") or body["kind"] == "answer"
        q = [q for q in await all_questions(client) if q.doctor_id == doctor_id][0]
        assert (q.status, q.stopped_by) == ("open", "Warning sign: threw up")
        assert [r["reply"] for r in q.intake] == [REPLY1, reply]


async def test_a_warning_sign_in_the_first_reply_stops_the_intake_at_once(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    fake_llm.script(own_clarify(), judge_message(
        signs=[{"term": "fever", "status": "present"}, {"term": "threw up", "status": "absent"}]))
    body = (await post_ask(client, h, "I have a fever.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Warning sign: fever")


async def test_an_unreadable_judge_blocks_every_kind(client, fake_llm):
    for index, main in enumerate((own_clarify(),
                                  final("close", CLOSE_TEXT + " [5]", [5]),
                                  final("answer", "Take amoxicillin 500 mg three times daily [2].",
                                        [2]))):
        h = await confirmed(client, fake_llm, headers=await auth_headers(client, f"pat-{index}"))
        doctor_id = await ask_first_doctor(client, fake_llm, f"dr.j{index}", f"Dr. J{index}")
        await start_clarify(client, fake_llm, h, doctor_id)
        fake_llm.script(main, {"content": "not json"})
        body = (await post_ask(client, h, REPLY1, doctor_id=doctor_id)).json()
        q = [q for q in await all_questions(client) if q.doctor_id == doctor_id][0]
        assert q.status == "open" and q.answer is None, main
        assert body["kind"] != "clarify" and body.get("pending") is None, main


async def test_the_judge_erroring_blocks_the_close(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    import agent
    fake_llm.route(agent.REPLY_JUDGE_PROMPT, lambda m: (_ for _ in ()).throw(LLMError("down")))
    original = fake_llm.chat

    async def chat(messages, tools=None, json_mode=False):
        if messages[0]["content"] == agent.REPLY_JUDGE_PROMPT:
            raise LLMError("down")
        return await original(messages, tools, json_mode)

    fake_llm.chat = chat
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    body = (await post_ask(client, h, REPLY1, doctor_id=doctor_id)).json()
    assert body["answer"] == fallback_answer("Dr. Ting Kai")
    (q,) = await all_questions(client)
    assert q.status == "open" and q.answer is None


async def test_a_round_3_clarify_is_refused_even_if_the_model_insists(client, fake_llm):
    # Review Focus 4: the third reply ends the intake
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    fake_llm.script(own_clarify(OWN_Q2), judge_message(signs=absent("fever", "threw up"), close=None))
    third = (await post_ask(client, h, REPLY2, doctor_id=doctor_id)).json()
    assert third["kind"] == "clarify" and third["pending"] is True
    again = own_clarify("Is it getting better?")
    # the model insists, twice; the judge is clean but names no close habit: handoff
    fake_llm.script(again, again, judge_message(signs=absent("fever", "threw up"), close=None))
    body = (await post_ask(client, h, "A little worse.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    assert "most rounds" in json.dumps(fake_llm.calls[-2][0])  # the repair round told it
    (q,) = await all_questions(client)
    assert q.status == "open" and len(q.intake) == 3
    assert [r["reply"] for r in q.intake] == [REPLY1, REPLY2, "A little worse."]


async def test_a_round_3_clarify_that_fails_falls_back_to_the_judges_close(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    fake_llm.script(own_clarify(OWN_Q2), judge_message(signs=absent("fever", "threw up"), close=None))
    await post_ask(client, h, REPLY2, doctor_id=doctor_id)
    again = own_clarify("Is it getting better?")
    fake_llm.script(again, again, judge_message(signs=absent("fever", "threw up"), close=5))
    body = (await post_ask(client, h, "About the same.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and body["answer"] == CLOSE_TEXT + " [5]"
    (q,) = await all_questions(client)
    assert (q.status, q.answer) == ("resolved", CLOSE_TEXT)


async def test_an_answer_after_the_intake_is_grounded_and_delivered(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    text = "Take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(reply_final("answer", text, [2], signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, REPLY2, doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text
    assert [s["number"] for s in body["sources"]] == [2]
    assert gate_calls(fake_llm) == 1
    (q,) = await all_questions(client)
    assert (q.status, q.answer) == ("resolved", "Take amoxicillin 500 mg three times daily.")
    # an ungrounded number is not delivered
    h2 = await confirmed(client, fake_llm, headers=await auth_headers(client, "pat-two"))
    d2 = await ask_first_doctor(client, fake_llm, "dr.two", "Dr. Two")
    await start_clarify(client, fake_llm, h2, d2)
    bad = final("answer", "Take amoxicillin 900 mg three times daily [2].", [2])
    fake_llm.script(bad, bad, judge_message(signs=absent("fever", "threw up"), close=None))
    body = (await post_ask(client, h2, REPLY1, doctor_id=d2)).json()
    assert body["kind"] == "not_covered"
    assert [q.status for q in await all_questions(client) if q.doctor_id == d2] == ["open"]


async def test_a_new_unrelated_question_mid_intake_ends_it_and_hands_the_symptom_off(
        client, fake_llm):
    # Review Focus 5
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    text = "Take amoxicillin 500 mg three times daily [2]."
    fake_llm.script(reply_final("answer", text, [2], replies=False,
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "How often do I take amoxicillin?",
                           doctor_id=doctor_id)).json()
    assert body["kind"] == "answer" and body["answer"] == text
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by, q.question) == (
        "open", NOT_ANSWERED, NOT_ANSWERED, ORIGINAL)
    assert q.patient_reply == "How often do I take amoxicillin?"
    # a clarify for a message that is not a reply is not delivered either
    h2 = await confirmed(client, fake_llm, headers=await auth_headers(client, "pat-two"))
    d2 = await ask_first_doctor(client, fake_llm, "dr.two", "Dr. Two")
    await start_clarify(client, fake_llm, h2, d2)
    fake_llm.script(own_clarify(), judge_message(replies=False, signs=absent("fever", "threw up")))
    body = (await post_ask(client, h2, "What is a stent?", doctor_id=d2)).json()
    assert body["kind"] == "not_covered" and body.get("pending") is None


async def test_the_handoff_shows_every_round(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    reply = "It started two days ago and I threw up twice."
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    await post_ask(client, h, reply, doctor_id=doctor_id)
    inbox = await doctor_inbox(client)
    (card,) = inbox["open"]
    assert card["rounds"] == [{"asked": ASK_QUESTIONS, "reply": REPLY1},
                              {"asked": OWN_Q, "reply": reply}]
    assert (card["asked"], card["patient_reply"], card["stopped_by"]) == (
        OWN_Q, reply, "Warning sign: threw up")
    (seen,) = await care_team(client, h)
    assert seen["rounds"] == card["rounds"]
    # and the draft context carries all of them
    import drafts
    async with client.app.state.session_factory() as session:
        row = await session.get(CareTeamQuestion, card["id"])
        message = drafts._user_message(row, drafts.DraftContext(evidence=[]), [])
    for part in (ASK_QUESTIONS, REPLY1, OWN_Q, reply):
        assert part in message


async def test_the_timeout_and_a_late_reply_keep_one_card_in_round_2(client, fake_llm):
    # Review Focus 3
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    inbox = await doctor_inbox(client)
    assert [q["rounds"] for q in inbox["waiting"]] == [
        [{"asked": ASK_QUESTIONS, "reply": REPLY1}, {"asked": OWN_Q, "reply": None}]]
    await time_out_waiting(client)
    inbox = await doctor_inbox(client)
    assert [q["stopped_by"] for q in inbox["open"]] == [NOT_ANSWERED] and inbox["waiting"] == []
    # a late reply asks round 3 of the same card
    fake_llm.script(own_clarify(OWN_Q2), judge_message(signs=absent("fever", "threw up"), close=None))
    body = (await post_ask(client, h, REPLY2, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify" and body["pending"] is True
    (q,) = await all_questions(client)  # never a second card
    assert (q.status, q.asked, q.patient_reply, q.reason) == (
        "waiting_on_patient", OWN_Q2, None, "")
    assert body["waiting_question_id"] == q.id
    inbox = await doctor_inbox(client)
    assert inbox["open"] == [] and [x["id"] for x in inbox["waiting"]] == [q.id]
    # and the third reply closes it on that card
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, "Better.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close"
    (q,) = await all_questions(client)
    assert (q.status, q.answer) == ("resolved", CLOSE_TEXT) and len(q.intake) == 3


async def test_a_round_asked_just_before_the_timeout_keeps_waiting(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await time_out_waiting(client)  # round 1 is 30 h old, the doctor's by now
    await doctor_inbox(client)
    fake_llm.script(own_clarify(), judge_message(signs=absent("fever", "threw up"), close=None))
    await post_ask(client, h, REPLY1, doctor_id=doctor_id)  # round 2 asked just now
    inbox = await doctor_inbox(client)
    assert inbox["open"] == [] and len(inbox["waiting"]) == 1


async def test_a_reply_after_the_doctor_answered_a_round_2_clarify_is_never_lost(
        client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    await set_waiting(client, status="answered", answer="Please send a photo.")
    fake_llm.script(own_clarify(OWN_Q2), judge_message(signs=absent("fever", "threw up"), close=None))
    body = (await post_ask(client, h, REPLY2, doctor_id=doctor_id)).json()
    # The case was settled meanwhile: no more questions, the reply goes to the doctor.
    assert body["kind"] == "not_covered" and not body.get("pending")
    qs = await all_questions(client)
    assert [q.status for q in qs] == ["answered", "open"]  # the doctor's stays
    assert qs[1].stopped_by == "Case was settled before this reply"
    assert [r["reply"] for r in qs[1].intake] == [REPLY1, REPLY2]


async def test_the_confidence_gate_reads_every_round(client, fake_llm):
    import confidence
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)
    await second_round(client, fake_llm, h, doctor_id)
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    await post_ask(client, h, REPLY2, doctor_id=doctor_id)
    (verify,) = [c for c in fake_llm.routed_calls if c[0][0]["content"] == confidence.VERIFY_PROMPT]
    text = json.dumps(verify[0])
    for part in (ORIGINAL, ASK_QUESTIONS, REPLY1, OWN_Q, REPLY2):
        assert part in text


# ---------- the clone interview (2026-11-17 spec, Task 3) ----------

REDNESS_Q = "How long has the redness been there?"
DISCHARGE_Q = "Is there any discharge from the incision?"
FEVER_Q = "Have you had a fever?"
FEVER_REPHRASE = "Have you noticed any fever since the surgery?"


async def interview_doctor(client, fake_llm):
    """Two single-question ask habits, a close [5] and a warning habit on fever.
    Returns (doctor_id, {quote: rule id})."""
    from models import GuidanceRule
    doctor_id = await make_doctor(client)
    entry = await seed_guidance(client, fake_llm, doctor_id, question="My incision is red.",
                                answer="Watch the incision.")
    await seed_habit(client, entry, "Asks how long the redness lasted", REDNESS_Q)
    await seed_habit(client, entry, "Closes with the follow-up visit", CLOSE_TEXT, "close")
    await seed_habit(client, entry, "Wants to hear about fever", "Tell me about any fever.",
                     "warning", warning_terms=["fever"])
    await seed_habit(client, entry, "Asks about discharge", DISCHARGE_Q)
    async with client.app.state.session_factory() as session:
        ids = {r.quote: r.id for r in await session.scalars(select(GuidanceRule))}
    return doctor_id, ids


class Interviewer:
    """Routes the interview's small model calls: each prompt answers from its
    own list, in order (the last answer repeats)."""

    def __init__(self, fake_llm, build, extracts=(), judges=(), questions=()):
        import interview
        self.lists = {"build": [build], "extract": list(extracts) or [{"items": []}],
                      "judge": list(judges), "question": list(questions)}
        self.calls = {k: 0 for k in self.lists}
        prompts = {"build": interview.BUILD_PROMPT, "extract": interview.EXTRACT_PROMPT,
                   "question": interview.QUESTION_PROMPT.format(doctor="Dr. Ting Kai")}
        import agent
        prompts["judge"] = agent.REPLY_JUDGE_PROMPT
        for key, prompt in prompts.items():
            fake_llm.route(prompt, self._responder(key))

    def _responder(self, key):
        def respond(messages):
            found = self.lists[key]
            assert found, f"no {key} answer scripted"
            answer = found[min(self.calls[key], len(found) - 1)]
            self.calls[key] += 1
            return {"content": answer if isinstance(answer, str) else json.dumps(answer)}
        return respond


def three_items(ids):
    return {"reports_symptom": True, "items": [
        {"label": "how long the redness lasted", "habits": [ids[REDNESS_Q]]},
        {"label": "incision discharge", "habits": [ids[DISCHARGE_Q]]}]}


def judged(replies=True, signs=(), concerns=False, close=None):
    return {"replies_to_questions": replies, "warning_signs": list(signs),
            "other_concerns": concerns, "close": close}


def first_turn_handoff(fake_llm):
    """The main model hands off and the ask-first picker finds nothing to ask:
    the 4.7 intake would act on this turn."""
    fake_llm.script(final("not_covered", "x", []),
                    {"content": json.dumps({"reports_symptom": True, "ask": []})})


async def start_interview(client, fake_llm, h, doctor_id):
    first_turn_handoff(fake_llm)
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify", body
    return body


async def not_sure(client, h, doctor_id, round_number):
    return await client.post("/api/ask", headers=h, json={
        "doctor_id": doctor_id, "question": "", "not_sure": True, "round": round_number})


async def test_interview_first_turn_asks_one_question_and_stores_items(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}])
    body = await start_interview(client, fake_llm, h, doctor_id)
    (q,) = await all_questions(client)
    assert body["pending"] is True and body["flagged"] is False
    assert body["answer"] == FEVER_Q
    assert body["questions"] == [{"text": FEVER_Q, "doctor": False}]
    assert body["interview"] == {"so_far": [], "round": 1, "max_rounds": 6, "item": "fever"}
    assert body["waiting_question_id"] == q.id
    assert (q.status, q.question, q.asked) == ("waiting_on_patient", ORIGINAL, FEVER_Q)
    assert [(i["label"], i["safety"], i["status"]) for i in q.intake["items"]] == [
        ("how long the redness lasted", False, "unknown"),
        ("incision discharge", False, "unknown"), ("fever", True, "unknown")]
    assert q.intake["items"][2]["attempts"] == 1
    assert q.intake["rounds"] == [{"asked": FEVER_Q, "item": 2, "reply": None}]
    assert (await clarify_message(client, doctor_id)).payload == body


async def test_interview_reply_fills_item_and_asks_next_without_agent_call(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(signs=absent("fever"))],
                       extracts=[{"items": [{"index": 0, "status": "answered",
                                             "quote": "red since yesterday"}]}])
    await start_interview(client, fake_llm, h, doctor_id)
    main_calls = len(fake_llm.calls)
    body = (await post_ask(client, h, "No fever at all, red since yesterday.",
                           doctor_id=doctor_id)).json()
    assert len(fake_llm.calls) == main_calls  # the agent model is not called
    assert talk.calls["judge"] == 1
    assert body["kind"] == "clarify" and body["pending"] is True
    # The fever was denied and the redness answered: the discharge habit is next.
    assert body["questions"] == [{"text": DISCHARGE_Q, "doctor": True}]
    assert body["habit_rule_ids"] == [ids[DISCHARGE_Q]]
    assert body["interview"] == {
        "so_far": [{"label": "how long the redness lasted", "status": "answered",
                    "quote": "red since yesterday"},
                   {"label": "fever", "status": "answered", "quote": ""}],
        "round": 2, "max_rounds": 6, "item": "incision discharge"}
    (q,) = await all_questions(client)
    assert (q.status, q.asked, q.patient_reply) == ("waiting_on_patient", DISCHARGE_Q, None)
    assert q.intake["rounds"] == [
        {"asked": FEVER_Q, "item": 2, "reply": "No fever at all, red since yesterday."},
        {"asked": DISCHARGE_Q, "item": 1, "reply": None}]
    assert body["waiting_question_id"] == q.id


async def test_interview_ends_and_closes_when_all_answered_and_no_signs(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(
        fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
        judges=[judged(signs=absent("fever")), judged(), judged(),
                judged(signs=absent("fever"), close=5)],
        extracts=[{"items": []}, {"items": []},
                  {"items": [{"index": 0, "status": "answered", "quote": "two days"}]},
                  {"items": [{"index": 1, "status": "answered", "quote": "no discharge"}]}])
    await start_interview(client, fake_llm, h, doctor_id)
    for reply, asked in (("No fever at all.", REDNESS_Q), ("About two days.", DISCHARGE_Q)):
        body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
        assert body["questions"] == [{"text": asked, "doctor": True}], body
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    body = (await post_ask(client, h, "There is no discharge.", doctor_id=doctor_id)).json()
    assert body["kind"] == "close" and body["answer"] == CLOSE_TEXT + " [5]", body
    text = first_user_message(fake_llm)
    assert REPLY_MARK in text and "No more rounds of questions are allowed" in text
    assert FEVER_Q in text and "About two days." in text
    (q,) = await all_questions(client)
    assert (q.status, q.answer, q.patient_reply) == (
        "resolved", CLOSE_TEXT, "There is no discharge.")
    assert [i["status"] for i in q.intake["items"]] == ["answered"] * 3
    assert q.intake["rounds"][-1]["reply"] == "There is no discharge."
    assert talk.calls["judge"] == 4  # three interview rounds and the agent's own judge


async def test_interview_safety_item_present_hands_off_with_summary(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                judges=[judged(signs=[{"term": "fever", "status": "present"}])],
                extracts=[{"items": [{"index": 0, "status": "answered",
                                      "quote": "red since Monday"}]}])
    await start_interview(client, fake_llm, h, doctor_id)
    main_calls = len(fake_llm.calls)
    body = (await post_ask(client, h, "Yes, a fever, and it is red since Monday.",
                           doctor_id=doctor_id)).json()
    assert len(fake_llm.calls) == main_calls
    assert body["kind"] == "not_covered" and body["answer"] == fallback_answer("Dr. Ting Kai")
    assert "interview" not in body
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by) == ("open", NEEDS_DOCTOR, "Warning sign: fever")
    (row,) = (await doctor_inbox(client))["open"]
    assert row["summary"] == ("Not answered: how long the redness lasted, incision discharge, "
                              "fever")
    assert row["rounds"] == [{"asked": FEVER_Q,
                              "reply": "Yes, a fever, and it is red since Monday."}]
    assert row["intake"]["items"][2]["label"] == "fever"


async def test_interview_unclear_safety_item_reasked_once_then_unsure(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    unclear = judged(signs=[{"term": "fever", "status": "unclear"}])
    talk = Interviewer(fake_llm, three_items(ids),
                       questions=[{"question": FEVER_Q}, {"question": FEVER_REPHRASE}],
                       judges=[unclear])
    await start_interview(client, fake_llm, h, doctor_id)
    body = (await post_ask(client, h, "I felt a bit warm maybe.", doctor_id=doctor_id)).json()
    assert body["questions"] == [{"text": FEVER_REPHRASE, "doctor": False}], body
    assert body["interview"]["round"] == 2 and body["interview"]["item"] == "fever"
    body = (await post_ask(client, h, "Hard to say.", doctor_id=doctor_id)).json()
    assert body["questions"] == [{"text": REDNESS_Q, "doctor": True}], body
    assert body["interview"]["so_far"] == [{"label": "fever", "status": "unsure", "quote": ""}]
    assert talk.calls["question"] == 2
    (q,) = await all_questions(client)
    fever = q.intake["items"][2]
    assert (fever["status"], fever["attempts"]) == ("unsure", 2)


async def test_not_sure_marks_item_and_moves_on(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}])
    await start_interview(client, fake_llm, h, doctor_id)
    main_calls = len(fake_llm.calls)
    resp = await not_sure(client, h, doctor_id, 1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(fake_llm.calls) == main_calls and talk.calls["judge"] == 0
    assert talk.calls["extract"] == 1  # the first turn's only
    assert body["questions"] == [{"text": REDNESS_Q, "doctor": True}]
    assert body["interview"]["so_far"] == [{"label": "fever", "status": "unsure", "quote": ""}]
    assert body["interview"]["round"] == 2
    messages = (await get_conversation(client, h, doctor_id)).json()["messages"]
    assert [m["content"] for m in messages if m["role"] == "user"] == [ORIGINAL, "I'm not sure."]
    (q,) = await all_questions(client)
    assert q.intake["rounds"][0]["reply"] == "I'm not sure."


async def test_not_sure_stale_round_is_ignored(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}])
    await start_interview(client, fake_llm, h, doctor_id)
    first = (await not_sure(client, h, doctor_id, 1)).json()
    count = len((await get_conversation(client, h, doctor_id)).json()["messages"])
    again = await not_sure(client, h, doctor_id, 1)  # the double tap
    assert again.status_code == 200, again.text
    assert again.json() == first
    assert len((await get_conversation(client, h, doctor_id)).json()["messages"]) == count
    (q,) = await all_questions(client)
    assert [i["status"] for i in q.intake["items"]] == ["unknown", "unknown", "unsure"]
    assert len(q.intake["rounds"]) == 2


async def test_new_question_mid_interview_hands_off(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(replies=False, signs=absent("fever"))])
    await start_interview(client, fake_llm, h, doctor_id)
    new = "How often do I take amoxicillin?"
    fake_llm.script(final("answer", "Take amoxicillin 500 mg three times daily [2].", [2]))
    body = (await post_ask(client, h, new, doctor_id=doctor_id)).json()
    # The agent answers the new question as usual, with its citation.
    assert body["kind"] == "answer", body
    assert body["answer"] == "Take amoxicillin 500 mg three times daily [2]."
    assert [s["number"] for s in body["sources"]] == [2] and "interview" not in body
    assert first_user_message(fake_llm).endswith(new)
    assert talk.calls["judge"] == 2  # the interview's and the agent's own
    # The original symptom is handed off with the interview's summary and rounds.
    (q,) = await all_questions(client)
    assert (q.status, q.reason, q.stopped_by, q.patient_reply, q.question) == (
        "open", NOT_ANSWERED, NOT_ANSWERED, new, ORIGINAL)
    (row,) = (await doctor_inbox(client))["open"]
    assert row["summary"] == ("Not answered: how long the redness lasted, incision discharge, "
                              "fever")
    assert row["rounds"] == [{"asked": FEVER_Q, "reply": new}]
    assert (await clarify_message(client, doctor_id)).payload["pending"] is False


async def test_present_sign_in_a_new_question_names_the_sign(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                judges=[judged(replies=False, signs=[{"term": "fever", "status": "present"}])])
    await start_interview(client, fake_llm, h, doctor_id)
    body = (await post_ask(client, h, "I have a fever, can I swim?", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered"
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Warning sign: fever")


async def test_expired_interview_row_is_not_an_interview_reply(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(signs=absent("fever"), close=5)])
    await start_interview(client, fake_llm, h, doctor_id)
    await time_out_waiting(client)  # asked 30 hours ago, the inbox not yet opened
    # "Not sure" no longer applies: the stored clarify comes back, nothing saved.
    stale = await not_sure(client, h, doctor_id, 1)
    assert stale.status_code == 200 and stale.json()["interview"]["round"] == 1
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    reply = "No fever at all."
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    # The agent reads the late reply (4.7); no interview round, no next question.
    assert talk.calls["question"] == 1 and talk.calls["judge"] == 1  # the agent's judge only
    assert REPLY_MARK in first_user_message(fake_llm)
    assert body["kind"] == "not_covered" and "interview" not in body, body
    (q,) = await all_questions(client)
    # The unasked redness and discharge do not block, but the fever was never
    # asked as an interview item: the close is not delivered, the doctor sees it.
    assert q.status == "open" and q.stopped_by == "Warning sign: fever"
    assert q.intake["rounds"] == [{"asked": FEVER_Q, "item": 2, "reply": reply}]


async def test_old_shape_intake_still_uses_4_7_flow(client, fake_llm):
    gate_on(client, fake_llm)
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    body = await start_clarify(client, fake_llm, h, doctor_id)  # the build finds no items
    assert "interview" not in body and body["questions"][0]["source"] == "doctor"
    (q,) = await all_questions(client)
    assert not q.intake
    reply = "No fever, never threw up."
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up")))
    body = (await post_ask(client, h, reply, doctor_id=doctor_id)).json()
    assert body["kind"] == "close", body
    (q,) = await all_questions(client)
    assert q.status == "resolved"
    assert q.intake == [{"asked": ASK_QUESTIONS, "reply": reply}]
    # The "Not sure" button means nothing on an old intake: nothing changes.
    resp = await not_sure(client, h, doctor_id, 1)
    assert resp.status_code == 409


async def test_doctor_without_ask_habits_uses_4_7_flow(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await make_doctor(client)
    entry = await seed_guidance(client, fake_llm, doctor_id)
    await seed_habit(client, entry, "Closes with the follow-up visit", CLOSE_TEXT, "close")
    talk = Interviewer(fake_llm, {"reports_symptom": True, "items": []})
    fake_llm.script(final("not_covered", "x", []))
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered" and "interview" not in body
    assert talk.calls["build"] == 0  # no ask habit, no warning sign: nothing to build
    assert [q.status for q in await all_questions(client)] == ["open"]
    # With an ask habit but no items (and no warning sign), the 4.7 clarify stands.
    doctor_id = await make_doctor(client, "dr.two", "Dr. Two")
    entry = await seed_guidance(client, fake_llm, doctor_id)
    await seed_habit(client, entry, "Asks about fever and vomiting", ASK_QUESTIONS)
    talk = Interviewer(fake_llm, {"reports_symptom": True, "items": []})
    fake_llm.script(final("clarify", ASK_QUESTIONS + " [4]", [4]))
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert body["kind"] == "clarify" and "interview" not in body and talk.calls["build"] == 1


async def test_interview_build_failure_keeps_agent_result(client, fake_llm, monkeypatch):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, "not json")
    first_turn_handoff(fake_llm)
    body = (await post_ask(client, h, ORIGINAL, doctor_id=doctor_id)).json()
    assert talk.calls["build"] == 1
    assert body["kind"] == "not_covered" and "interview" not in body
    assert [q.status for q in await all_questions(client)] == ["open"]

    async def no_embedding(*args, **kwargs):
        raise LLMError("embedding down")

    monkeypatch.setattr("routes_agent.similar_case_habits", no_embedding)
    h2 = await confirmed(client, fake_llm, await auth_headers(client, "bob"))
    first_turn_handoff(fake_llm)
    resp = await post_ask(client, h2, ORIGINAL, doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "not_covered" and talk.calls["build"] == 1


async def test_interview_unsure_safety_item_blocks_the_close(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                judges=[judged(), judged(), judged(signs=absent("fever"), close=5)],
                extracts=[{"items": []},
                          {"items": [{"index": 0, "status": "answered", "quote": "two days"}]},
                          {"items": [{"index": 1, "status": "answered",
                                      "quote": "no discharge"}]}])
    await start_interview(client, fake_llm, h, doctor_id)
    await not_sure(client, h, doctor_id, 1)
    await post_ask(client, h, "About two days.", doctor_id=doctor_id)
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    body = (await post_ask(client, h, "There is no discharge.", doctor_id=doctor_id)).json()
    assert body["kind"] == "not_covered", body  # "Not sure" about a danger sign never closes
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Warning sign: fever")


async def test_interview_emergency_reply_gets_the_emergency_answer(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}])
    await start_interview(client, fake_llm, h, doctor_id)
    fake_llm.script(answer("Call 911 now.", [], kind="emergency"))
    body = (await post_ask(client, h, "No fever but I have chest pain.",
                           doctor_id=doctor_id)).json()
    assert body["kind"] == "emergency" and body["emergency"]["banner"] is True
    assert talk.calls["judge"] == 0
    (q,) = await all_questions(client)
    assert (q.status, q.stopped_by) == ("open", "Emergency terms")
    assert q.intake["rounds"] == [
        {"asked": FEVER_Q, "item": 2, "reply": "No fever but I have chest pain."}]


# ---------- denied danger signs on interview replies (2026-11-17 spec 4.4, Task 4) ----------

BREATHING_DENIAL = "No fever and no trouble breathing."


async def interview_reply(client, fake_llm, h, doctor_id, reply):
    """The response to `reply` and the waiting row after it."""
    resp = await post_ask(client, h, reply, doctor_id=doctor_id)
    assert resp.status_code == 200, resp.text
    (q,) = await all_questions(client)
    return resp.json(), q


def the_agent_hands_off(fake_llm):
    """The agent's own call on an emergency turn: the fixed answer wins anyway."""
    fake_llm.script(final("not_covered", "x", []))


def assert_emergency(body, q):
    assert body["kind"] == "emergency" and body["answer"] == EMERGENCY_ANSWER_COPY, body
    assert (q.status, q.stopped_by) == ("open", "Emergency terms")


async def test_interview_reply_denying_breathing_continues(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(signs=absent("fever", "trouble breathing"))])
    await start_interview(client, fake_llm, h, doctor_id)
    main_calls = len(fake_llm.calls)
    body, q = await interview_reply(client, fake_llm, h, doctor_id, BREATHING_DENIAL)
    assert len(fake_llm.calls) == main_calls  # the agent model is not called
    # One judge call decides both the backstop and the round.
    assert talk.calls["judge"] == 1
    assert body["kind"] == "clarify" and body["questions"] == [
        {"text": REDNESS_Q, "doctor": True}], body
    assert body["interview"]["so_far"] == [{"label": "fever", "status": "answered", "quote": ""}]
    assert (q.status, q.asked) == ("waiting_on_patient", REDNESS_Q)
    assert q.intake["rounds"][0]["reply"] == BREATHING_DENIAL


@pytest.mark.parametrize("reply", [
    "No fever but I can't breathe",                    # an affirmed term
    "I did not faint. Well I did",                     # a retraction
    "No trouble breathing but he is not breathing",    # an M4 synonym probe
    "I don't want to hurt myself",                     # self-harm: never relaxed
])
async def test_interview_reply_not_a_plain_denial_is_emergency(client, fake_llm, reply):
    from safety import detect_emergency
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                judges=[judged(signs=absent("fever", *detect_emergency(reply, [])))])
    await start_interview(client, fake_llm, h, doctor_id)
    the_agent_hands_off(fake_llm)
    assert_emergency(*await interview_reply(client, fake_llm, h, doctor_id, reply))


@pytest.mark.parametrize("judge", [
    judged(signs=[*absent("fever"), {"term": "trouble breathing", "status": "unclear"}]),
    judged(signs=absent("fever")),                     # the term left out
    "not json",                                        # the judge failed
])
async def test_interview_reply_denial_unconfirmed_by_judge_is_emergency(client, fake_llm, judge):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}], judges=[judge])
    await start_interview(client, fake_llm, h, doctor_id)
    the_agent_hands_off(fake_llm)
    assert_emergency(*await interview_reply(client, fake_llm, h, doctor_id, BREATHING_DENIAL))


async def test_interview_reply_denial_without_a_model_is_emergency(app_and_client, fake_llm):
    app, client = app_and_client
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                judges=[judged(signs=absent("fever", "trouble breathing"))])
    await start_interview(client, fake_llm, h, doctor_id)
    app.state.llm = None
    body, q = await interview_reply(client, fake_llm, h, doctor_id, BREATHING_DENIAL)
    assert_emergency(body, q)


async def test_first_message_denial_is_not_relaxed(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(signs=absent("fever", "trouble breathing"))])
    the_agent_hands_off(fake_llm)
    resp = await post_ask(client, h, "My incision is red. No trouble breathing.",
                          doctor_id=doctor_id)
    body = resp.json()
    # The backstop's terms reach the agent and the banner: no judge, no interview.
    assert body["emergency"] == {"terms": ["trouble breathing"], "banner": True}, body
    assert "interview" not in body
    assert talk.calls["judge"] == 0 and talk.calls["build"] == 0


async def test_old_shape_intake_denial_is_not_relaxed(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id = await ask_first_doctor(client, fake_llm)
    await start_clarify(client, fake_llm, h, doctor_id)  # the 4.7 list-shape intake
    reply = "No fever and no trouble breathing."
    fake_llm.script(reply_final("close", CLOSE_TEXT + " [5]", [5],
                                signs=absent("fever", "threw up", "trouble breathing")))
    assert_emergency(*await interview_reply(client, fake_llm, h, doctor_id, reply))


async def test_expired_interview_denial_is_not_relaxed(client, fake_llm):
    h = await confirmed(client, fake_llm)
    doctor_id, ids = await interview_doctor(client, fake_llm)
    talk = Interviewer(fake_llm, three_items(ids), questions=[{"question": FEVER_Q}],
                       judges=[judged(signs=absent("fever", "trouble breathing"), close=5)])
    await start_interview(client, fake_llm, h, doctor_id)
    await time_out_waiting(client)
    fake_llm.script(final("close", CLOSE_TEXT + " [5]", [5]))
    resp = await post_ask(client, h, BREATHING_DENIAL, doctor_id=doctor_id)
    body = resp.json()
    assert body["kind"] == "emergency" and body["answer"] == EMERGENCY_ANSWER_COPY, body
    assert talk.calls["judge"] == 0
