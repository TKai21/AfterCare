import json

import httpx
import pytest_asyncio
from sqlalchemy import select, text, update

from agent import fallback_answer
from app import create_app
from auth import create_doctor
from config import Settings
from druglabels import DEMO_LABELS, LabelClient, fixture_transport
from conftest import auth_headers
from guidance import embed_entry
from knowledge import KnowledgeStore
from llm import LLMClient, LLMError
from models import ChatMessage, DoctorGuidance, DrugLabel, User
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
    h = await confirmed(client, fake_llm)
    fake_llm.script(answer("Call if your fever is above 101 F, even while taking amoxicillin [3].", [3]))
    body = (await post_ask(client, h, "When should I call?")).json()
    assert body["kind"] == "answer" and "amoxicillin" in body["answer"]
    fake_llm.script(answer("Call if your fever is above 101 F, even while taking Amoxil [3].", [3]))
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
        searched.append(args[-2])
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
