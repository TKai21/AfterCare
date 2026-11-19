"""De-identification at the model boundary (privacy.py)."""
import json

import pytest

import privacy
from conftest import auth_headers
from fake_llm import FakeLLM
from llm import LLMError
from privacy import DOCTOR, PATIENT, Masker, PrivateLLM, Vault, identifiers_in
from test_routes_agent import answer, default_doctor, post_ask
from test_routes_documents import upload_text

DOC = """Lakeview Family Medical Center - Discharge Summary
Patient: Sam Delgado (fictional), 52-year-old adult
MRN: 00482913
Discharge date: 2026-03-02
Attending: Dr. Helen Okafor, Family Medicine

Take colchicine 0.6 mg by mouth twice daily for 10 days.

Take naproxen 500-1000 mg with food.

See Dr. Helen Okafor on March 12, 2026.

Call the clinic at 555-0177 if you have diarrhea."""


def test_identifiers_read_from_a_document():
    found = identifiers_in(DOC)
    assert (PATIENT, "Sam Delgado") in found and (DOCTOR, "Helen Okafor") in found
    assert (privacy.FACILITY, "Lakeview Family Medical Center") in found
    assert (privacy.ID, "00482913") in found
    # Numbered paragraphs, as the agent sends the document.
    assert (PATIENT, "Sam Delgado") in identifiers_in("[2] Patient: Sam Delgado, 52")


@pytest.mark.parametrize("text, expected", [
    ("Patient: JOHN DOE", [(PATIENT, "JOHN DOE")]),
    ("Patient: Mary O'Brien-Smith (she/her)", [(PATIENT, "Mary O'Brien-Smith")]),
    ("Attending: Dr. J. Park, MD", [(DOCTOR, "J. Park"), (DOCTOR, "J. Park")]),
    ("You are Dr. Ting Kai's AI clone.", [(DOCTOR, "Ting Kai")]),
    ("See Dr. Helen Okafor On Monday.", [(DOCTOR, "Helen Okafor")]),
    ("Patient: 52-year-old adult", []),
    ("Call the clinic if you have a fever.", []),
    ("The patient: take it with food.", []),
])
def test_identifier_rules(text, expected):
    assert identifiers_in(text) == expected


def test_mask_round_trip_keeps_medical_content():
    masker = Masker()
    text = DOC + "\nPatient message: I'm Sam, reach me at (213) 555-0199 or sam.d@example.com."
    masked = masker.mask(text)
    for secret in ("Sam", "Delgado", "Helen", "Okafor", "Lakeview", "555-0177", "00482913",
                   "555-0199", "sam.d@example.com"):
        assert secret not in masked, secret
    for kept in ("colchicine 0.6 mg", "twice daily for 10 days", "500-1000 mg", "52-year-old",
                 "March 12, 2026", "2026-03-02", "diarrhea", "Family Medicine"):
        assert kept in masked, kept
    assert "Patient: [PATIENT_1] (fictional)" in masked
    assert "See Dr. [DOCTOR_1] on March 12" in masked and "Attending: Dr. [DOCTOR_1]," in masked
    assert masker.unmask(masked) == text
    assert masker.leaked(masked) is None


def test_same_value_same_placeholder_across_texts():
    masker = Masker([(DOCTOR, "Helen Okafor"), (PATIENT, "Sam Delgado")])
    first = masker.mask("You are Dr. Helen Okafor's AI clone.")
    second = masker.mask("sam delgado asks: is Dr. Okafor in? Call 555-0177.")
    assert first == "You are Dr. [DOCTOR_1]'s AI clone."
    # A surname alone has its own placeholder, so it is restored as written.
    assert second == "[PATIENT_1] asks: is Dr. [DOCTOR_2] in? Call [PHONE_1]."
    assert masker.unmask(second) == "Sam Delgado asks: is Dr. Okafor in? Call 555-0177."
    assert masker.unmask("Tell [PATIENT_1] to call [PHONE_1].") == "Tell Sam Delgado to call 555-0177."


def test_a_name_part_that_is_an_everyday_or_medical_word_is_not_masked_alone():
    masker = Masker([(PATIENT, "Rose Pain"), (PATIENT, "May Aspirin")],
                    is_medical=lambda word: word.lower() == "aspirin")
    masked = masker.mask("Rose Pain has chest pain. Pain is worse in May. Aspirin helps. "
                         "May Aspirin called.")
    assert masked == ("[PATIENT_2] has chest pain. Pain is worse in May. Aspirin helps. "
                      "[PATIENT_1] called.")


@pytest.mark.parametrize("text", [
    "Take 500-1000 mg every 6 hours.", "Call 911.", "A fever of 101 F (38.3 C).",
    "Drink 2 liters a day.", "Limit sodium to 2,300 mg.", "On 2026-11-24 at 10:30.",
    "Pain 7 out of 10 for 3 days.", "Read https://medlineplus.gov/gout.html for more.",
])
def test_numbers_and_public_links_are_not_masked(text):
    assert Masker().mask(text) == text


@pytest.mark.parametrize("text, kind", [
    ("Call 555-0177 today.", "PHONE"), ("Call (213) 555-0199.", "PHONE"),
    ("Call +1 213-555-0199.", "PHONE"), ("Write to a.b@clinic.org.", "EMAIL"),
    ("SSN 123-45-6789.", "ID"), ("I live at 1200 W Jefferson Blvd, Apt 4.", "ADDRESS"),
])
def test_patterns_are_masked(text, kind):
    masker = Masker()
    masked = masker.mask(text)
    assert f"[{kind}_1]" in masked and masker.unmask(masked) == text


def test_guard_finds_a_name_the_rules_missed():
    masker = Masker([(PATIENT, "Sam Delgado")])
    masker.mask("Sam Delgado")
    assert masker.leaked("note for SamDelgado") is None  # not the name as a word
    assert masker.leaked("note for sam   delgado") == "sam delgado"


# ---------- the wrapper ----------

async def vault():
    v = Vault()
    v.add(PATIENT, "Sam Delgado")
    v.add(DOCTOR, "Helen Okafor")
    return v


async def test_chat_masks_out_and_unmasks_back():
    inner = FakeLLM()
    inner.script({"content": "Tell [PATIENT_1] to call [PHONE_1].", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "add_care_team_question",
            "arguments": json.dumps({"question": "[PATIENT_1] asks about Dr. [DOCTOR_1]"})}}]})
    llm = PrivateLLM(inner, vault)
    reply = await llm.chat([
        {"role": "system", "content": "You are Dr. Helen Okafor's AI clone."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c0", "type": "function",
            "function": {"name": "x", "arguments": json.dumps({"q": "Sam Delgado"})}}]},
        {"role": "tool", "tool_call_id": "c0", "content": json.dumps({"phone": "555-0177"})},
        {"role": "user", "content": "I'm Sam. Call me at 555-0177."},
    ], tools=[{"type": "function"}], json_mode=True)
    (sent, tools, json_mode), = inner.calls
    assert tools == [{"type": "function"}] and json_mode is True
    assert sent[0]["content"] == "You are Dr. [DOCTOR_1]'s AI clone."
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"q": "[PATIENT_1]"}
    assert json.loads(sent[2]["content"]) == {"phone": "[PHONE_1]"}
    assert sent[3]["content"] == "I'm [PATIENT_2]. Call me at [PHONE_1]."
    assert reply["content"] == "Tell Sam Delgado to call 555-0177."
    assert json.loads(reply["tool_calls"][0]["function"]["arguments"]) == {
        "question": "Sam Delgado asks about Dr. Helen Okafor"}


async def test_embed_masks_and_transcribe_passes_through():
    inner = FakeLLM()
    seen = []
    original = inner.embed

    async def embed(texts):
        seen.extend(texts)
        return await original(texts)
    inner.embed = embed
    inner.transcripts.append("hello")
    llm = PrivateLLM(inner, vault)
    vectors = await llm.embed(["Sam Delgado has gout", "call 555-0177"])
    assert seen == ["[PATIENT_1] has gout", "call [PHONE_1]"] and len(vectors) == 2
    assert await llm.transcribe(b"x", "a.webm", "audio/webm") == "hello"
    assert llm.embedding_model == inner.embedding_model


async def test_unknown_placeholder_in_a_reply_is_a_failed_call():
    inner = FakeLLM()
    inner.script({"content": "Call [PHONE_3]."})
    with pytest.raises(LLMError):
        await PrivateLLM(inner, vault).chat([{"role": "user", "content": "Call 555-0177?"}])


async def test_a_payload_that_still_holds_a_name_is_not_sent(monkeypatch):
    inner = FakeLLM()
    monkeypatch.setattr(Masker, "_replace_value", lambda self, text, kind, value: text)
    llm = PrivateLLM(inner, vault)
    with pytest.raises(LLMError):
        await llm.chat([{"role": "user", "content": "I am Sam Delgado."}])
    with pytest.raises(LLMError):
        await llm.embed(["I am Sam Delgado."])
    assert inner.calls == []


async def test_failed_vault_load_keeps_the_text_rules():
    async def broken():
        raise RuntimeError("database down")
    inner = FakeLLM()
    inner.script({"content": "ok"})
    await PrivateLLM(inner, broken).chat([{"role": "user", "content": "Patient: Sam Delgado"}])
    assert inner.calls[0][0][0]["content"] == "Patient: [PATIENT_1]"


# ---------- through the app: nothing identifying reaches the model ----------

SECRETS = ("Sam", "Delgado", "Helen", "Okafor", "Lakeview", "555-0177", "00482913")
ITEMS = [{"category": "medication", "text": "Take colchicine 0.6 mg twice daily for 10 days.",
          "source_quote": "Take colchicine 0.6 mg by mouth twice daily for 10 days.",
          "details": {"name": "Colchicine"}}]


def sent_text(fake_llm) -> str:
    return json.dumps([c[0] for c in [*fake_llm.calls, *fake_llm.routed_calls]])


async def test_upload_and_ask_send_no_identifier(client, fake_llm):
    assert isinstance(client.app.state.llm, PrivateLLM)
    h = await auth_headers(client)
    resp = await upload_text(client, h, fake_llm, text=DOC, items=ITEMS)
    assert resp.status_code == 200, resp.text
    # The patient still sees the original.
    assert any("Sam Delgado" in p for p in resp.json()["paragraphs"])
    assert (await client.post("/api/documents/active/confirm", headers=h)).status_code == 200
    paragraphs = resp.json()["paragraphs"]
    n = next(i for i, p in enumerate(paragraphs, start=1) if "555-0177" in p)
    # The model answers with the placeholder it was shown; the patient gets the number.
    fake_llm.script(answer(f"Call the clinic at [PHONE_1] if you have diarrhea [{n}].", [n]))
    body = (await post_ask(client, h, "I'm Sam Delgado. Who do I call about diarrhea?")).json()
    assert body["kind"] == "answer"
    assert body["answer"] == f"Call the clinic at 555-0177 if you have diarrhea [{n}]."
    sent = sent_text(fake_llm)
    for secret in SECRETS:
        assert secret not in sent, secret
    assert "[PATIENT_1]" in sent and "colchicine 0.6 mg" in sent
    # The conversation is stored as the patient wrote it.
    doctor_id = await default_doctor(client)
    messages = (await client.get(f"/api/conversation?doctor_id={doctor_id}", headers=h)).json()
    assert messages["messages"][0]["content"] == "I'm Sam Delgado. Who do I call about diarrhea?"


def test_masking_can_be_turned_off(tmp_path):
    from app import create_app
    from config import Settings
    from druglabels import DEMO_LABELS, LabelClient, fixture_transport
    from knowledge import KnowledgeStore
    fake = FakeLLM()
    app = create_app(
        Settings(database_url="sqlite+aiosqlite:///:memory:", jwt_secret="test-secret-not-for-production",
                 privacy_masking=False),
        llm=fake, store=KnowledgeStore(tmp_path / "index.sqlite3", fake),
        label_client=LabelClient(fixture_transport(DEMO_LABELS)))
    assert app.state.llm is fake
