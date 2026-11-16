"""The doctor's inbox: newest first, and repeated asks collapse into one card."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from conftest import auth_headers, doctor_headers
from models import CareTeamQuestion, DoctorGuidance, User
from routes_care_team import add_question

# Recent, so a waiting row is not yet timed out into the doctor's open list.
BASE = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0) - timedelta(minutes=90)


async def ids(client, *usernames):
    async with client.app.state.session_factory() as s:
        return [(await s.scalar(select(User).where(User.username == n))).id for n in usernames]


async def seed(client, rows, user="alice", doctor="dr.ting", document_id=None):
    """rows: (question, minutes after BASE, extra column values)."""
    uid, did = await ids(client, user, doctor)
    out = []
    async with client.app.state.session_factory() as s:
        for text, minute, extra in rows:
            q = CareTeamQuestion(
                user_id=uid, document_id=extra.pop("document_id", document_id), question=text,
                reason="R", source="agent", doctor_id=extra.pop("doctor_id", did),
                created_at=BASE + timedelta(minutes=minute), **extra)
            s.add(q)
            await s.flush()
            out.append(q.id)
        await s.commit()
    return out


async def inbox(client, dh):
    return (await client.get("/api/doctor/questions", headers=dh)).json()


async def setup(client):
    await auth_headers(client)
    return await doctor_headers(client)


async def test_newest_question_is_first_and_duplicates_form_one_card(client):
    dh = await setup(client)
    old = await seed(client, [("Old one?", 0, {}), ("Old two?", 1, {}), ("Old three?", 2, {})])
    dup = await seed(client, [("What causes gout?", 3, {}), ("what causes  gout", 4, {})])
    new = await seed(client, [("Can I drink juice when taking medicine?", 10, {})])
    data = await inbox(client, dh)
    got = [q["id"] for q in data["open"]]
    assert got[0] == new[0]
    # 3 old + 1 grouped card + 1 new
    assert len(got) == 5
    card = next(q for q in data["open"] if q["id"] == dup[1])
    assert card["asked_count"] == 2
    assert dup[0] not in got
    assert got == [new[0], dup[1], old[2], old[1], old[0]]


async def test_waiting_list_is_newest_first(client):
    dh = await setup(client)
    a, b = await seed(client, [("First?", 0, {"status": "waiting_on_patient"}),
                               ("Second?", 5, {"status": "waiting_on_patient"})])
    assert [q["id"] for q in (await inbox(client, dh))["waiting"]] == [b, a]


async def test_bumped_question_sorts_as_recent(client):
    dh = await setup(client)
    a, b = await seed(client, [("Asked early?", 0, {"asked_count": 3,
                                                     "last_asked_at": BASE + timedelta(minutes=60)}),
                               ("Asked later?", 30, {})])
    data = await inbox(client, dh)
    assert [q["id"] for q in data["open"]] == [a, b]
    assert data["open"][0]["asked_count"] == 3


async def test_group_sums_asked_count(client):
    dh = await setup(client)
    a, b = await seed(client, [("Same?", 0, {"asked_count": 2}), ("Same?", 1, {})])
    (card,) = (await inbox(client, dh))["open"]
    assert card["id"] == b and card["asked_count"] == 3


async def test_answering_group_answers_all_and_records_guidance_once(client):
    dh = await setup(client)
    a, b, c = await seed(client, [("Miss a dose?", 0, {}), ("miss a dose?", 1, {}),
                                  ("Miss a dose??", 2, {})])
    other = (await seed(client, [("Something else?", 3, {})]))[0]
    resp = await client.post(
        f"/api/doctor/questions/{c}/answer", headers=dh,
        json={"answer": "Take it when you remember.", "general": False, "saved_question": None,
              "drug_tags": [], "condition_tags": []})
    assert resp.status_code == 200, resp.text
    data = await inbox(client, dh)
    assert [q["id"] for q in data["open"]] == [other]
    assert {q["id"] for q in data["answered"]} == {a, b, c}
    assert all(q["answer"] == "Take it when you remember." for q in data["answered"])
    async with client.app.state.session_factory() as s:
        assert len(list(await s.scalars(select(DoctorGuidance)))) == 1


async def test_different_doctor_or_document_never_groups(client):
    dh = await setup(client)
    rows = await seed(client, [("Same?", 0, {}), ("Same?", 1, {"doctor_id": None}),
                               ("Same?", 2, {"document_id": 99999})])
    data = await inbox(client, dh)
    assert len(data["open"]) == 3
    assert {q["asked_count"] for q in data["open"]} == {1}


async def test_add_question_bumps_instead_of_inserting(client):
    await setup(client)
    uid, did = await ids(client, "alice", "dr.ting")
    async with client.app.state.session_factory() as s:
        first = await add_question(s, uid, None, "Can I drink juice?", "R", "agent", doctor_id=did)
        again = await add_question(s, uid, None, "  can I  drink juice ?! ", "R2", "agent", doctor_id=did)
        await s.commit()
        assert again.id == first.id
        assert again.asked_count == 2 and again.last_asked_at is not None
        assert len(list(await s.scalars(select(CareTeamQuestion)))) == 1


async def test_add_question_keeps_separate_rows_across_doctor_document_and_waiting(client):
    await setup(client)
    await doctor_headers(client, "dr.lee", "Dr. Lee")
    uid, did, lee = await ids(client, "alice", "dr.ting", "dr.lee")
    async with client.app.state.session_factory() as s:
        base = await add_question(s, uid, None, "Same?", "R", "agent", doctor_id=did)
        other_doc = await add_question(s, uid, 777, "Same?", "R", "agent", doctor_id=did)
        other_doctor = await add_question(s, uid, None, "Same?", "R", "agent", doctor_id=lee)
        no_doctor = await add_question(s, uid, None, "Same?", "R", "agent")
        waiting = await add_question(s, uid, None, "Same?", "R", "agent", doctor_id=did,
                                     status="waiting_on_patient")
        waiting2 = await add_question(s, uid, None, "Same?", "R", "agent", doctor_id=did,
                                      status="waiting_on_patient")
        later = await add_question(s, uid, None, "Same?", "R", "agent", doctor_id=did)
        await s.commit()
        distinct = {r.id for r in (base, other_doc, other_doctor, no_doctor, waiting, waiting2)}
        assert len(distinct) == 6
        assert later.id == base.id and base.asked_count == 2
        assert waiting.asked_count == 1 and waiting2.asked_count == 1
