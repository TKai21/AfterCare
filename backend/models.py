from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, ForeignKey, Integer, LargeBinary, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.dialects.mysql import LONGBLOB, LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="patient", server_default="patient")
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # A doctor's specialty as patients see it ("Rheumatology"); None for patients.
    specialty: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DischargeDocument(Base):
    __tablename__ = "discharge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # pending_review | confirmed | archived
    status: Mapped[str] = mapped_column(String(20), default="pending_review")
    filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT, "mysql"))
    discharge_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    anchor_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # document | confirmation
    date_source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    unverified: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Equals user_id while the document is pending_review/confirmed, NULL once archived.
    # A plain UNIQUE index (NULLs do not collide) enforces one active document per
    # user on both MySQL and SQLite.
    active_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, unique=True)


class DocumentFile(Base):
    """The uploaded PDF itself, shown beside the instructions. Its own table so
    loading a document never reads the bytes, and so an existing database
    gains it from create_all without a migration."""
    __tablename__ = "document_files"

    document_id: Mapped[int] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="CASCADE"), primary_key=True
    )
    content_type: Mapped[str] = mapped_column(String(100))
    data: Mapped[bytes] = mapped_column(LargeBinary().with_variant(LONGBLOB, "mysql"))


class ExtractedItem(Base):
    __tablename__ = "extracted_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(50))
    text: Mapped[str] = mapped_column(Text)
    source_quote: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    # active | rejected
    status: Mapped[str] = mapped_column(String(20), default="active")


class ChecklistItem(Base):
    __tablename__ = "checklist_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="CASCADE"), index=True
    )
    extracted_item_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("extracted_items.id", ondelete="SET NULL"), nullable=True
    )
    label: Mapped[str] = mapped_column(String(500))
    # morning | midday | evening | bedtime | anytime
    slot: Mapped[str] = mapped_column(String(20), default="anytime")
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    one_time: Mapped[bool] = mapped_column(Boolean, default=False)
    approximate: Mapped[bool] = mapped_column(Boolean, default=False)
    undated: Mapped[bool] = mapped_column(Boolean, default=False)


class ChecklistCompletion(Base):
    __tablename__ = "checklist_completions"
    __table_args__ = (UniqueConstraint("checklist_item_id", "day", name="uq_completion_item_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    checklist_item_id: Mapped[int] = mapped_column(
        ForeignKey("checklist_items.id", ondelete="CASCADE"), index=True
    )
    day: Mapped[date] = mapped_column(Date)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="CASCADE"), index=True
    )
    # user | assistant
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # The doctor whose clone this conversation is with (NULL for older messages).
    doctor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class CareTeamQuestion(Base):
    __tablename__ = "care_team_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="SET NULL"), nullable=True
    )
    question: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    # agent | rejected_item | validation_fallback
    source: Mapped[str] = mapped_column(String(30))
    # open | answered | resolved
    status: Mapped[str] = mapped_column(String(20), default="open", server_default="open")
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    answered_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # The doctor this question is for; NULL means any doctor may answer it.
    doctor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DoctorGuidance(Base):
    __tablename__ = "doctor_guidance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    # general | patient
    scope: Mapped[str] = mapped_column(String(20))
    patient_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="SET NULL"), nullable=True
    )
    source_question_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("care_team_questions.id", ondelete="SET NULL"), nullable=True
    )
    drug_tags: Mapped[list] = mapped_column(JSON, default=list)
    condition_tags: Mapped[list] = mapped_column(JSON, default=list)
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    embedding: Mapped[Optional[list]] = mapped_column(JSON(none_as_null=True), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # The rules read from the answer (guidance_rules): None when not tried (no
    # model), "ready" when there are rules, "failed" when reading them failed.
    rules_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class GuidanceRule(Base):
    """One standalone rule read from a guidance entry's answer. `quote` is an
    exact span of the answer: it, not the restated `statement`, is the evidence."""
    __tablename__ = "guidance_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    guidance_id: Mapped[int] = mapped_column(
        ForeignKey("doctor_guidance.id", ondelete="CASCADE"), index=True
    )
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    statement: Mapped[str] = mapped_column(Text)
    quote: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Optional[list]] = mapped_column(JSON(none_as_null=True), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class GuidanceConflict(Base):
    """A newer rule that a patient could not follow together with the doctor's
    earlier rule (`other_rule_id`) or an approved drug label passage (`label_*`)."""
    __tablename__ = "guidance_conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    rule_id: Mapped[int] = mapped_column(
        ForeignKey("guidance_rules.id", ondelete="CASCADE"), index=True
    )
    other_rule_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("guidance_rules.id", ondelete="CASCADE"), nullable=True, index=True
    )
    label_rxcui: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    label_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    label_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    label_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[str] = mapped_column(Text)
    # open | dismissed
    status: Mapped[str] = mapped_column(String(20), default="open", server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DrugLabel(Base):
    """One FDA label per ingredient, shared by all patients. A doctor approves it
    before the library may use it (see the drug label design spec, section 6)."""
    __tablename__ = "drug_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rxcui: Mapped[str] = mapped_column(String(20), unique=True)
    ingredient: Mapped[str] = mapped_column(String(200))
    # fetching | pending | approved | rejected | no_label | failed
    status: Mapped[str] = mapped_column(String(20), default="fetching")
    # Up to 5 {setid, title, published, labeler}, newest first.
    candidates: Mapped[list] = mapped_column(JSON, default=list)
    setid: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    text: Mapped[Optional[str]] = mapped_column(
        Text().with_variant(LONGTEXT, "mysql"), nullable=True
    )
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    collected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class DocumentDrug(Base):
    """One confirmed medication item and one of its ingredients. A combination
    product has several rows. rxcui is NULL while unresolved or failed."""
    __tablename__ = "document_drugs"
    __table_args__ = (UniqueConstraint("extracted_item_id", "rxcui", name="uq_document_drugs_item_rxcui"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("discharge_documents.id", ondelete="CASCADE"), index=True
    )
    extracted_item_id: Mapped[int] = mapped_column(
        ForeignKey("extracted_items.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    rxcui: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # fetching | resolved | unresolved | failed
    status: Mapped[str] = mapped_column(String(20), default="fetching")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
