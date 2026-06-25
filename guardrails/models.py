"""Pydantic schemas for the Guard Rails ingestion API.

Field names mirror the spec exactly so a capturing client (a hook, an IDE
adapter) can POST a near-1:1 JSON body. Only `prompt_text` is required — an
approval event without the verbatim prompt isn't worth recording — everything
else is optional so a client can send whatever it has and patch the rest later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# A generous hard cap at the schema layer; the daemon additionally truncates to
# the configured GUARDRAILS_MAX_FIELD_CHARS before storing.
_MAX = 100_000


class ApprovalEventIn(BaseModel):
    prompt_text: str = Field(min_length=1, max_length=_MAX,
                             description="The approval prompt, verbatim.")
    user_request: str | None = Field(default=None, max_length=_MAX)
    agent_action: str | None = Field(default=None, max_length=_MAX)
    options: list[str] | None = Field(default=None, max_length=50)
    selected_option: str | None = Field(default=None, max_length=2000)
    result: str | None = Field(default=None, max_length=2000)
    result_detail: str | None = Field(default=None, max_length=_MAX)
    # Provenance / metadata.
    session_id: str | None = Field(default=None, max_length=2000)
    ide: str | None = Field(default=None, max_length=200)
    agent: str | None = Field(default=None, max_length=200)
    repository: str | None = Field(default=None, max_length=2000)
    branch: str | None = Field(default=None, max_length=2000)
    tool_name: str | None = Field(default=None, max_length=200)
    metadata: dict | None = Field(default=None)


class ApprovalEventPatch(BaseModel):
    """Fill in a previously-logged pending event (the decision, then the
    result). Status is recomputed server-side from these fields."""
    selected_option: str | None = Field(default=None, max_length=2000)
    result: str | None = Field(default=None, max_length=2000)
    result_detail: str | None = Field(default=None, max_length=_MAX)
    options: list[str] | None = Field(default=None, max_length=50)
    metadata: dict | None = Field(default=None)
