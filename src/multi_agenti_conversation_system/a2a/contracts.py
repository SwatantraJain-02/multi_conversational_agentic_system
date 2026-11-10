"""Agent contracts: what a caller sends an agent over A2A, and what it gets back.

    caller ── TaskRequest ──► agent          (a fresh task)
    caller ── ResumeRequest ─► agent          (answer to an input-required task, e.g. HITL)
    caller ◄── TaskResult ──── agent          (completed, input_required or failed)

Each model travels as ONE A2A `Part` holding the model's JSON, with media type
`application/json` (`to_part` / `from_parts`). Not an A2A DataPart: a DataPart is a
protobuf `Value`, which stores every number as a double, so SQL rows would come back with
`product_id` 100 as 100.0. The JSON text round-trips exactly.

Both ends validate with these models, so a malformed message is caught at the boundary
(`ContractError`) instead of deep inside an agent. The agent card publishes the JSON
schemas through an A2A extension (`contract_extension`): together with the skills on the
card, these are the "agent contracts".

The trace context is not in these models: it travels in the A2A message metadata, so a
ResumeRequest (which carries a decision, not a task) keeps the caller's trace too.
"""

import uuid
from collections.abc import Sequence
from typing import Any, Literal, Self

from a2a.types import AgentExtension, Part
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONTRACT_MEDIA_TYPE = "application/json"
CONTRACT_EXTENSION_URI = "urn:multi-agent-capstone:a2a-contracts:v1"

MAX_TASK_CHARS = 4000
# Size cap on one contract part, checked before parsing (the A2A boundary guard).
MAX_PART_CHARS = 1_000_000

ErrorCode = Literal[
    "timeout",  # no answer within the deadline
    "unavailable",  # card fetch or connection failed, or the HTTP call was refused
    "invalid_output",  # the agent's reply didn't match TaskResult
    "bad_request",  # the request didn't match TaskRequest / the agent's skills
    "failed",  # the agent ran and reported a failure (or the task ended in failed)
    "internal",  # the agent crashed; the executor caught it
]


class ContractError(ValueError):
    """A message part is missing, too large, or doesn't match the contract model."""


class AgentError(Exception):
    """Raised by an agent's own code to fail the task with a specific, caller-visible error.

    The executor turns it into `TaskResult.failure(code, message)`. Use it for conditions
    the caller can act on (a dependency is down); anything else raised is reported as a
    generic `internal` error, without its text.
    """

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskContext(_Contract):
    """Only the memory relevant to this task (top-k), never the whole store."""

    profile: dict[str, Any] = Field(default_factory=dict)
    memories: list[str] = Field(default_factory=list)
    session: dict[str, Any] = Field(default_factory=dict)


class TaskConstraints(_Contract):
    timeout_s: float | None = Field(default=None, gt=0)
    max_rows: int | None = Field(default=None, ge=1)


class TaskRequest(_Contract):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_task_id: str | None = None  # set when an agent delegates to another agent
    sender: str
    skill: str
    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    # Skill-specific input, e.g. synthesis takes {"internal": TaskResult, "external": ...}.
    data: dict[str, Any] = Field(default_factory=dict)
    context: TaskContext = Field(default_factory=TaskContext)
    constraints: TaskConstraints = Field(default_factory=TaskConstraints)


class ResumeRequest(_Contract):
    """The caller's answer to an input-required task. `value` becomes Command(resume=...)."""

    sender: str
    value: dict[str, Any]


class TaskError(_Contract):
    code: ErrorCode
    message: str
    retryable: bool = False


class TaskResult(_Contract):
    status: Literal["completed", "input_required", "failed"]
    agent: str
    answer: str = ""
    # Structured output: rows, the HITL payload when input_required, ...
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sql: str | None = None
    error: TaskError | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    # A2A ids, filled in by the client. Needed to resume an input_required task.
    a2a_task_id: str | None = None
    context_id: str | None = None

    @classmethod
    def failure(
        cls, agent: str, code: ErrorCode, message: str, *, retryable: bool = False
    ) -> Self:
        return cls(
            status="failed",
            agent=agent,
            error=TaskError(code=code, message=message, retryable=retryable),
        )

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class SynthesisData(_Contract):
    """`TaskRequest.data` for the Synthesis agent's skill `synthesise_findings`.

    The question is `TaskRequest.task`. It lives here, with the other contracts, because
    the supervisor must build it without importing any agent.
    """

    internal: TaskResult | None = None  # what the SQL agent found
    external: TaskResult | None = None  # what the Web agent found
    # When `external` is missing or unusable, Synthesis asks the Web agent itself. This is
    # the question for it (the external side of the user's question only). Optional.
    external_question: str | None = Field(default=None, max_length=MAX_TASK_CHARS)


# ------------------------------------------------------------------ wire format
def to_part(model: BaseModel) -> Part:
    return Part(text=model.model_dump_json(), media_type=CONTRACT_MEDIA_TYPE)


def from_parts[M: BaseModel](parts: Sequence[Part], model: type[M]) -> M:
    """Parse the first contract part into `model`, or raise ContractError."""
    for part in parts:
        if part.media_type == CONTRACT_MEDIA_TYPE and part.HasField("text"):
            if len(part.text) > MAX_PART_CHARS:
                raise ContractError(
                    f"{model.__name__} is too large ({len(part.text)} chars)"
                )
            try:
                return model.model_validate_json(part.text)
            except ValidationError as exc:
                raise ContractError(_short(model, exc)) from exc
    raise ContractError(f"no {CONTRACT_MEDIA_TYPE} part holding a {model.__name__}")


def _short(model: type[BaseModel], exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"]) or "(root)"
    return f"invalid {model.__name__}: {where}: {first['msg']} ({exc.error_count()} error(s))"


def to_struct(data: dict[str, Any]) -> Struct:
    return ParseDict(data, Struct())


def contract_extension(
    skill_data: dict[str, type[BaseModel] | None],
) -> AgentExtension:
    """Card extension that publishes the request/result schemas, per skill.

    `skill_data` maps each skill id to the model of its `TaskRequest.data` (None when the
    skill takes only the task text).
    """
    return AgentExtension(
        uri=CONTRACT_EXTENSION_URI,
        description="Every message carries one application/json part: a TaskRequest or "
        "ResumeRequest in, a TaskResult out. Schemas below.",
        required=True,
        params=to_struct(
            {
                "request_schema": TaskRequest.model_json_schema(),
                "resume_schema": ResumeRequest.model_json_schema(),
                "result_schema": TaskResult.model_json_schema(),
                "skills": {
                    skill: {"data_schema": m.model_json_schema() if m else None}
                    for skill, m in skill_data.items()
                },
            }
        ),
    )
