"""Phase P0 schemas - strict contracts for ControlCore calls and results."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import re
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
HANDLE_RE = re.compile(r"^[A-Za-z0-9 _.\-]{3,64}$")


class SchemaVersioned(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verbosity(str, Enum):
    minimal = "minimal"
    standard = "standard"
    full = "full"


class Determinism(str, Enum):
    strict = "strict"
    best_effort = "best_effort"
    off = "off"


class RedactionMode(str, Enum):
    auto = "auto"
    off = "off"  # only allowed with explicit override ack


class IntentClass(str, Enum):
    lookup = "lookup"
    summarize = "summarize"
    extract = "extract"
    compare = "compare"
    draft = "draft"
    classify = "classify"
    reason = "reason"
    critique = "critique"
    translate = "translate"
    unknown = "unknown"


class TargetType(str, Enum):
    model = "model"
    tool = "tool"  # must be read-only (enforced by bouncer)


class TrustTier(str, Enum):
    trusted = "trusted"
    standard = "standard"
    untrusted = "untrusted"


class CallStatus(str, Enum):
    complete = "complete"
    queued = "queued"
    running = "running"
    failed = "failed"


class Caller(SchemaVersioned):
    handle: str = Field(..., description="User handle, not an email.")
    account_id: str = Field(..., description="Opaque UUID-like string.")
    key_id: Optional[str] = Field(None, description="Public key fingerprint or key identifier.")
    fingerprint_ref: Optional[str] = Field(
        None, description="Opaque reference to cognitive fingerprint attestation, not the fingerprint itself."
    )

    @field_validator("handle")
    @classmethod
    def validate_handle(cls, v: str) -> str:
        if not HANDLE_RE.match(v):
            raise ValueError("Invalid handle format")
        return v

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except Exception:
            if len(v) < 8:
                raise ValueError("account_id must be UUID-like or sufficiently long opaque id")
        return v


class Intent(SchemaVersioned):
    cls: IntentClass = Field(..., alias="class")
    detail: Optional[str] = Field(None, max_length=512)


class Target(SchemaVersioned):
    type: TargetType
    alias: str = Field(..., min_length=1, max_length=128, description="Registry alias, like qwen:32b or api_hub:search")
    trust_tier: TrustTier = TrustTier.standard
    capability_tags: List[str] = Field(default_factory=list)

    @field_validator("capability_tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        if len(v) > 64:
            raise ValueError("Too many capability tags")
        for t in v:
            if len(t) > 64:
                raise ValueError("Capability tag too long")
        return v


class Timeouts(SchemaVersioned):
    soft_ms: int = Field(15000, ge=1000, le=600000)
    hard_ms: int = Field(60000, ge=1000, le=1200000)


class Params(SchemaVersioned):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)


class RedactionOverride(SchemaVersioned):
    enabled: bool = False
    acknowledgements: List[str] = Field(default_factory=list, description="Explicit opt-in phrases.")
    reason: Optional[str] = Field(None, max_length=256)


class RedactionPolicy(SchemaVersioned):
    mode: RedactionMode = RedactionMode.auto
    override: Optional[RedactionOverride] = None


class CallOptions(SchemaVersioned):
    verbosity: Verbosity = Verbosity.standard
    determinism: Determinism = Determinism.best_effort
    timeouts: Timeouts = Field(default_factory=Timeouts)
    redaction: RedactionPolicy = Field(default_factory=RedactionPolicy)
    allow_variants: bool = True
    max_variants: int = Field(5, ge=1, le=5)


class ContextPart(SchemaVersioned):
    part_id: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=2_000_000)
    sha256: Optional[str] = Field(None, description="Optional precomputed hash for audit anchoring.")


class ControlCoreCall(SchemaVersioned):
    schema_version: str = Field("1.0.0")
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    caller: Caller
    intent: Intent
    target: Target

    prompt: str = Field(..., min_length=1, max_length=2_000_000)
    context: List[ContextPart] = Field(default_factory=list)

    params: Params = Field(default_factory=Params)
    options: CallOptions = Field(default_factory=CallOptions)

    signature: Optional[str] = Field(None, description="Optional signature for the call envelope.")
    entrypoint: Optional[str] = Field(
        None,
        description="Must remain unused by ControlCore. Exists for auditing explicit entrypoints elsewhere.",
    )

    @field_validator("schema_version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        if not SEMVER_RE.match(v):
            raise ValueError("schema_version must be semver like 1.0.0")
        return v

    @model_validator(mode="after")
    def validate_redaction_override(self) -> "ControlCoreCall":
        if self.options.redaction.mode == RedactionMode.off:
            ov = self.options.redaction.override
            if not ov or not ov.enabled:
                raise ValueError("redaction.mode off requires redaction.override.enabled true")
        return self


class ErrorCode(str, Enum):
    validation_error = "validation_error"
    permission_denied = "permission_denied"
    adapter_error = "adapter_error"
    timeout = "timeout"
    refused = "refused"
    unknown = "unknown"


class CallError(SchemaVersioned):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class Confidence(SchemaVersioned):
    self_reported: Optional[float] = Field(None, ge=0.0, le=1.0)
    system_estimate: Optional[float] = Field(None, ge=0.0, le=1.0)
    third_party: Optional[float] = Field(None, ge=0.0, le=1.0)


class Provenance(SchemaVersioned):
    model_alias: str
    model_version: Optional[str] = None
    adapter: Optional[str] = None
    adapter_version: Optional[str] = None
    trust_tier: TrustTier = TrustTier.standard

    started_at: str
    finished_at: Optional[str] = None

    request_hash: Optional[str] = None
    response_hash: Optional[str] = None

    raw: Optional[Dict[str, Any]] = None


class RedactionReportItem(SchemaVersioned):
    kind: str  # "api_key", "token", "email", "phone", "address", "id", "other"
    count: int = Field(ge=1)
    note: Optional[str] = None


class RedactionReport(SchemaVersioned):
    performed: bool = False
    items: List[RedactionReportItem] = Field(default_factory=list)
    user_notice: Optional[str] = None
    override_used: bool = False


class NormalizationReport(SchemaVersioned):
    applied: bool = False
    original_payload_hash: Optional[str] = None
    fixes: List[str] = Field(default_factory=list)
    variants_generated: int = 0


class ControlCoreCallResult(SchemaVersioned):
    schema_version: str = Field("1.0.0")
    call_id: str

    status: CallStatus = CallStatus.complete
    job_id: Optional[str] = None  # for queued/running
    partial: bool = False
    retryable: bool = False

    answer: Optional[str] = None
    answers: Optional[List[str]] = None  # used when returning multiple variants

    confidence: Confidence = Field(default_factory=Confidence)
    provenance: Provenance
    redaction: RedactionReport = Field(default_factory=RedactionReport)
    normalization: NormalizationReport = Field(default_factory=NormalizationReport)

    errors: List[CallError] = Field(default_factory=list)

    signature: Optional[str] = None  # optional signature for result
