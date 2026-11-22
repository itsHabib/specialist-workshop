"""Freeze the host contract and operating settings for a reloadable decision checkpoint."""

from dataclasses import dataclass, field

from .contracts import read_json
from .core import digest, load_model, predict, write_new
from .steering import rank_candidates, validate_ranker


FORMAT = "decision-workshop/policy-v1"


def _policy_body(model, kind, adapter_revision, fallback_revision, min_score):
    for name, revision in (("adapter_revision", adapter_revision),
                           ("fallback_revision", fallback_revision)):
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"{name} must be a nonempty revision")
    body = {"format": FORMAT, "kind": kind, "model_sha256": model["sha256"],
            "adapter_revision": adapter_revision, "fallback_revision": fallback_revision}
    if kind == "fixed":
        if type(min_score) not in (float, int) or not 0 <= min_score <= 1:
            raise ValueError("fixed policies require min_score between 0 and 1")
        body["min_score"] = min_score
    elif kind == "ranking":
        if min_score is not None:
            raise ValueError("a ranking policy takes no min_score")
        validate_ranker(model)
        body["tie_order"] = "native"
    else:
        raise ValueError("unsupported policy kind")
    return body


def save_policy(path, checkpoint_path, *, kind, adapter_revision, fallback_revision,
                min_score=None):
    """Write a new policy descriptor and return its digest for a caller to pin."""
    model = load_model(checkpoint_path)
    body = _policy_body(model, kind, adapter_revision, fallback_revision, min_score)
    sha256 = digest(body)
    write_new(path, {**body, "sha256": sha256})
    return sha256


@dataclass(frozen=True)
class DecisionPolicy:
    """A verified policy for one adapter and fallback implementation."""

    sha256: str
    kind: str
    adapter_revision: str
    fallback_revision: str
    _model: dict = field(repr=False)
    _min_score: float | None

    def decide(self, state, candidates):
        """Score the host's currently eligible actions at the frozen cutoff."""
        if self.kind != "fixed":
            raise ValueError("decide requires a fixed policy")
        return predict(self._model, state, candidates, min_score=self._min_score)

    def rank(self, candidates):
        """Rank the current roster. Equal scores preserve the host's native order."""
        if self.kind != "ranking":
            raise ValueError("rank requires a ranking policy")
        if candidates is None:
            raise ValueError("candidate roster is unrecorded")
        return rank_candidates(self._model, candidates, native_ties=True)


def load_policy(path, checkpoint_path, *, adapter_revision, fallback_revision,
                expected_sha256=None):
    """Verify both artifacts and the revisions the host actually supplies."""
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("unsupported policy format")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    sha256 = digest(body)
    if manifest.get("sha256") != sha256:
        raise ValueError("policy digest mismatch")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("policy differs from the pinned digest")
    if body.get("adapter_revision") != adapter_revision:
        raise ValueError("feature adapter revision mismatch")
    if body.get("fallback_revision") != fallback_revision:
        raise ValueError("host fallback revision mismatch")
    model = load_model(checkpoint_path)
    if model["sha256"] != body.get("model_sha256"):
        raise ValueError("policy checkpoint digest mismatch")
    kind = body.get("kind")
    min_score = body.get("min_score")
    if body != _policy_body(model, kind, adapter_revision, fallback_revision, min_score):
        raise ValueError("unsupported policy settings")
    return DecisionPolicy(sha256, kind, adapter_revision, fallback_revision, model, min_score)
