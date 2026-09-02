"""Collect durable, machine-readable evidence from local and Coiled runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landsat_lst.evidence_contract import load_contract

_SECRET_PARTS = ("token", "secret", "password", "credential", "access_key", "session_key")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|ACCESS_KEY|SESSION_KEY)[A-Z0-9_]*)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+")


def sha256_file(path: Path) -> str:
    """Return the content digest of one retained artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact_text(value: str) -> str:
    """Redact credential assignments and secret values embedded in log text."""
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", value)
    redacted = _BEARER.sub(r"\1<redacted>", redacted)
    for key, secret in os.environ.items():
        if len(secret) >= 4 and any(part in key.lower() for part in _SECRET_PARTS):
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _safe(value: Any) -> Any:
    """Make API data JSON-safe and redact credential-shaped fields."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if any(part in str(key).lower() for part in _SECRET_PARTS)
                else _safe(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]

    safe_value = value
    if isinstance(value, datetime):
        safe_value = value.isoformat()
    elif isinstance(value, str):
        safe_value = _redact_text(value)
    elif isinstance(value, Path):
        safe_value = str(value)
    try:
        json.dumps(safe_value)
    except (TypeError, ValueError):
        safe_value = repr(safe_value)
    return safe_value


def _git_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(  # nosec B603 B607
            ["git", *args], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def _run_artifacts(run_id: str, storage: Any) -> list[dict[str, Any]]:
    """Collect whole-tile plus shard state/profile/log evidence, never intermediates."""
    keys: dict[str, Any] = dict(storage.list_prefix(storage.run_prefix(run_id)))
    shard_prefix = f"_shards/{run_id}/"
    for key, modified in storage.list_prefix(shard_prefix).items():
        if "/state/" in key or key == f"{shard_prefix}fleet.json":
            keys[key] = modified
    timing_prefix = f"_shards/timings/{run_id}/"
    keys.update(storage.list_prefix(timing_prefix))

    artifacts = []
    for key, modified in sorted(keys.items()):
        text = storage.read_text(key)
        raw = text.encode() if text is not None else b""
        item: dict[str, Any] = {
            "key": key,
            "modified": _safe(modified),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if key.endswith(".json") and text is not None:
            try:
                item["content"] = _safe(json.loads(text))
            except json.JSONDecodeError:
                item["parse_error"] = "invalid JSON"
        artifacts.append(item)
    return artifacts


def _resolve_artifact(contract_path: Path, value: str) -> Path:
    source = Path(value)
    return source if source.is_absolute() else contract_path.parent / source


def _validate_equivalence_result(source: Path) -> None:
    """Require an explicit passing post-run scientific comparison result."""
    try:
        result = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read output-equivalence result {source}: {exc}") from exc
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError(
            f"output-equivalence result must be a JSON object with passed=true: {source}"
        )


def _retain_artifact(source: Path, output_dir: Path, stem: str) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_dir = output_dir / "contract-artifacts"
    destination_dir.mkdir(exist_ok=True)
    destination = destination_dir / f"{stem}{source.suffix}"
    shutil.copy2(source, destination)
    return {
        "path": str(destination.relative_to(output_dir)),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _worker_code_verification(
    artifacts: list[dict[str, Any]], expected_revision: str
) -> dict[str, Any]:
    """Verify identities emitted by worker processes against the contract."""
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in artifacts:
        content = artifact.get("content")
        identity = content.get("code_identity") if isinstance(content, dict) else None
        if not isinstance(identity, dict):
            continue
        marker = json.dumps(identity, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            identities.append(identity)
    revisions: set[str] = set()
    for identity in identities:
        revision = identity.get("revision")
        if isinstance(revision, str):
            revisions.add(revision)
    if not revisions:
        raise ValueError("worker code identity is unavailable in retained run artifacts")
    mismatches = sorted(revision for revision in revisions if revision != expected_revision)
    if mismatches:
        raise ValueError(
            "worker code revision does not match contract treatment revision: "
            + ", ".join(mismatches)
        )
    return {
        "expected_revision": expected_revision,
        "status": "verified",
        "identities": identities,
    }


def _coiled_cluster(
    cluster_id: int, workspace: str, metric_queries: tuple[str, ...]
) -> dict[str, Any]:
    import coiled  # noqa: PLC0415

    record: dict[str, Any] = {"cluster_id": cluster_id, "workspace": workspace}
    with coiled.Cloud(workspace=workspace) as cloud:
        details = cloud.cluster_details(cluster_id, workspace=workspace)
        if not isinstance(details, dict) or not isinstance(details.get("name"), str):
            raise RuntimeError(f"Coiled cluster {cluster_id} details did not contain a string name")
        for private_name in ("server", "_do_request", "_sync"):
            if not hasattr(cloud, private_name):
                version = getattr(coiled, "__version__", "unknown")
                raise RuntimeError(
                    f"Coiled {version} lacks collector API {private_name}; update the collector"
                )
        record["details"] = _safe(details)
        record["logs"] = _safe(dict(cloud.cluster_logs(cluster_id, workspace=workspace)))
        cluster_name = details["name"]
        pages = []
        for page in range(1, 101):
            result = coiled.get_billing_activity(account=workspace, cluster=cluster_name, page=page)
            pages.append(_safe(result))
            if not isinstance(result, dict) or not result.get("next"):
                break
        record["billing_pages"] = pages

        async def fetch_metric(query: str) -> Any:
            url = f"{cloud.server}/api/v2/metrics/account/{workspace}/cluster/{cluster_id}"
            response = await cloud._do_request("GET", url, params={"query": query})
            if response.status >= 400:
                return {"status": response.status, "error": await response.text()}
            return await response.json()

        record["metrics"] = {
            query: _safe(cloud._sync(fetch_metric, query)) for query in metric_queries
        }

        async def fetch_spans() -> Any:
            url = f"{cloud.server}/api/v2/analytics/{workspace}/cluster/{cluster_id}/spans"
            response = await cloud._do_request("GET", url)
            if response.status >= 400:
                return {"status": response.status, "error": await response.text()}
            return await response.json()

        record["coiled_spans"] = _safe(cloud._sync(fetch_spans))
    return record


def collect_evidence(
    *,
    output_dir: Path,
    contract_path: Path,
    run_id: str | None = None,
    cluster_ids: tuple[int, ...] = (),
    workspace: str | None = None,
    metric_queries: tuple[str, ...] = (),
    attachments: tuple[Path, ...] = (),
    storage: Any | None = None,
) -> Path:
    """Write one canonical evidence bundle and return its manifest path."""
    contract = load_contract(contract_path)
    destination = output_dir / "evidence.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence bundle: {destination}")
    names = [source.name for source in attachments]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError("attachment basenames must be unique: " + ", ".join(duplicate_names))

    equivalence_source = _resolve_artifact(
        contract_path, contract["output_equivalence"]["result_artifact"]
    )
    _validate_equivalence_result(equivalence_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_artifacts = {
        "baseline": _retain_artifact(
            _resolve_artifact(contract_path, contract["baseline"]["artifact"]),
            output_dir,
            "baseline",
        ),
        "output_equivalence": _retain_artifact(
            equivalence_source,
            output_dir,
            "output-equivalence",
        ),
    }
    attachment_dir = output_dir / "attachments"
    attachment_records = []
    for source in attachments:
        if not source.is_file():
            raise FileNotFoundError(source)
        attachment_dir.mkdir(exist_ok=True)
        retained = attachment_dir / source.name
        if retained.exists():
            raise FileExistsError(f"refusing to overwrite attachment: {retained}")
        shutil.copy2(source, retained)
        attachment_records.append(
            {
                "path": str(retained.relative_to(output_dir)),
                "bytes": retained.stat().st_size,
                "sha256": sha256_file(retained),
            }
        )

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "collected_at": datetime.now(UTC).isoformat(),
        "contract": contract,
        "contract_artifacts": contract_artifacts,
        "code": {
            **_git_identity(),
            "package_version": importlib.metadata.version("landsat-lst"),
        },
        "attachments": attachment_records,
        "evidence_classes": {
            "measured": "directly observed and retained here",
            "derived": "calculated from retained measured inputs",
            "assumed": "not observed",
            "user_reported": "reported externally, not independently retained",
            "unknown": "not presently knowable from retained evidence",
        },
    }
    if run_id is not None:
        if storage is None:
            from landsat_lst.storage import get_storage  # noqa: PLC0415

            storage = get_storage()
        bundle["run_id"] = run_id
        run_artifacts = _run_artifacts(run_id, storage)
        bundle["run_artifacts"] = run_artifacts
        bundle["worker_code_verification"] = _worker_code_verification(
            run_artifacts, contract["inputs"]["treatment_revision"]
        )
    else:
        bundle["worker_code_verification"] = {"status": "not_collected", "identities": []}
    if cluster_ids:
        if not workspace:
            raise ValueError("workspace is required with cluster IDs")
        bundle["coiled_clusters"] = [
            _coiled_cluster(cid, workspace, metric_queries) for cid in cluster_ids
        ]

    destination.write_text(json.dumps(_safe(bundle), indent=2, sort_keys=True) + "\n")
    return destination


def _run_frisky(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run Frisky and turn process failures into concise operator errors."""
    try:
        return subprocess.run(  # nosec B603 B607
            ["frisky", *args], check=True, capture_output=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError("frisky executable was not found; install the frisky extra") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip() if exc.stderr else "no stderr"
        raise RuntimeError(f"frisky {' '.join(args[:2])} failed: {detail}") from exc


def capture_frisky(source: str, output_dir: Path, *, limit: int = 1_000_000_000) -> list[Path]:
    """Persist Frisky spans plus an agent-readable offline overview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)
    spans = output_dir / "frisky-spans.json"
    if source_path.is_file():
        shutil.copy2(source_path, spans)
    else:
        result = _run_frisky(["observe", "spans", source, "--limit", str(limit)])
        spans.write_bytes(result.stdout)

    overview = output_dir / "frisky-overview.json"
    result = _run_frisky(["observe", "overview", str(spans), "--json"])
    overview.write_bytes(result.stdout)
    return [spans, overview]
