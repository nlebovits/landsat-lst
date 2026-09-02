"""Collect durable, machine-readable evidence from local and Coiled runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from landsat_lst.evidence_contract import load_contract

_SECRET_PARTS = ("token", "secret", "password", "credential", "access_key", "session_key")


def sha256_file(path: Path) -> str:
    """Return the content digest of one retained artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


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
    artifacts = []
    for key, modified in sorted(storage.list_prefix(storage.run_prefix(run_id)).items()):
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


def _coiled_cluster(
    cluster_id: int, workspace: str, metric_queries: tuple[str, ...]
) -> dict[str, Any]:
    import coiled  # noqa: PLC0415

    record: dict[str, Any] = {"cluster_id": cluster_id, "workspace": workspace}
    with coiled.Cloud(workspace=workspace) as cloud:
        details = cloud.cluster_details(cluster_id, workspace=workspace)
        record["details"] = _safe(details)
        record["logs"] = _safe(dict(cloud.cluster_logs(cluster_id, workspace=workspace)))
        cluster_name = details["name"]
        pages = []
        for page in range(1, 101):
            result = coiled.get_billing_activity(cluster=cluster_name, page=page)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    attachment_dir = output_dir / "attachments"
    attachment_records = []
    for source in attachments:
        if not source.is_file():
            raise FileNotFoundError(source)
        attachment_dir.mkdir(exist_ok=True)
        destination = attachment_dir / source.name
        shutil.copy2(source, destination)
        attachment_records.append(
            {
                "path": str(destination.relative_to(output_dir)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "collected_at": datetime.now(UTC).isoformat(),
        "contract": contract,
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
        bundle["run_artifacts"] = _run_artifacts(run_id, storage)
    if cluster_ids:
        if not workspace:
            raise ValueError("workspace is required with cluster IDs")
        bundle["coiled_clusters"] = [
            _coiled_cluster(cid, workspace, metric_queries) for cid in cluster_ids
        ]

    destination = output_dir / "evidence.json"
    destination.write_text(json.dumps(_safe(bundle), indent=2, sort_keys=True) + "\n")
    return destination


def capture_frisky(source: str, output_dir: Path, *, limit: int = 1_000_000_000) -> list[Path]:
    """Persist Frisky spans plus an agent-readable offline overview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)
    spans = output_dir / "frisky-spans.json"
    if source_path.is_file():
        shutil.copy2(source_path, spans)
    else:
        result = subprocess.run(  # nosec B603 B607
            ["frisky", "observe", "spans", source, "--limit", str(limit)],
            check=True,
            capture_output=True,
        )
        spans.write_bytes(result.stdout)

    overview = output_dir / "frisky-overview.json"
    result = subprocess.run(  # nosec B603 B607
        ["frisky", "observe", "overview", str(spans), "--json"],
        check=True,
        capture_output=True,
    )
    overview.write_bytes(result.stdout)
    return [spans, overview]
