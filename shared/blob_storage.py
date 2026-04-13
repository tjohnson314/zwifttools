"""
Azure Blob Storage helper for persisting race data cache.

Race data is stored locally while being processed, but synced to/from
Azure Blob Storage so it survives deployments.  When the connection
string is not configured the module is a no-op — all functions return
gracefully so the app still works with local-only caching.

Configuration (environment variables):
    AZURE_STORAGE_CONNECTION_STRING  — full connection string
    AZURE_STORAGE_CONTAINER          — container name (default: "race-cache")
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy-initialised client (created on first use)
_container_client = None
_initialised = False


def _get_container():
    """Return the BlobContainerClient, or None if not configured."""
    global _container_client, _initialised
    if _initialised:
        return _container_client
    _initialised = True

    conn_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not conn_str:
        return None

    try:
        from azure.storage.blob import BlobServiceClient
        container_name = os.environ.get('AZURE_STORAGE_CONTAINER', 'race-cache')
        service = BlobServiceClient.from_connection_string(conn_str)
        _container_client = service.get_container_client(container_name)
        # Create the container if it doesn't exist
        if not _container_client.exists():
            _container_client.create_container()
        logger.info("Blob storage configured: container=%s", container_name)
    except Exception as e:
        logger.warning("Blob storage init failed: %s", e)
        _container_client = None

    return _container_client


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def is_configured() -> bool:
    """Return True if blob storage is available."""
    return _get_container() is not None


def upload_race_dir(race_id: str, local_dir: str | Path) -> bool:
    """Upload every file in *local_dir* to blob storage under *race_id/*.

    Sub-directories (like ``cleaned_data/``) are uploaded recursively.
    Returns True on success, False if blob storage is unavailable or upload fails.
    """
    container = _get_container()
    if container is None:
        return False

    local_dir = Path(local_dir)
    if not local_dir.exists():
        return False

    try:
        for file_path in local_dir.rglob('*'):
            if not file_path.is_file():
                continue
            blob_name = f"{race_id}/{file_path.relative_to(local_dir).as_posix()}"
            metadata = None
            if file_path.name == 'race_meta.json':
                try:
                    meta = json.loads(file_path.read_text())
                    race_name = meta.get('race_name')
                    if race_name:
                        metadata = {'race_name': race_name}
                except Exception:
                    pass
            with open(file_path, 'rb') as f:
                container.upload_blob(blob_name, f, overwrite=True, metadata=metadata)
        return True
    except Exception as e:
        logger.error("Blob upload failed for %s: %s", race_id, e)
        return False


def download_race_dir(race_id: str, local_dir: str | Path) -> bool:
    """Download all blobs under *race_id/* into *local_dir*.

    Returns True if at least one file was downloaded, False otherwise.
    """
    container = _get_container()
    if container is None:
        return False

    local_dir = Path(local_dir)
    try:
        blobs = list(container.list_blobs(name_starts_with=f"{race_id}/"))
        if not blobs:
            return False

        for blob in blobs:
            # blob.name is e.g. "race_data_123456/race_meta.json"
            relative = blob.name[len(race_id) + 1:]  # strip prefix + '/'
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'wb') as f:
                data = container.download_blob(blob.name).readall()
                f.write(data)
        return True
    except Exception as e:
        logger.error("Blob download failed for %s: %s", race_id, e)
        return False


def race_exists_in_blob(race_id: str) -> bool:
    """Check whether *race_id* has cached data in blob storage.

    Looks for the ``complete_race_summary.csv`` sentinel file.
    """
    container = _get_container()
    if container is None:
        return False

    try:
        blob_name = f"{race_id}/complete_race_summary.csv"
        blob_client = container.get_blob_client(blob_name)
        return blob_client.exists()
    except Exception:
        return False


def list_races() -> List[dict]:
    """List all race IDs stored in blob storage.

    Returns a list of dicts with at least ``race_id`` and optionally
    ``race_name`` (read from blob metadata attached during upload).

    Uses a single ``list_blobs`` call with ``include=['metadata']`` so
    no per-race downloads are needed.
    """
    container = _get_container()
    if container is None:
        return []

    try:
        seen = set()
        meta_names = {}
        # Single pass: collect all race IDs and race names from metadata
        for blob in container.list_blobs(name_starts_with="race_data_", include=['metadata']):
            race_id = blob.name.split('/')[0]
            seen.add(race_id)
            if blob.name.endswith('/race_meta.json') and blob.metadata:
                name = blob.metadata.get('race_name')
                if name:
                    meta_names[race_id] = name

        races = [
            {'race_id': rid, 'race_name': meta_names.get(rid)}
            for rid in seen
        ]
        return races
    except Exception as e:
        logger.error("Blob list_races failed: %s", e)
        return []


def upload_file(race_id: str, local_path: str | Path, blob_relative: Optional[str] = None) -> bool:
    """Upload a single file to blob storage under *race_id/*.

    *blob_relative* defaults to the file's name.
    """
    container = _get_container()
    if container is None:
        return False

    local_path = Path(local_path)
    if not local_path.is_file():
        return False

    blob_name = f"{race_id}/{blob_relative or local_path.name}"
    try:
        with open(local_path, 'rb') as f:
            container.upload_blob(blob_name, f, overwrite=True)
        return True
    except Exception as e:
        logger.error("Blob upload_file failed for %s: %s", blob_name, e)
        return False
