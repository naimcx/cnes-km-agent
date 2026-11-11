import os
import logging
from pathlib import Path
from typing import Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

logger = logging.getLogger("QdrantService")

def get_qdrant_client() -> QdrantClient:
    """Returns an instance of the Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant_schema(client: QdrantClient):
    """Creates an index on the file_path field for efficient filtering and deletion without a full scan."""
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="file_path",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

def get_indexed_files_map(client: QdrantClient) -> Dict[str, str]:
    """Returns a mapping of file paths to their corresponding SHA-256 hashes from the Qdrant collection."""
    files_map = {}
    offset = None
    try:
        while True:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["file_path", "file_hash"],
                with_vectors=False,
            )
            for point in records:
                path = point.payload.get("file_path")
                f_hash = point.payload.get("file_hash")
                if path and f_hash:
                    files_map[path] = f_hash
            if offset is None:
                break
    except Exception as e:
        logger.debug(f"Collection scroll error: {e}")
    return files_map

def delete_file_points(client: QdrantClient, file_path_str: str):
    """Deletes all points in the Qdrant collection associated with a specific file path."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_path",
                            match=rest.MatchValue(value=file_path_str),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Delete all old vectors : {Path(file_path_str).name}")
    except Exception as e:
        logger.warning(f"Error delete for {file_path_str} : {e}")
