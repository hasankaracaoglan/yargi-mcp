import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "disk_document_cache.json")

class DocumentCache:
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(f"Loaded {len(self.data)} documents from cache.")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.data = {}

    def save(self):
        try:
            temp_file = self.cache_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, self.cache_file)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def get(self, document_id: str) -> Optional[str]:
        return self.data.get(document_id)

    def set(self, document_id: str, content: str):
        self.data[document_id] = content
        self.save()

# Global cache instance
doc_cache = DocumentCache()
