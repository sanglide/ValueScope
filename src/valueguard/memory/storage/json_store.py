"""JSON file storage backend for ValueGuard memory."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


def datetime_decoder(dct: dict) -> dict:
    """Decode datetime strings in JSON objects."""
    for key, value in dct.items():
        if isinstance(value, str):
            # Try to parse ISO format datetime
            try:
                if "T" in value and len(value) >= 19:
                    dct[key] = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
    return dct


class BaseStorage(ABC, Generic[T]):
    """Abstract base class for storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get item by key."""
        pass

    @abstractmethod
    def store(self, key: str, value: T) -> None:
        """Store item with key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item by key. Returns True if deleted."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        pass

    @abstractmethod
    def query(self, **filters: Any) -> list[T]:
        """Query items with filters."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored items."""
        pass


class JsonStorage(BaseStorage[dict[str, Any]]):
    """JSON file-based storage backend.

    Stores each collection in a separate JSON file.
    Suitable for small to medium datasets.
    """

    def __init__(self, base_path: str):
        """Initialize JSON storage.

        Args:
            base_path: Base directory for storage files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    def _get_file_path(self, collection: str) -> Path:
        """Get file path for a collection."""
        return self.base_path / f"{collection}.json"

    def _load_collection(self, collection: str) -> dict[str, Any]:
        """Load a collection from disk."""
        if collection in self._cache:
            return self._cache[collection]

        file_path = self._get_file_path(collection)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f, object_hook=datetime_decoder)
                self._cache[collection] = data
                return data
        return {}

    def _save_collection(self, collection: str, data: dict[str, Any]) -> None:
        """Save a collection to disk."""
        file_path = self._get_file_path(collection)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2, ensure_ascii=False)
        self._cache[collection] = data

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get item by key.

        Key format: collection/item_id
        """
        parts = key.split("/", 1)
        if len(parts) != 2:
            return None

        collection, item_id = parts
        data = self._load_collection(collection)
        return data.get(item_id)

    def store(self, key: str, value: dict[str, Any]) -> None:
        """Store item with key.

        Key format: collection/item_id
        """
        parts = key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid key format: {key}. Expected 'collection/item_id'")

        collection, item_id = parts
        data = self._load_collection(collection)
        data[item_id] = value
        self._save_collection(collection, data)

    def delete(self, key: str) -> bool:
        """Delete item by key."""
        parts = key.split("/", 1)
        if len(parts) != 2:
            return False

        collection, item_id = parts
        data = self._load_collection(collection)
        if item_id in data:
            del data[item_id]
            self._save_collection(collection, data)
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys with optional prefix filter."""
        keys = []

        # If prefix contains a slash, search within that collection
        if "/" in prefix:
            collection, item_prefix = prefix.split("/", 1)
            data = self._load_collection(collection)
            for item_id in data.keys():
                if item_id.startswith(item_prefix):
                    keys.append(f"{collection}/{item_id}")
        else:
            # List all collections matching prefix
            for file_path in self.base_path.glob("*.json"):
                collection = file_path.stem
                if collection.startswith(prefix):
                    data = self._load_collection(collection)
                    for item_id in data.keys():
                        keys.append(f"{collection}/{item_id}")

        return keys

    def query(self, collection: str = "", **filters: Any) -> list[dict[str, Any]]:
        """Query items with filters.

        Args:
            collection: Collection to query
            **filters: Field filters (supports __contains, __gte, __lte suffixes)

        Returns:
            List of matching items
        """
        if not collection:
            # Query all collections
            results = []
            for file_path in self.base_path.glob("*.json"):
                coll = file_path.stem
                results.extend(self.query(collection=coll, **filters))
            return results

        data = self._load_collection(collection)
        results = []

        for item in data.values():
            if self._matches_filters(item, filters):
                results.append(item)

        return results

    def _matches_filters(self, item: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if item matches all filters."""
        for key, value in filters.items():
            # Handle special filter suffixes
            if key.endswith("__contains"):
                field = key[:-10]
                if field not in item:
                    return False
                if value not in item[field]:
                    return False
            elif key.endswith("__gte"):
                field = key[:-5]
                if field not in item:
                    return False
                if item[field] < value:
                    return False
            elif key.endswith("__lte"):
                field = key[:-5]
                if field not in item:
                    return False
                if item[field] > value:
                    return False
            else:
                # Exact match
                if key not in item:
                    return False
                if item[key] != value:
                    return False

        return True

    def clear(self) -> None:
        """Clear all stored items."""
        self._cache.clear()
        for file_path in self.base_path.glob("*.json"):
            os.remove(file_path)

    def clear_collection(self, collection: str) -> None:
        """Clear a specific collection."""
        if collection in self._cache:
            del self._cache[collection]
        file_path = self._get_file_path(collection)
        if file_path.exists():
            os.remove(file_path)

    def get_all(self, collection: str) -> dict[str, Any]:
        """Get all items in a collection."""
        return self._load_collection(collection)

    def count(self, collection: str) -> int:
        """Count items in a collection."""
        return len(self._load_collection(collection))
