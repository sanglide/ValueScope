"""Vector search skill for semantic code retrieval."""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Optional

from valueguard.skills.base_skill import BaseSkill


class VectorSearchSkill(BaseSkill):
    """Skill for building and searching code vector indices.

    Uses embeddings for semantic similarity search across code.
    Supports FAISS for efficient nearest neighbor search.
    """

    name = "vector_search"
    description = "Build and search code vector index using embeddings"
    version = "1.0.0"

    # Supported file extensions for indexing
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".cpp",
        ".c",
        ".h",
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self._embedding_model = config.get(
            "embedding_model", "BAAI/bge-small-en-v1.5"
        ) if config else "BAAI/bge-small-en-v1.5"
        self._chunk_size = config.get("chunk_size", 512) if config else 512
        self._overlap = config.get("overlap", 64) if config else 64
        self._cache_dir = config.get("cache_dir", ".valueguard/index") if config else ".valueguard/index"

        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self._model = None

    def validate_args(self, **kwargs: Any) -> None:
        """Validate arguments."""
        action = kwargs.get("action")
        if action not in ("index", "search", "clear"):
            raise ValueError(f"Invalid action: {action}. Use 'index', 'search', or 'clear'")

        if action == "index" and not kwargs.get("repo_path"):
            raise ValueError("repo_path is required for indexing")

        if action == "search" and not kwargs.get("query"):
            raise ValueError("query is required for searching")

    def execute(
        self,
        action: str,
        repo_path: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
    ) -> Any:
        """Execute vector search operations.

        Args:
            action: "index" to build index, "search" to query, "clear" to remove index
            repo_path: Path to repository (required for indexing)
            query: Search query (required for searching)
            top_k: Number of results to return (for search)

        Returns:
            For index: dict with indexing stats
            For search: list of search results
            For clear: bool indicating success
        """
        if action == "index":
            return self._build_index(repo_path)
        elif action == "search":
            return self._search(query, top_k)
        elif action == "clear":
            return self._clear_index(repo_path)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _build_index(self, repo_path: str) -> dict[str, Any]:
        """Build vector index for a repository."""
        repo_path = Path(repo_path).resolve()
        cache_path = self._get_cache_path(repo_path)

        # Check if cached index exists and is fresh
        if self._load_cached_index(cache_path, repo_path):
            return {
                "status": "cached",
                "chunks": len(self._chunks),
                "cache_path": str(cache_path),
            }

        # Load embedding model
        self._load_model()

        # Collect and chunk code files
        self._chunks = self._chunk_codebase(repo_path)

        if not self._chunks:
            return {"status": "empty", "chunks": 0}

        # Build FAISS index
        self._build_faiss_index()

        # Cache the index
        self._save_index(cache_path)

        return {
            "status": "built",
            "chunks": len(self._chunks),
            "cache_path": str(cache_path),
        }

    def _search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search the index for relevant code."""
        if self._index is None or not self._chunks:
            return []

        # Load model if needed
        self._load_model()

        # Embed query
        try:
            query_embedding = self._model.encode([query], normalize_embeddings=True)
        except Exception:
            return []

        # Search index
        try:
            import numpy as np

            distances, indices = self._index.search(
                np.array(query_embedding).astype("float32"), min(top_k, len(self._chunks))
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue

                chunk = self._chunks[idx]
                score = 1.0 - float(distances[0][i])  # Convert distance to similarity

                results.append({
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "content": chunk["content"],
                    "score": max(0.0, min(1.0, score)),
                })

            return results

        except Exception:
            return []

    def _clear_index(self, repo_path: Optional[str]) -> bool:
        """Clear the index cache."""
        if repo_path:
            cache_path = self._get_cache_path(Path(repo_path))
            if cache_path.exists():
                os.remove(cache_path)

        self._index = None
        self._chunks = []
        return True

    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embedding_model)
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )

    def _chunk_codebase(self, repo_path: Path) -> list[dict[str, Any]]:
        """Chunk all code files in the repository."""
        chunks = []

        for root, _, files in os.walk(repo_path):
            # Skip hidden directories and common non-code directories
            rel_root = Path(root).relative_to(repo_path)
            if any(
                part.startswith(".") or part in ("node_modules", "venv", "__pycache__", "dist", "build")
                for part in rel_root.parts
            ):
                continue

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix not in self.CODE_EXTENSIONS:
                    continue

                try:
                    file_chunks = self._chunk_file(file_path, repo_path)
                    chunks.extend(file_chunks)
                except Exception:
                    continue

        return chunks

    def _chunk_file(
        self, file_path: Path, repo_path: Path
    ) -> list[dict[str, Any]]:
        """Chunk a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            return []

        if len(content) < 50:
            return []

        rel_path = str(file_path.relative_to(repo_path))
        lines = content.split("\n")
        chunks = []

        # Simple line-based chunking
        current_chunk = []
        current_start = 1

        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            chunk_size = sum(len(l) for l in current_chunk)

            if chunk_size >= self._chunk_size:
                chunks.append({
                    "file_path": rel_path,
                    "start_line": current_start,
                    "end_line": i,
                    "content": "\n".join(current_chunk),
                })

                # Overlap: keep last few lines
                overlap_lines = max(1, len(current_chunk) // 4)
                current_chunk = current_chunk[-overlap_lines:]
                current_start = i - overlap_lines + 1

        # Add remaining content
        if current_chunk:
            chunks.append({
                "file_path": rel_path,
                "start_line": current_start,
                "end_line": len(lines),
                "content": "\n".join(current_chunk),
            })

        return chunks

    def _build_faiss_index(self) -> None:
        """Build FAISS index from chunks."""
        if not self._chunks:
            return

        try:
            import faiss
            import numpy as np

            # Get embeddings for all chunks
            texts = [c["content"] for c in self._chunks]
            embeddings = self._model.encode(texts, normalize_embeddings=True)

            # Build index
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            self._index.add(np.array(embeddings).astype("float32"))

        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu")

    def _get_cache_path(self, repo_path: Path) -> Path:
        """Get cache file path for a repository."""
        # Create hash of repo path
        repo_hash = hashlib.md5(str(repo_path).encode()).hexdigest()[:12]
        cache_dir = Path(self._cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"index_{repo_hash}.pkl"

    def _load_cached_index(self, cache_path: Path, repo_path: Path) -> bool:
        """Load index from cache if valid."""
        if not cache_path.exists():
            return False

        try:
            # Check if cache is stale (compare mtime)
            cache_mtime = cache_path.stat().st_mtime
            repo_mtime = max(
                f.stat().st_mtime
                for f in repo_path.rglob("*")
                if f.is_file() and f.suffix in self.CODE_EXTENSIONS
            )

            if repo_mtime > cache_mtime:
                return False

            # Load cached data
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            self._chunks = data["chunks"]
            self._index = data["index"]
            return True

        except Exception:
            return False

    def _save_index(self, cache_path: Path) -> None:
        """Save index to cache."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"chunks": self._chunks, "index": self._index},
                    f,
                )
        except Exception:
            pass
