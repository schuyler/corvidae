"""Local workspace file indexer with semantic search.

Uses sentence-transformers for embeddings and ChromaDB for vector storage.
Indexes text files in the workspace directory with token-aware chunking.

Designed to complement — not replace — otterwiki's MCP semantic search,
which handles wiki pages. This covers workspace-text only: MEMORY.md,
blog posts, notes, config files, etc.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton embedding model (lazy-loaded)
_model: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first use."""
    global _model
    if _model is None:
        logger.info("loading embedding model %s (~80 MB)", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# Token-aware chunking: ~500 tokens per chunk with overlap
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100
ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Count tokens in text using cl100k_base encoding (GPT-4/3.5 compatible)."""
    return len(ENCODER.encode(text))


def _chunk_text(text: str, chunk_size: int = _DEFAULT_CHUNK_SIZE,
                overlap: int = _DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    tokens = ENCODER.encode(text)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENCODER.decode(chunk_tokens))
        start += chunk_size - overlap
        if start >= len(tokens):
            break

    return chunks


def _find_text_files(root: Path, exclude: list[str] | None = None) -> list[Path]:
    """Recursively find text files in the workspace directory."""
    excluded_extensions = {".pyc", ".pyo", "__pycache__", ".git", "node_modules",
                           ".venv", "venv", "egg-info", "*.lock"}
    if exclude:
        excluded_extensions.update(exclude)

    text_files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        suffix = path.suffix.lower()

        # Binary extensions to skip
        binary_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
                       ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
                       ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
                       ".exe", ".dll", ".so", ".dylib", ".bin", ".o",
                       ".sqlite", ".db", ".pickle", ".pkl"}

        if suffix in binary_exts or name == "__pycache__":
            continue

        # Skip files in excluded paths
        parts = [str(p) for p in path.parts]
        if any(excl in parts for excl in excluded_extensions):
            continue

        # Try to read as text; skip if it fails
        try:
            path.read_text(encoding="utf-8", errors="strict")
            text_files.append(path)
        except (UnicodeDecodeError, OSError):
            continue

    return text_files


class WorkspaceIndexer:
    """Indexes workspace text files and provides semantic search."""

    def __init__(self, workspace_root: str | Path, collection_name: str = "workspace_index"):
        self.workspace_root = Path(workspace_root)
        self.collection_name = collection_name
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # ChromaDB management
    # ------------------------------------------------------------------

    def _get_client(self) -> chromadb.PersistentClient:
        """Get or create the persistent ChromaDB client."""
        if self._client is None:
            db_path = self.workspace_root / ".corvidae_index"
            db_path.mkdir(exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        """Get or create the workspace index collection."""
        if self._collection is None:
            client = self._get_client()
            # Delete and recreate to force full rebuild (simple approach)
            try:
                client.delete_collection(self.collection_name)
            except chromadb.errors.NotFoundError:
                pass
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build or rebuild the index from scratch.

        Returns stats about the indexing operation.
        """
        model = _get_model()
        files = _find_text_files(self.workspace_root)
        total_tokens = 0
        total_chunks = 0
        indexed_files: list[str] = []

        for filepath in files:
            try:
                text = filepath.read_text(encoding="utf-8", errors="strict")
                # Skip very small files (under 20 tokens)
                if _token_count(text) < 20:
                    continue

                chunks = _chunk_text(text)
                rel_path = str(filepath.relative_to(self.workspace_root))

                for i, chunk in enumerate(chunks):
                    token_count = _token_count(chunk)
                    total_tokens += token_count
                    total_chunks += 1
                    indexed_files.append(rel_path)

                    self._get_collection().add(
                        documents=[chunk],
                        metadatas=[{
                            "source": rel_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "token_count": token_count,
                            "file_size": filepath.stat().st_size,
                        }],
                        ids=[f"{rel_path}::{i}"],
                        embeddings=model.encode(chunk).tolist(),
                    )

            except Exception as exc:
                logger.warning("failed to index %s: %s", filepath, exc)

        return {
            "files_indexed": len(files),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "indexed_files": sorted(set(indexed_files)),
        }

    def update(self) -> dict[str, Any]:
        """Incremental update — rebuild only changed files.

        Returns stats about the update operation.
        """
        # For now, just do a full rebuild (simple and correct)
        logger.info("incremental update: performing full rebuild")
        return self.build()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Search indexed files for semantic similarity to query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return (default 8).

        Returns:
            List of result dicts with keys: source, chunk_index, snippet,
            token_count, file_size, score.
        """
        model = _get_model()
        collection = self._get_collection()

        # Check if collection is empty
        count = collection.count()
        if count == 0:
            return []

        embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # Convert cosine distance to similarity score (1 - distance)
            score = round(1.0 - distance, 4)

            output.append({
                "source": metadata.get("source", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "snippet": results["documents"][0][i],
                "token_count": metadata.get("token_count", 0),
                "file_size": metadata.get("file_size", 0),
                "score": score,
            })

        return output


class WorkspaceIndexerPlugin:
    """Plugin that provides the workspace_search tool."""

    depends_on = set()

    def __init__(self, pm) -> None:
        self.pm = pm
        self._indexer: WorkspaceIndexer | None = None
        self._workspace_root: Path = Path.cwd()

    def _get_indexer(self) -> WorkspaceIndexer:
        """Get or create the workspace indexer instance."""
        if self._indexer is None:
            self._indexer = WorkspaceIndexer(self._workspace_root)
        return self._indexer

    async def build_index(self, root: str | None = None) -> str:
        """Build or rebuild the workspace index.

        Args:
            root: Optional workspace directory path. Defaults to CWD.

        Returns:
            Status string with indexing statistics.
        """
        if root is not None:
            self._workspace_root = Path(root).resolve()

        stats = self._get_indexer().build()
        files_list = ", ".join(stats["indexed_files"][:5])
        if len(stats["indexed_files"]) > 5:
            files_list += f" (+{len(stats['indexed_files']) - 5} more)"
        return (f"Indexed {stats['files_indexed']} files, "
                f"{stats['total_chunks']} chunks, {stats['total_tokens']} tokens. "
                f"Files: {files_list}")

    @property
    def workspace_search(self) -> callable:
        """Return the workspace_search tool function."""
        plugin = self  # capture for closure

        async def _search(query: str, max_results: int | None = 8) -> str:
            """Search workspace files via semantic similarity.

            Returns formatted results with source file, chunk snippet, and relevance score.
            """
            results = plugin._get_indexer().search(query, top_k=max_results or 8)

            if not results:
                return "No results found in workspace index."

            lines = [f"Workspace search for '{query}':\n"]
            for r in results:
                score_str = f"{r['score']:.3f}"
                source_short = r["source"] if len(r["source"]) < 60 else "..." + r["source"][-57:]
                lines.append(
                    f"- [{score_str}] {r['source']} (chunk {r['chunk_index']})\n"
                    f"  {r['snippet']}\n"
                )

            return "\n".join(lines)

        return _search

    def register_tools(self, tool_registry: list) -> None:
        """Register workspace_search and build_index tools."""
        plugin = self
        ws_search = plugin.workspace_search
        ws_search.__name__ = "workspace_search"
        from corvidae.tool import Tool
        tool_registry.extend([
            Tool.from_function(ws_search),
            Tool.from_function(plugin.build_index),
        ])
