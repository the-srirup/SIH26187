"""
Face detection and recognition using InsightFace.

InsightFace bundles SCRFD (detector) + ArcFace (embeddings).
We use it for watchlist matching — a stretch goal that adds
investigative value when combined with the rule alerts.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.face")

# Optional import — InsightFace is a stretch dependency
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    log.warning("InsightFace not installed — face recognition disabled")


@dataclass
class FaceMatch:
    """Result of matching a detected face against the watchlist."""
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    embedding: np.ndarray                # 512-dim ArcFace embedding
    watchlist_id: Optional[int] = None
    watchlist_name: Optional[str] = None
    similarity: float = 0.0
    matched: bool = False


@dataclass
class WatchlistEntry:
    """A person on the watchlist with pre-computed embedding."""
    id: int
    name: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)


class FaceRecognizer:
    """
    SCRFD detector + ArcFace embedding + cosine-similarity watchlist.

    Thread-safe singleton. Initializes the model on first use.
    """

    _instance: Optional["FaceRecognizer"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._app = None
        self._watchlist: dict[int, WatchlistEntry] = {}
        self._next_watchlist_id = 1
        self._similarity_threshold = 0.5  # cosine similarity threshold
        self._enabled = INSIGHTFACE_AVAILABLE

        if self._enabled:
            self._init_model()

    def _init_model(self) -> None:
        """Initialize InsightFace model (buffalo_l = SCRFD + ArcFace)."""
        try:
            # buffalo_l is the standard model pack
            self._app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            log.info("InsightFace model loaded (buffalo_l)")
        except Exception as e:
            log.error("Failed to load InsightFace: %s", e)
            self._enabled = False

    # ------------------------------------------------------------------ #
    # Watchlist management
    # ------------------------------------------------------------------ #

    def add_watchlist_entry(self, name: str, image_path: str, metadata: Optional[dict] = None) -> Optional[int]:
        """
        Add a person to the watchlist from a reference image.

        Returns the watchlist ID or None on failure.
        """
        if not self._enabled:
            log.warning("Face recognition not available")
            return None

        img = cv2.imread(image_path)
        if img is None:
            log.error("Failed to read image: %s", image_path)
            return None

        faces = self._app.get(img)
        if not faces:
            log.error("No face found in: %s", image_path)
            return None

        # Use the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding  # Already L2-normalized

        entry = WatchlistEntry(
            id=self._next_watchlist_id,
            name=name,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._watchlist[self._next_watchlist_id] = entry
        self._next_watchlist_id += 1
        log.info("Added to watchlist: %s (id=%d)", name, entry.id)
        return entry.id

    def add_watchlist_from_embedding(self, name: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> int:
        """Add a watchlist entry from a pre-computed embedding."""
        entry = WatchlistEntry(
            id=self._next_watchlist_id,
            name=name,
            embedding=embedding.astype(np.float32),
            metadata=metadata or {},
        )
        self._watchlist[self._next_watchlist_id] = entry
        self._next_watchlist_id += 1
        return entry.id

    def remove_watchlist_entry(self, watchlist_id: int) -> bool:
        return self._watchlist.pop(watchlist_id, None) is not None

    def clear_watchlist(self) -> None:
        self._watchlist.clear()

    def get_watchlist(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "name": e.name,
                "metadata": e.metadata,
            }
            for e in self._watchlist.values()
        ]

    def set_threshold(self, threshold: float) -> None:
        """Set cosine similarity threshold for matching."""
        self._similarity_threshold = max(0.0, min(1.0, threshold))

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #

    def recognize(self, frame: np.ndarray) -> list[FaceMatch]:
        """
        Detect faces in frame and match against watchlist.

        Returns list of FaceMatch objects (one per detected face).
        """
        if not self._enabled or self._app is None:
            return []

        try:
            faces = self._app.get(frame)
        except Exception as e:
            log.error("Face detection failed: %s", e)
            return []

        matches: list[FaceMatch] = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))  # x1, y1, x2, y2
            embedding = face.normed_embedding

            # Match against watchlist
            best_match_id = None
            best_name = None
            best_sim = -1.0

            for wl_id, entry in self._watchlist.items():
                # Cosine similarity (embeddings are already normalized)
                sim = float(np.dot(embedding, entry.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_match_id = wl_id
                    best_name = entry.name

            matched = best_sim >= self._similarity_threshold
            matches.append(FaceMatch(
                bbox=bbox,
                embedding=embedding,
                watchlist_id=best_match_id if matched else None,
                watchlist_name=best_name if matched else None,
                similarity=best_sim,
                matched=matched,
            ))

        return matches

    def draw_matches(self, frame: np.ndarray, matches: list[FaceMatch]) -> np.ndarray:
        """Draw face boxes and watchlist matches on frame."""
        out = frame.copy()
        for m in matches:
            x1, y1, x2, y2 = m.bbox
            if m.matched:
                color = (0, 0, 255)  # Red for watchlist match
                label = f"WATCHLIST: {m.watchlist_name} ({m.similarity:.2f})"
            else:
                color = (0, 255, 0)  # Green for unknown
                label = f"Face ({m.similarity:.2f})"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


# Convenience singleton getter
def get_face_recognizer() -> FaceRecognizer:
    return FaceRecognizer()