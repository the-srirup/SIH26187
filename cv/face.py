"""
Face detection and watchlist recognition for IBVAP.

This module uses InsightFace because it ships SCRFD (fast face detection)
and ArcFace (embedding model) in one package. The implementation is tuned
for edge CCTV workloads rather than a demoware "detect every face in every
frame" path:

* Face embeddings are cached by YOLO track-id, so we do not re-embed the same
  person on every frame.
* Recognition runs on a sampling cadence (``FACE_RECOGNITION_EVERY_N_FRAMES``)
  and can optionally crop from person detections.
* The watchlist is persisted in SQLite, so camera restarts do not erase
  enrolled faces.
* A matched face can fire a ``watchlist_match`` alert into the same hash-chain
  tamper-evident log used by the rest of the system.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.face")

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
    track_id: Optional[int] = None       # YOLO track-id if linked to a person
    watchlist_id: Optional[int] = None
    watchlist_name: Optional[str] = None
    similarity: float = 0.0
    matched: bool = False
    det_score: float = 0.0


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

    Thread-safe singleton. Initializes the model on first use and persists
    watchlist entries to SQLite. The in-memory dictionary is kept in sync with
    the database so the request path stays fast.
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
        self._similarity_threshold = settings.FACE_SIMILARITY_THRESHOLD
        self._enabled = INSIGHTFACE_AVAILABLE
        self._frame_count = 0
        self._last_processed_frame = -1
        self._inference_lock = threading.Lock()

        # track_id -> (watchlist_id, watchlist_name, similarity, matched,
        #              last_seen_monotonic)
        # This cache stops expensive ArcFace embedding on every frame for the
        # same person. We only refresh after a configurable TTL.
        self._track_match_cache: dict[
            int, tuple[Optional[int], Optional[str], float, bool, float]
        ] = {}

        if self._enabled:
            self._init_model()

        self._load_watchlist_from_db()

    # ------------------------------------------------------------------ #
    # Model / DB management
    # ------------------------------------------------------------------ #

    def _init_model(self) -> None:
        """Initialize InsightFace model (buffalo_l = SCRFD + ArcFace)."""
        try:
            self._app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            log.info("InsightFace model loaded (SCRFD + ArcFace)")
        except Exception as e:
            log.error("Failed to load InsightFace: %s", e)
            self._enabled = False

    def _load_watchlist_from_db(self) -> None:
        """Load persisted watchlist entries from SQLite into memory."""
        try:
            from core.database import SessionLocal, init_db
            from core.models import WatchlistEntry as DBWatchlistEntry

            init_db()
            db = SessionLocal()
            try:
                rows = db.query(DBWatchlistEntry).order_by(DBWatchlistEntry.id.asc()).all()
                seen = 0
                for row in rows:
                    try:
                        embedding = np.asarray(
                            json.loads(row.embedding_json), dtype=np.float32
                        )
                        if embedding.size != 512:
                            log.warning(
                                "Skipping watchlist entry %d: unexpected embedding size %d",
                                row.id,
                                embedding.size,
                            )
                            continue
                        norm = float(np.linalg.norm(embedding) or 1.0)
                        embedding = embedding / norm

                        metadata = {}
                        if row.metadata_json:
                            try:
                                metadata = json.loads(row.metadata_json)
                            except json.JSONDecodeError:
                                metadata = {}

                        self._watchlist[row.id] = WatchlistEntry(
                            id=row.id,
                            name=row.name,
                            embedding=embedding,
                            metadata=metadata,
                        )
                        seen += 1
                    except Exception as e:
                        log.warning("Failed to restore watchlist row %s: %s", row.id, e)

                if seen:
                    self._next_watchlist_id = max(
                        max(self._watchlist.keys(), default=0) + 1,
                        self._next_watchlist_id,
                    )
            finally:
                db.close()
        except Exception as e:
            log.warning("Could not load persisted watchlist entries: %s", e)

    # ------------------------------------------------------------------ #
    # Watchlist management
    # ------------------------------------------------------------------ #

    def add_watchlist_entry(self, name: str, image_path: str, metadata: Optional[dict] = None) -> Optional[int]:
        """
        Add a person to the watchlist from a reference image.

        Returns the watchlist ID or None on failure.
        """
        if not self._enabled or self._app is None:
            log.warning("Face recognition not available")
            return None

        img = cv2.imread(image_path)
        if img is None:
            log.error("Failed to read image: %s", image_path)
            return None

        faces = self._safe_face_get(img)
        if not faces:
            log.error("No face found in: %s", image_path)
            return None

        face = self._select_best_face(faces)
        embedding = self._normalize_embedding(face.normed_embedding)

        return self._add_watchlist_embedding(name, embedding, metadata or {})

    def add_watchlist_from_embedding(self, name: str, embedding: np.ndarray, metadata: Optional[dict] = None) -> int:
        """Add a watchlist entry from a pre-computed embedding."""
        return self._add_watchlist_embedding(name, embedding, metadata or {})

    def _add_watchlist_embedding(self, name: str, embedding: np.ndarray, metadata: dict) -> int:
        embedding = self._normalize_embedding(embedding)
        if embedding.size != 512:
            raise ValueError(f"Expected 512-dimensional face embedding, got {embedding.size}")

        watchlist_id = self._next_watchlist_id
        entry = WatchlistEntry(
            id=watchlist_id,
            name=name,
            embedding=embedding,
            metadata=metadata,
        )
        self._watchlist[watchlist_id] = entry
        self._next_watchlist_id += 1

        self._persist_watchlist_entry(entry)
        log.info("Added to watchlist: %s (id=%d)", name, watchlist_id)
        return watchlist_id

    def _persist_watchlist_entry(self, entry: WatchlistEntry) -> None:
        """Write one watchlist entry into the SQLite store."""
        try:
            from core.database import SessionLocal, init_db
            from core.models import WatchlistEntry as DBWatchlistEntry

            init_db()
            db = SessionLocal()
            try:
                row = (
                    db.query(DBWatchlistEntry)
                    .filter(DBWatchlistEntry.id == entry.id)
                    .first()
                )
                if row is None:
                    row = DBWatchlistEntry(id=entry.id)
                    db.add(row)
                row.name = entry.name
                row.embedding_json = json.dumps(entry.embedding.tolist())
                row.metadata_json = json.dumps(entry.metadata, default=str)
                row.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.error("Failed to persist watchlist entry %s: %s", entry.id, e)

    def remove_watchlist_entry(self, watchlist_id: int) -> bool:
        removed = self._watchlist.pop(watchlist_id, None) is not None
        try:
            from core.database import SessionLocal
            from core.models import WatchlistEntry as DBWatchlistEntry

            db = SessionLocal()
            try:
                row = (
                    db.query(DBWatchlistEntry)
                    .filter(DBWatchlistEntry.id == watchlist_id)
                    .first()
                )
                if row:
                    db.delete(row)
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            log.error("Failed to remove persisted watchlist entry %s: %s", watchlist_id, e)
        return removed

    def clear_watchlist(self) -> None:
        self._watchlist.clear()
        self._track_match_cache.clear()
        try:
            from core.database import SessionLocal
            from core.models import WatchlistEntry as DBWatchlistEntry

            db = SessionLocal()
            try:
                db.query(DBWatchlistEntry).delete()
                db.commit()
            finally:
                db.close()
        except Exception as e:
            log.error("Failed to clear persisted watchlist: %s", e)

    def get_watchlist(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "name": e.name,
                "metadata": e.metadata,
            }
            for e in self._watchlist.values()
        ]

    def get_watchlist_count(self) -> int:
        return len(self._watchlist)

    def set_threshold(self, threshold: float) -> None:
        """Set cosine similarity threshold for matching."""
        self._similarity_threshold = max(0.0, min(1.0, threshold))

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #

    def recognize(
        self,
        frame: np.ndarray,
        detections: Optional[list] = None,
        frame_number: Optional[int] = None,
        force: bool = False,
    ) -> list[FaceMatch]:
        """
        Detect faces in a frame and match against the watchlist.

        When ``detections`` is provided we use YOLO person boxes to associate
        face embeddings with track-ids and cache similarly.
        """
        self._frame_count = frame_number if frame_number is not None else self._frame_count + 1

        if not self._enabled or self._app is None:
            return []

        frame_height = frame.shape[0]
        min_height = max(20, settings.FACE_MIN_HEIGHT)

        # Sampling cadence: edge CPUs cannot run SCRFD + ArcFace at 30 FPS.
        # We still refresh every N frames. For immediate tests/API calls force
        # can be used to bypass the cadence.
        cadence_ok = (
            self._frame_count - self._last_processed_frame
            >= settings.FACE_RECOGNITION_EVERY_N_FRAMES
        )
        if not force and not cadence_ok:
            return self._cached_matches_for_frame(detections)

        self._last_processed_frame = self._frame_count

        # Build track_id -> person bbox map. We prefer to crop/search around
        # people only when person detections are available. If none are
        # available (for example direct face-test endpoint), process the full
        # frame once.
        person_map = {}
        if detections:
            for det in detections:
                if getattr(det, "class_name", "").lower() == "person":
                    person_map[det.track_id] = det.bbox

        with self._inference_lock:
            try:
                if person_map:
                    faces_and_tracks = self._detect_in_person_boxes(frame, person_map, min_height)
                else:
                    faces_and_tracks = self._detect_full_frame(frame, min_height)
            except Exception as e:
                log.error("Face detection failed: %s", e)
                return []

        matches: list[FaceMatch] = []
        now = time.monotonic()
        for face, track_id, bbox in faces_and_tracks:
            embedding = self._normalize_embedding(face.normed_embedding)
            best_match_id, best_name, best_sim = self._best_match(embedding)
            matched = best_sim >= self._similarity_threshold

            match = FaceMatch(
                bbox=bbox,
                embedding=embedding,
                track_id=track_id,
                watchlist_id=best_match_id if matched else None,
                watchlist_name=best_name if matched else None,
                similarity=best_sim,
                matched=matched,
                det_score=float(getattr(face, "det_score", 0.0)),
            )
            matches.append(match)

            if track_id is not None:
                self._track_match_cache[track_id] = (
                    match.watchlist_id,
                    match.watchlist_name,
                    match.similarity,
                    match.matched,
                    now,
                )

        return matches

    def _cached_matches_for_frame(self, detections: Optional[list]) -> list[FaceMatch]:
        """
        Rebuild FaceMatch results from cached track-id data between actual
        inference frames. This keeps streaming pipelines lightweight.
        """
        now = time.monotonic()
        ttl = settings.FACE_MATCH_CACHE_SECONDS
        matches: list[FaceMatch] = []
        if not detections:
            return matches

        for det in detections:
            if getattr(det, "class_name", "").lower() != "person":
                continue
            cached = self._track_match_cache.get(det.track_id)
            if not cached:
                continue
            watchlist_id, name, sim, matched, last_seen = cached
            if now - last_seen > ttl:
                self._track_match_cache.pop(det.track_id, None)
                continue
            bbox = tuple(map(int, det.bbox))
            matches.append(
                FaceMatch(
                    bbox=bbox,
                    embedding=np.zeros(512, dtype=np.float32),
                    track_id=det.track_id,
                    watchlist_id=watchlist_id if matched else None,
                    watchlist_name=name if matched else None,
                    similarity=sim,
                    matched=matched,
                    det_score=0.0,
                )
            )
        return matches

    def _detect_full_frame(self, frame: np.ndarray, min_height: int) -> list[tuple]:
        """
        Run SCRFD on the full frame, returning ``(face, track_id, bbox)``.
        No YOLO track association is available, so ``track_id`` is None.
        """
        faces = self._safe_face_get(frame)
        return [
            (face, None, self._bbox_from_face(face))
            for face in faces
            if self._face_height(face) >= min_height
        ]

    def _detect_in_person_boxes(
        self,
        frame: np.ndarray,
        person_map: dict[int, tuple[int, int, int, int]],
        min_height: int,
    ) -> list[tuple]:
        """Run face detection inside each person bbox, with full-frame fallback."""
        results: list[tuple] = []
        seen_regions: list[tuple[int, int, int, int]] = []

        for track_id, bbox in person_map.items():
            x1, y1, x2, y2 = map(int, bbox)
            # Expand slightly so a hat/forehead is not excluded.
            h, w = frame.shape[:2]
            crop_x1 = max(0, x1 - int((x2 - x1) * 0.15))
            crop_y1 = max(0, y1 - int((y2 - y1) * 0.35))
            crop_x2 = min(w, x2 + int((x2 - x1) * 0.15))
            crop_y2 = min(h, y2 + int((y2 - y1) * 0.10))
            if crop_x2 - crop_x1 < min_height or crop_y2 - crop_y1 < min_height:
                continue

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            faces = self._safe_face_get(crop)
            for face in faces:
                if self._face_height(face) < min_height:
                    continue

                # Convert crop-local bbox back to full-frame coordinates.
                face_bbox = self._bbox_from_face(face, offset=(crop_x1, crop_y1))
                results.append((face, track_id, face_bbox))
                seen_regions.append(face_bbox)

        # If YOLO person detections missed any face (e.g. partial occlusion,
        # near-field person), fallback to full-frame once and associate by
        # geometry. This preserves recall without running full-frame every
        # cadence tick.
        if not results:
            full_results = self._detect_full_frame(frame, min_height)
            for face, _, face_bbox in full_results:
                best_track_id = self._associate_face_bbox(face_bbox, person_map)
                results.append((face, best_track_id, face_bbox))
                seen_regions.append(face_bbox)

        return results

    def _associate_face_bbox(
        self,
        face_bbox: tuple[int, int, int, int],
        person_map: dict[int, tuple[int, int, int, int]],
    ) -> Optional[int]:
        """Associate a face bbox with the nearest YOLO person track-id."""
        fx1, fy1, fx2, fy2 = face_bbox
        face_center = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)
        best_track_id = None
        best_distance = float("inf")

        for track_id, (px1, py1, px2, py2) in person_map.items():
            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0
            distance_sq = (face_center[0] - cx) ** 2 + (face_center[1] - cy) ** 2
            if distance_sq < best_distance:
                best_distance = distance_sq
                best_track_id = track_id

        # Do not associate absurdly far faces to a person.
        if best_track_id is not None and best_distance > (settings.FRAME_WIDTH / 3) ** 2:
            return None
        return best_track_id

    def _best_match(self, embedding: np.ndarray) -> tuple[Optional[int], Optional[str], float]:
        best_match_id: Optional[int] = None
        best_name: Optional[str] = None
        best_sim = -1.0

        for wl_id, entry in self._watchlist.items():
            sim = float(np.dot(embedding, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best_match_id = wl_id
                best_name = entry.name

        return best_match_id, best_name, best_sim

    # ------------------------------------------------------------------ #
    # Visualization / utilities
    # ------------------------------------------------------------------ #

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
            cv2.putText(out, label, (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        arr = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr) or 1.0)
        return arr / norm

    def _safe_face_get(self, img: np.ndarray):
        return self._app.get(img)

    @staticmethod
    def _select_best_face(faces):
        """Choose the largest, highest-scoring face for watchlist enrollment."""
        if len(faces) == 1:
            return faces[0]

        def area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        # Prefer large, high-det-score faces. Enrollment is most useful when
        # the reference image is not an accidental tiny / blurry face.
        return max(
            faces,
            key=lambda f: (
                area(f) > 0,
                area(f),
                float(getattr(f, "det_score", 0.0)),
            ),
        )

    @staticmethod
    def _bbox_from_face(
        face,
        offset: tuple[int, int] = (0, 0),
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = face.bbox
        return (
            int(x1) + offset[0],
            int(y1) + offset[1],
            int(x2) + offset[0],
            int(y2) + offset[1],
        )

    @staticmethod
    def _face_height(face) -> float:
        x1, y1, x2, y2 = face.bbox
        return float(y2 - y1)

    def clear_track_cache(self) -> None:
        self._track_match_cache.clear()

    def get_metrics(self) -> dict:
        return {
            "enabled": self._enabled,
            "watchlist_count": len(self._watchlist),
            "threshold": self._similarity_threshold,
            "cached_tracks": len(self._track_match_cache),
            "last_processed_frame": self._last_processed_frame,
        }


# Convenience singleton getter
def get_face_recognizer() -> FaceRecognizer:
    return FaceRecognizer()
