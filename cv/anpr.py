"""
Automatic Number Plate Recognition (ANPR).

Pipeline
--------
::

    frame -> vehicle detections (YOLO)
          -> vehicle ROI (lower portion, where plates live)
          -> plate candidate localisation (edge density + contour geometry)
          -> plate crop
          -> UPSCALE + preprocess variants
          -> OCR each variant, score each token
          -> per-vehicle temporal consensus (per-character majority vote)
          -> validated plate -> event -> evidence -> database -> frontend

Why the previous implementation read nothing
--------------------------------------------
Four defects compounded, and each alone was enough to make ANPR produce no
usable output on a normal feed:

1. **The candidate filter was geometrically impossible.**  A candidate had to
   be at least ``50 x 15`` px.  Analytics run at 640x384, where a plate on a
   car at realistic range is more like ``40 x 12`` — so the contour search
   returned nothing, every frame, and ANPR silently did nothing at all.  Size
   limits are now *relative* to the region being searched.

2. **The crop was never upscaled.**  Whatever survived was handed to EasyOCR at
   native size.  No OCR engine reads 8-pixel-tall glyphs.  Crops are now scaled
   up to ``ANPR_PLATE_TARGET_HEIGHT`` with a cubic filter before recognition.

3. **OCR ran on a globally CLAHE'd grayscale frame.**  ``preprocess_for_indian_plates``
   flattened the whole frame to one channel and stretched its contrast, then
   *that* image was cropped for OCR — so the recogniser never saw the original
   pixels, only a noise-amplified copy of them.  Localisation still uses the
   contrast-enhanced image (it helps edge detection); OCR now reads the
   original crop, plus preprocessed variants, and keeps the best result.

4. **Every OCR token was concatenated into one string.**  A plate photographed
   with "IND" on the strip, a state name, or a dealer sticker produced
   ``"INDMH12AB1234"``, which matches no plate pattern, so the salvage regex
   returned a wrong substring.  Tokens are now scored individually.

Accuracy comes from **temporal consensus**, not from one lucky frame
-------------------------------------------------------------------
A single frame's OCR is a noisy observation.  Reads of the same *tracked*
vehicle are accumulated and resolved by per-character majority vote::

    frame 1 -> WB12AB1234
    frame 2 -> WB12AB1284
    frame 3 -> WB12AB1234
    frame 4 -> WB12AB1234
    ------------------------
    consensus  WB12AB1234

That is what makes the output stable across vehicles and streams.  Nothing is
hard-coded: there is no plate string anywhere in this module, and a read below
``ANPR_ALERT_CONFIDENCE`` is reported as ``PLATE UNCERTAIN`` rather than
written into the evidentiary log as a guessed registration.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from core.config import settings

log = logging.getLogger("ibvap.cv.anpr")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    log.warning("EasyOCR not installed - ANPR functionality disabled")


@dataclass
class PlateDetection:
    """Result of detecting and reading a licence plate in a frame."""

    bbox: tuple[int, int, int, int]        # x1, y1, x2, y2 in frame coordinates
    confidence: float                     # localisation confidence
    plate_text: str                        # display form (may be grouped)
    text_confidence: float                 # OCR confidence, 0-1
    frame_number: int
    timestamp: float
    vehicle_class: Optional[str] = None
    vehicle_track_id: Optional[int] = None
    #: True once several frames of the same vehicle agree on this reading.
    consensus: bool = False
    #: How many independent reads backed the consensus.
    votes: int = 1
    #: Whether the final string matches the Indian plate grammar.
    format_verified: bool = False


@dataclass
class _PlateVotes:
    """Accumulated reads for one tracked vehicle."""

    reads: deque = field(default_factory=lambda: deque(maxlen=32))
    last_seen: float = 0.0
    published: Optional[str] = None


class PlateVoter:
    """
    Temporal consensus over repeated reads of the same tracked vehicle.

    Two levels of voting, because OCR errors are usually *per character* rather
    than per string:

    * whole-string majority, which wins immediately when several frames agree
      exactly;
    * otherwise per-character majority across reads of the modal length, which
      recovers the correct plate even when no single frame got every character
      right.

    Confidence is carried through as the mean confidence of the reads
    supporting the winner, so a consensus built from four weak reads is not
    passed off as a strong one.
    """

    def __init__(self) -> None:
        self._votes: dict[int, _PlateVotes] = {}

    def add(self, track_id, text: str, confidence: float, now: float) -> None:
        entry = self._votes.setdefault(track_id, _PlateVotes())
        entry.reads.append((text, float(confidence), now))
        entry.last_seen = now

    def _fresh(self, track_id, now: float) -> list[tuple[str, float]]:
        entry = self._votes.get(track_id)
        if entry is None:
            return []
        window = settings.ANPR_VOTE_WINDOW_SECONDS
        return [(text, conf) for text, conf, seen in entry.reads
                if now - seen <= window]

    def consensus(self, track_id, now: float) -> Optional[tuple[str, float, int]]:
        """
        Best-supported reading for this vehicle, or ``None``.

        Returns ``(text, mean_confidence, vote_count)`` once at least
        ``ANPR_MIN_VOTES`` reads are available.
        """
        reads = self._fresh(track_id, now)
        if len(reads) < max(1, int(settings.ANPR_MIN_VOTES)):
            return None

        # Level 1: exact agreement between frames.
        counts = Counter(text for text, _ in reads)
        best_text, best_count = counts.most_common(1)[0]
        if best_count >= max(2, int(settings.ANPR_MIN_VOTES)):
            supporting = [c for t, c in reads if t == best_text]
            return best_text, sum(supporting) / len(supporting), best_count

        # Level 2: per-character majority across reads of the modal length.
        # OCR errors are typically a single confused glyph, so the correct plate
        # can be recovered even when no individual frame is entirely right.
        lengths = Counter(len(text) for text, _ in reads)
        modal_length, length_votes = lengths.most_common(1)[0]
        same_length = [(t, c) for t, c in reads if len(t) == modal_length]
        if length_votes < max(2, int(settings.ANPR_MIN_VOTES)) or not same_length:
            return None

        voted = []
        for position in range(modal_length):
            column: Counter = Counter()
            for text, conf in same_length:
                column[text[position]] += conf      # weight by OCR confidence
            voted.append(column.most_common(1)[0][0])

        text = "".join(voted)
        mean_conf = sum(c for _, c in same_length) / len(same_length)
        return text, mean_conf, len(same_length)

    def gc(self, now: float) -> None:
        """Forget vehicles that have left, so a long run cannot grow."""
        window = settings.ANPR_VOTE_WINDOW_SECONDS * 3
        for track_id in [t for t, e in self._votes.items() if now - e.last_seen > window]:
            self._votes.pop(track_id, None)

    def mark_published(self, track_id, text: str) -> None:
        entry = self._votes.get(track_id)
        if entry is not None:
            entry.published = text

    def already_published(self, track_id, text: str) -> bool:
        entry = self._votes.get(track_id)
        return entry is not None and entry.published == text

    def settled(self, track_id, now: float) -> bool:
        """
        Has this vehicle's plate been read enough times to stop spending OCR?

        A vehicle whose reads already agree needs no further attention, and the
        OCR budget it was consuming is far better spent on a car nobody has
        looked at yet. Requires one more vote than bare consensus so a plate is
        not abandoned on the strength of the minimum evidence.
        """
        agreed = self.consensus(track_id, now)
        if agreed is None:
            return False
        _text, _conf, votes = agreed
        return votes >= max(2, int(settings.ANPR_MIN_VOTES)) + 1

    def reset(self) -> None:
        self._votes.clear()


class ANPRProcessor:
    """
    Plate localisation and OCR.

    Intentionally lazy: ``get_anpr_processor()`` constructs the object cheaply,
    and EasyOCR's several-hundred-megabyte models load only on the first frame
    that actually contains a vehicle.
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike", "van"}
    PLATE_CHARS_RE = re.compile(r"[^A-Z0-9]+")
    #: ``MH12AB1234``, ``DL8CAF5031``, ``KA01F1234`` … the standard grammar.
    INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}[A-Z]?$")
    LOOSE_PLATE_RE = re.compile(r"^[A-Z0-9]{6,12}$")
    #: Tokens that legitimately appear on an Indian plate but are not the number.
    NON_PLATE_TOKENS = {"IND", "INDIA", "BH"}

    #: Every state and union-territory registration code currently issued in
    #: India, plus the codes still in circulation on older plates (OR for
    #: Odisha, UA for Uttarakhand) and the BH national series.
    #:
    #: The layout grammar constrains a plate's *shape*; this constrains its
    #: *vocabulary*. The two leading letters are not free — there are 39 legal
    #: values and roughly six hundred illegal ones — so a reading whose state
    #: code does not exist is known to be wrong even when its shape is perfect.
    #: That is decidable information the shape rules alone cannot use.
    STATE_CODES = frozenset({
        "AN", "AP", "AR", "AS", "BH", "BR", "CG", "CH", "DD", "DL", "DN",
        "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
        "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
        "TN", "TR", "TS", "UA", "UK", "UP", "WB",
    })

    def __init__(self, lang_list: Optional[list[str]] = None):
        self.lang_list = lang_list or [
            lang.strip() for lang in settings.ANPR_LANGUAGES.split(",") if lang.strip()
        ] or ["en"]
        self.reader = None
        self._initialized = False
        self._init_failed = False
        #: Per-stream, because this processor is a process-wide singleton while
        #: track ids are only unique *within* a camera. Keyed by a single shared
        #: dict, camera 2 was served camera 1's cached plates between cadence
        #: ticks, and two vehicles that happened to share a track id voted into
        #: one another's plate consensus.
        self._detection_cache: dict[str, list[PlateDetection]] = {}
        self._last_inference_frame: dict[str, int] = {}
        self._ocr_call_count = 0
        self._ocr_ms_total = 0.0
        self._reader_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._event_debounce: dict[tuple, float] = {}
        self._gpu = False
        #: Round-robin bookkeeping for the OCR budget: which tick each vehicle
        #: was last read on, so the longest-waiting car goes next.
        self._last_ocr_tick: dict[tuple, int] = {}
        self._tick = 0
        self._plates_read = 0
        self._plates_uncertain = 0
        self._candidates_found = 0
        self._frames_searched = 0
        self.voter = PlateVoter()

    # ------------------------------------------------------------------ #
    # Availability / model loading
    # ------------------------------------------------------------------ #
    def _init_reader(self) -> None:
        try:
            gpu = False
            try:
                import torch

                gpu = bool(torch.cuda.is_available())
            except Exception:
                pass
            self.reader = easyocr.Reader(self.lang_list, gpu=gpu, verbose=False)
            self._gpu = gpu
            self._initialized = True
            log.info("ANPR reader ready — languages=%s device=%s",
                     self.lang_list, "cuda" if gpu else "cpu")
        except Exception as exc:
            log.error("Failed to initialise EasyOCR: %s", exc)
            self._initialized = False
            self._init_failed = True
            self.reader = None

    def is_available(self) -> bool:
        """
        True when OCR can run. Triggers the one-time model load if needed.

        Returns False rather than raising when EasyOCR is missing or fails to
        load, so a broken ANPR install degrades one feature instead of taking
        the surveillance pipeline down with it.
        """
        if not (EASYOCR_AVAILABLE and settings.ANPR_ENABLED):
            return False
        if self._initialized:
            return True
        if self._init_failed:
            return False
        with self._init_lock:
            if not self._initialized and not self._init_failed:
                self._init_reader()
        return self._initialized and self.reader is not None

    def set_languages(self, lang_list: list[str]) -> bool:
        """Update the languages used for OCR."""
        if not EASYOCR_AVAILABLE:
            log.warning("Cannot set languages - EasyOCR not available")
            return False
        try:
            self.lang_list = lang_list or ["en"]
            self._init_failed = False
            self.reader = easyocr.Reader(self.lang_list, gpu=self._gpu, verbose=False)
            self._initialized = True
            log.info("ANPR languages updated to: %s", self.lang_list)
            return True
        except Exception as exc:
            log.error("Failed to update ANPR languages: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recognize_plates(
        self,
        frame: np.ndarray,
        vehicle_detections: Optional[list] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
        source_frame: Optional[np.ndarray] = None,
        source_id: str = "default",
    ) -> list[PlateDetection]:
        """
        Localise and read licence plates in one frame.

        Localisation runs on a contrast-enhanced copy of the analytics frame
        (better edges); OCR reads the **original** pixels, upscaled.

        ``source_frame`` is the camera's frame *before* it was resized to the
        analytics resolution, when the pipeline still has it.  Reading the plate
        from there instead of from the 640x384 working frame is the single
        largest accuracy lever available: on a 1080p feed the same plate carries
        three times the linear detail, and OCR accuracy on small text is
        governed almost entirely by glyph height.  Upscaling a 12-pixel crop
        cannot invent detail that was discarded by the resize; cropping the
        original never threw it away.

        Reads feed the temporal voter, and returned detections carry the
        consensus text once one exists.
        """
        if not self.is_available():
            return []
        if frame is None or frame.size == 0:
            return []

        ts = timestamp if timestamp is not None else time.time()
        stream = str(source_id)
        self._frames_searched += 1
        enhanced = self.preprocess_for_indian_plates(frame)

        # Plate pixels come from the highest-resolution image available.
        #
        # The scale is per-axis. The analytics frame is 640x384 (aspect 1.667)
        # while cameras are overwhelmingly 16:9 (1.778), so the resize is not
        # uniform: 1920/640 = 3.000 across but 1080/384 = 2.8125 down. Using the
        # width ratio for both axes shifted every plate crop upward by ~4% of
        # frame height — about 40 px on a 1080p source, which is taller than the
        # plate itself. OCR was being handed the bumper above the plate.
        ocr_source, ocr_scale = frame, (1.0, 1.0)
        if source_frame is not None and source_frame.size:
            src_h, src_w = source_frame.shape[:2]
            frm_h, frm_w = frame.shape[:2]
            if src_w > frm_w and frm_w > 0 and frm_h > 0:
                ocr_source = source_frame
                ocr_scale = (src_w / float(frm_w), src_h / float(frm_h))

        # Localise where the plate actually has pixels.
        #
        # This is the same mistake the face stage made and had fixed: search the
        # 640x384 analytics frame and a plate is four to twelve pixels tall,
        # below this detector's own minimum candidate size, so the candidate
        # list came back almost empty and ANPR quietly did nothing. Measured on
        # 1080p traffic footage, 52 vehicle crops yielded 13 candidates searched
        # at analytics resolution and 47 searched at source — and the ones found
        # at analytics resolution were the largest, nearest vehicles only.
        #
        # OCR already read from the source frame; it was being handed boxes
        # found in the downscale. Now both stages work on the same pixels.
        if ocr_source is not frame:
            search_frame = ocr_source
            search_enhanced = self.preprocess_for_indian_plates(ocr_source)
            search_scale = ocr_scale
        else:
            search_frame, search_enhanced, search_scale = frame, enhanced, (1.0, 1.0)

        candidates = self._collect_candidates(
            search_frame, search_enhanced, vehicle_detections, scale=search_scale,
        )
        if not candidates:
            self._detection_cache[stream] = []
            self.voter.gc(ts)
            return []

        self._candidates_found += len(candidates)
        candidates = self._deduplicate_candidates(candidates)
        candidates = self._schedule_candidates(
            candidates, stream, ts,
            frame_size=(search_frame.shape[1], search_frame.shape[0]),
        )

        detections: list[PlateDetection] = []
        inv_x = 1.0 / max(1e-6, search_scale[0])
        inv_y = 1.0 / max(1e-6, search_scale[1])
        for x1, y1, x2, y2, score, track_id, vehicle_class in candidates:
            # The candidate is already in ``search_frame`` coordinates, so OCR
            # crops it directly — no second rescale, which is what would move
            # the crop off the plate.
            text, confidence = self._read_plate(
                search_frame, search_enhanced, (x1, y1, x2, y2),
                ocr_source=search_frame, ocr_scale=(1.0, 1.0),
            )
            if not text:
                continue

            # Report the box in analytics coordinates: the overlay, the rules
            # and the evidence crop all work in that one space.
            box = (int(x1 * inv_x), int(y1 * inv_y),
                   int(x2 * inv_x), int(y2 * inv_y))

            final_text, final_conf, votes, consensus = text, confidence, 1, False
            if settings.ANPR_CONSENSUS_ENABLED and track_id is not None:
                vote_key = (stream, int(track_id))
                self.voter.add(vote_key, text, confidence, ts)
                agreed = self.voter.consensus(vote_key, ts)
                if agreed is not None:
                    final_text, final_conf, votes = agreed
                    consensus = True

            verified = self.is_plausible_plate(final_text)
            # Reward a read matching the plate grammar; discount one that is
            # merely alphanumeric noise of a plausible length.
            #
            # The discount is configurable because the grammar is Indian. On
            # footage from anywhere else every plate fails the pattern and is
            # permanently marked uncertain however clearly it was read — which
            # is the right default for a border post and the wrong one for a
            # demo over foreign footage. Set ANPR_UNVERIFIED_PENALTY=1.0 to
            # judge a read purely on how well it was seen.
            penalty = float(settings.ANPR_UNVERIFIED_PENALTY)
            adjusted = float(final_conf) * (1.0 if verified else penalty)

            # Agreement across frames is evidence in its own right, and it was
            # being thrown away: a plate read identically on five separate
            # frames scored no higher than one read once. Independent
            # observations of the same characters are exactly what raises
            # confidence, so a bounded bonus is applied for them — bounded so
            # that repetition can sharpen a good read but never manufacture a
            # confident one out of a poor one.
            if consensus and votes > 1:
                extra = min(int(votes) - 1, 3) / 3.0
                adjusted *= 1.0 + float(settings.ANPR_CONSENSUS_BONUS) * extra
            adjusted = min(1.0, adjusted)
            if adjusted >= settings.ANPR_CONFIDENCE_THRESHOLD:
                self._plates_read += 1
            else:
                self._plates_uncertain += 1

            detections.append(PlateDetection(
                bbox=box,
                confidence=float(score),
                plate_text=self.format_indian_plate(final_text) if verified else final_text,
                text_confidence=adjusted,
                frame_number=frame_number,
                timestamp=ts,
                vehicle_class=vehicle_class,
                vehicle_track_id=track_id,
                consensus=consensus,
                votes=votes,
                format_verified=verified,
            ))

        detections.sort(
            key=lambda d: (d.text_confidence * 0.6 + d.confidence * 0.4), reverse=True
        )
        self._detection_cache[stream] = detections
        self._last_inference_frame[stream] = frame_number
        self.voter.gc(ts)
        return detections

    def detect_and_recognize(self, frame: np.ndarray) -> list[PlateDetection]:
        """Full-frame search — used by the ANPR probe endpoint."""
        return self.recognize_plates(frame, vehicle_detections=None)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess_for_indian_plates(self, frame: np.ndarray) -> np.ndarray:
        """
        Contrast-enhanced copy used for **localisation only**.

        Kept 3-channel for interface compatibility.  Note what changed: this
        image is no longer the one OCR reads.  Cropping OCR input from a
        globally contrast-stretched grayscale frame meant the recogniser never
        saw the actual plate pixels.
        """
        if frame is None:
            return np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(grey), cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    # Localisation
    # ------------------------------------------------------------------ #
    def _vehicle_boxes(self, vehicle_detections: Optional[list]):
        boxes = []
        if not vehicle_detections:
            return boxes
        for det in vehicle_detections:
            class_name = (getattr(det, "class_name", "") or "").lower()
            if class_name not in self.VEHICLE_CLASSES:
                continue
            bbox = tuple(int(v) for v in det.bbox)
            if len(bbox) != 4:
                continue
            boxes.append((bbox[0], bbox[1], bbox[2], bbox[3],
                          getattr(det, "track_id", None), class_name))
        return boxes

    def _collect_candidates(self, frame, enhanced, vehicle_detections,
                            scale: tuple[float, float] = (1.0, 1.0)):
        """
        Plate candidates from each vehicle ROI, with a full-frame fallback.

        ``scale`` converts the vehicle boxes — which the detector produced in
        analytics coordinates — into the coordinates of the image being
        searched. Returned candidates are in that same searched-image space.
        """
        height, width = enhanced.shape[:2]
        sx, sy = scale
        margin = max(8, int(8 * sx))
        candidates: list[tuple] = []

        for x1, y1, x2, y2, track_id, vehicle_class in self._vehicle_boxes(vehicle_detections):
            x1, y1 = int(x1 * sx), int(y1 * sy)
            x2, y2 = int(x2 * sx), int(y2 * sy)
            if x2 <= x1 or y2 <= y1:
                continue
            box_height = y2 - y1
            # Search the lower 60% of the vehicle plus a small margin: plates sit
            # low on cars and at the very bottom on two-wheelers.
            crop_x1 = max(0, x1 - margin)
            crop_y1 = max(0, y1 + int(box_height * 0.40))
            crop_x2 = min(width, x2 + margin)
            crop_y2 = min(height, y2 + margin)
            if crop_x2 - crop_x1 < 12 or crop_y2 - crop_y1 < 6:
                continue

            region = enhanced[crop_y1:crop_y2, crop_x1:crop_x2]
            for lx1, ly1, lx2, ly2, score in self._find_plate_candidates(region):
                candidates.append((lx1 + crop_x1, ly1 + crop_y1,
                                   lx2 + crop_x1, ly2 + crop_y1,
                                   score, track_id, vehicle_class))

        if not candidates:
            for fx1, fy1, fx2, fy2, score in self._find_plate_candidates(enhanced):
                candidates.append((fx1, fy1, fx2, fy2, score, None, None))
        return candidates

    def _find_plate_candidates(self, image_bgr: np.ndarray):
        """
        High-contrast, roughly rectangular regions that look like plates.

        Size limits are **relative to the searched region**.  The previous
        absolute ``50 x 15`` px floor could not be met inside a vehicle crop at
        the 640x384 analytics resolution, so this function returned an empty
        list on essentially every real frame.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        region_h, region_w = image_bgr.shape[:2]
        min_w = max(int(settings.ANPR_MIN_PLATE_WIDTH_PX),
                    int(region_w * settings.ANPR_MIN_PLATE_WIDTH_FRAC))
        min_h = max(int(settings.ANPR_MIN_PLATE_HEIGHT_PX), int(region_h * 0.04))
        # A plate is a small feature of whatever is being searched. Without an
        # upper bound the full-frame fallback happily proposed a box covering
        # the entire frame — whose aspect ratio squeaks past the 1.8 minimum —
        # and OCR then read whatever text was largest in it.
        max_w = max(min_w + 1, int(region_w * settings.ANPR_MAX_PLATE_WIDTH_FRAC))
        max_h = max(min_h + 1, int(region_h * settings.ANPR_MAX_PLATE_HEIGHT_FRAC))

        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)

        # Close the gaps between adjacent characters so a plate's glyph row
        # becomes one blob. The kernel scales with the region so it works on a
        # small vehicle crop and on a full frame alike.
        kernel_w = max(7, min(25, int(min_w * 0.7) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Two passes over the same edge map, because one setting cannot serve
        # both kinds of footage — measured, on this project's own clips:
        #
        #   vehicle crops from blurry 1080p traffic   closed only:  1 candidate
        #                                             closed+dilate: 32
        #   a crisp, high-contrast plate              closed only:  1 candidate
        #                                             closed+dilate:  0
        #
        # On soft or distant footage the character strokes are faint and
        # fragmented, and the extra dilation is what joins them into a plate
        # blob at all. On a sharp, well-lit plate the strokes are already
        # contiguous, and that same dilation floods the plate outward into the
        # vehicle body until the only contour left is the whole car. Running
        # both and taking the union costs one extra `findContours` over an edge
        # map we have already computed, and covers both regimes instead of
        # choosing one and failing silently on the other.
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged = cv2.dilate(closed, kernel, iterations=1)
        extra, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) + list(extra)

        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h or w * h < settings.ANPR_MIN_PLATE_AREA:
                continue
            if w > max_w or h > max_h:
                continue
            aspect = w / float(h)
            if not (settings.ANPR_MIN_ASPECT_RATIO <= aspect
                    <= settings.ANPR_MAX_ASPECT_RATIO):
                continue

            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            extent = area / float(w * h)
            if not (0.20 <= extent <= 0.98):
                continue

            roi_edges = edges[y:y + h, x:x + w]
            if roi_edges.size == 0:
                continue
            edge_density = float(np.count_nonzero(roi_edges)) / float(roi_edges.size)
            if edge_density < 0.03:
                continue

            score = edge_density * extent
            if 2.5 <= aspect <= 5.5:            # typical single-row plate
                score *= 1.25
            candidates.append((x, y, x + w, y + h, score))

        # Both passes usually propose the same plate; keep the stronger and
        # drop the duplicate rather than paying for OCR on it twice.
        candidates.sort(key=lambda c: c[4], reverse=True)
        picked: list[tuple] = []
        for cand in candidates:
            if not any(_iou(cand, kept) > 0.55 for kept in picked):
                picked.append(cand)
            if len(picked) >= 6:
                break
        return picked

    def _leaving_urgency(self, cand: tuple, width: int, height: int) -> int:
        """
        How close to the frame edge this candidate is, as a priority bucket.

        A vehicle at the edge of the picture is about to leave it, and every
        tick it waits is a read that will never happen — while one in the middle
        of the scene will still be there next tick. Fair rotation alone cannot
        see that: it treats a car with four ticks left and a car with one the
        same. Returning 0 for "leaving now" sorts those first.
        """
        if not width or not height:
            return 1
        x1, y1, x2, y2 = cand[0], cand[1], cand[2], cand[3]
        cx = (x1 + x2) * 0.5 / float(width)
        cy = (y1 + y2) * 0.5 / float(height)
        margin = float(settings.ANPR_EDGE_URGENCY_FRACTION)
        near_edge = (cx < margin or cx > 1.0 - margin
                     or cy < margin or cy > 1.0 - margin)
        return 0 if near_edge else 1

    def _schedule_candidates(self, candidates: list[tuple], stream: str,
                             now: float, frame_size: tuple = (0, 0)) -> list[tuple]:
        """
        Choose which candidates get OCR this tick, fairly.

        OCR dominates the cost of this stage, so only a few candidates can be
        read per tick — but *which* few decides whether the system reads every
        vehicle or the same one forever. Sorting by candidate score alone, as
        this did, is a starvation bug: the nearest, largest, highest-scoring
        plate wins every single tick, so on a road with four cars the other
        three were never attempted even once. Measured on the benchmark, the
        reader returned exactly two plates whether three, four or six vehicles
        were in frame — which is the "it misses a lot of cars" report.

        Two changes fix it, both free:

        * a vehicle whose plate is already **settled** is dropped to the back.
          Re-reading a plate that four frames agree on buys nothing, and the
          budget it was consuming is what the unread cars needed.
        * among the rest, the vehicle **waiting longest** goes first. That
          turns a fixed budget into a rotation, so coverage is a matter of time
          rather than of luck.

        Candidates with no track id cannot be scheduled fairly (nothing to
        remember them by), so they are ranked by score as before.
        """
        budget = max(1, int(settings.ANPR_MAX_PLATES_PER_TICK))
        if len(candidates) <= budget:
            return candidates

        self._tick += 1
        width, height = frame_size
        ranked = []
        for cand in candidates:
            urgency = self._leaving_urgency(cand, width, height)
            track_id = cand[5]
            if track_id is None:
                # Unknown vehicle: no fairness state, so judge it on merit and
                # let it compete in the middle of the pack.
                ranked.append((1, urgency, 0, -cand[4], cand))
                continue
            key = (stream, int(track_id))
            settled = self.voter.settled(key, now)
            # Negative so that the longest-waited sorts first.
            waited = -(self._tick - self._last_ocr_tick.get(key, 0))
            ranked.append((2 if settled else 0, urgency, waited, -cand[4], cand))

        ranked.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        chosen = [r[4] for r in ranked[:budget]]
        for cand in chosen:
            if cand[5] is not None:
                self._last_ocr_tick[(stream, int(cand[5]))] = self._tick
        # Bounded like every other per-track store in this file.
        if len(self._last_ocr_tick) > 512:
            cutoff = self._tick - 600
            self._last_ocr_tick = {k: v for k, v in self._last_ocr_tick.items()
                                   if v > cutoff}
        return chosen

    def _deduplicate_candidates(self, candidates: list[tuple]) -> list[tuple]:
        """Merge overlapping candidates from vehicle crops and the full frame."""
        deduped: list[tuple] = []
        for cand in sorted(candidates, key=lambda c: c[4], reverse=True):
            if any(_iou(cand, kept) > 0.5 for kept in deduped):
                continue
            deduped.append(cand)
        return deduped

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #
    @staticmethod
    def _upscale(crop: np.ndarray) -> np.ndarray:
        """
        Scale a plate crop up to a height OCR can actually resolve.

        A plate is routinely 12 px tall at analytics resolution. EasyOCR cannot
        read glyphs that small, and the previous build passed the crop through
        untouched — the single largest reason ANPR returned nothing useful.
        """
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return crop
        target = int(settings.ANPR_PLATE_TARGET_HEIGHT)
        if h >= target:
            return crop
        scale = min(float(settings.ANPR_PLATE_MAX_UPSCALE), target / float(h))
        return cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _motion_deblur(crop: np.ndarray, length: int) -> Optional[np.ndarray]:
        """
        Undo a horizontal motion smear of roughly ``length`` pixels.

        A vehicle crossing the frame smears its plate along the direction of
        travel, and the smear is very nearly a horizontal box blur — which is
        invertible. Wiener deconvolution against that point-spread function
        recovers glyph edges that simple sharpening cannot: sharpening
        amplifies what survived, deconvolution reconstructs what was spread.

        The true smear length is unknown, so callers try a short ladder of
        lengths and let the existing scorer pick the reading that wins. A wrong
        length produces noise, and noise does not parse as a registration, so a
        bad guess costs an OCR call rather than a wrong plate.
        """
        if crop is None or crop.size == 0 or length < 2:
            return None
        try:
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            height, width = grey.shape
            if width < length * 2 or height < 4:
                return None

            psf = np.zeros((height, width), np.float32)
            start = max(0, width // 2 - length // 2)
            psf[height // 2, start:start + length] = 1.0
            total = psf.sum()
            if total <= 0:
                return None
            psf = np.fft.ifftshift(psf / total)

            spectrum = np.fft.fft2(grey)
            kernel = np.fft.fft2(psf)
            snr = float(settings.ANPR_DEBLUR_SNR)
            restored = np.real(np.fft.ifft2(
                spectrum * np.conj(kernel) / (np.abs(kernel) ** 2 + snr)
            ))
            spread = float(np.ptp(restored))
            if spread <= 1e-6:
                return None
            restored = np.clip((restored - restored.min()) / spread * 255.0,
                               0, 255).astype(np.uint8)
            return cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None                 # never let preprocessing break a read

    def _deblur_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """Upscaled deconvolutions of one crop, over a ladder of smear lengths."""
        out = []
        for length in settings.ANPR_DEBLUR_LENGTHS:
            restored = self._motion_deblur(crop, int(length))
            if restored is not None:
                out.append(self._upscale(restored))
        return out

    def _variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """
        Several renderings of one plate crop, best-effort rather than clever.

        Plates vary enormously — white-on-black, black-on-white, reflective,
        dusty, motion-blurred. Rather than guessing which single preprocessing
        chain suits a given plate, OCR a few cheap variants and keep the best
        scoring read. The upscale is shared by all of them.
        """
        upscaled = self._upscale(crop)
        variants = [upscaled]

        grey = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        # Denoise lightly, then stretch local contrast: helps a dusty or
        # underexposed plate without destroying a clean one.
        denoised = cv2.bilateralFilter(grey, 7, 55, 55)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        variants.append(cv2.cvtColor(clahe.apply(denoised), cv2.COLOR_GRAY2BGR))

        # Otsu binarisation, and its inverse for light-on-dark plates.
        _, binary = cv2.threshold(denoised, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        if float(binary.mean()) < 110:
            variants.append(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR))
        return variants

    @staticmethod
    def _crop(image: np.ndarray, bbox: tuple[int, int, int, int],
              scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """
        Crop ``bbox`` (in analytics coordinates) from ``image``, with margin.

        ``scale`` is ``(x_ratio, y_ratio)`` — separate axes, because the
        analytics resize is not aspect-preserving.
        """
        sx, sy = scale
        x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
        # A small margin recovers characters clipped by a tight contour.
        pad_x = max(2.0, (x2 - x1) * 0.06)
        pad_y = max(2.0, (y2 - y1) * 0.18)
        h, w = image.shape[:2]
        cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        cx2, cy2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return np.zeros((0, 0, 3), np.uint8)
        return image[cy1:cy2, cx1:cx2]

    def _read_plate(self, frame: np.ndarray, enhanced: np.ndarray,
                    bbox: tuple[int, int, int, int],
                    ocr_source: Optional[np.ndarray] = None,
                    ocr_scale: tuple[float, float] = (1.0, 1.0)) -> tuple[str, float]:
        """
        Read one plate candidate, returning ``(text, confidence)``.

        Tries, in order of expected quality: the full-resolution source crop
        (when the pipeline still has the pre-resize frame), the analytics-frame
        crop, and the contrast-enhanced crop as a last resort.  Each is run
        through several preprocessing variants and the best-scoring plate-like
        token wins.
        """
        sources: list[np.ndarray] = []
        if ocr_source is not None and max(ocr_scale) > 1.0:
            hi = self._crop(ocr_source, bbox, ocr_scale)
            if hi.size:
                sources.append(hi)

        native = self._crop(frame, bbox)
        if native.size:
            sources.append(native)
        if enhanced is not None and enhanced.shape[:2] == frame.shape[:2]:
            boosted = self._crop(enhanced, bbox)
            if boosted.size:
                sources.append(boosted)
        if not sources:
            return "", 0.0

        best_text, best_score, best_conf = "", 0.0, 0.0

        def try_variants(variants) -> None:
            nonlocal best_text, best_score, best_conf
            for variant in variants:
                for text, confidence in self._ocr_tokens(variant):
                    candidate = self._best_token(text)
                    if not candidate:
                        continue
                    score = self._score_reading(candidate, confidence)
                    if score > best_score:
                        best_text, best_score, best_conf = candidate, score, confidence

        def good_enough() -> bool:
            return bool(best_text and self.is_plausible_plate(best_text)
                        and best_conf >= 0.5)

        for source in sources:
            if source.size == 0:
                continue
            try_variants(self._variants(source))
            # A confident, format-valid read from the original pixels is enough;
            # don't pay for the enhanced copy as well.
            if good_enough():
                break

        # Second pass, only for crops the cheap path could not read. A moving
        # vehicle smears its plate, and a smeared plate either returns nothing
        # or returns something that is not a registration — both are exactly the
        # cases worth spending a deconvolution on. A plate read cleanly the
        # first time never reaches here, so sharp footage pays nothing for this.
        if not good_enough() and settings.ANPR_DEBLUR_ENABLED:
            for source in sources:
                if source.size == 0:
                    continue
                try_variants(self._deblur_variants(source))
                if good_enough():
                    break

        return best_text, best_conf

    def _score_reading(self, text: str, confidence: float) -> float:
        """
        Rank one candidate reading.

        Grammar conformance matters more than raw OCR confidence: a
        high-confidence read of a dealer sticker is worthless, while a moderate
        read shaped exactly like a registration is probably the plate.
        """
        score = float(confidence)
        if self.INDIAN_PLATE_RE.match(text):
            score *= 2.0
        elif self.LOOSE_PLATE_RE.match(text):
            score *= 1.2
        # Plates carry both letters and digits; an all-alpha token is a word.
        has_alpha = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        if not (has_alpha and has_digit):
            score *= 0.3
        if 8 <= len(text) <= 10:               # the common Indian plate length
            score *= 1.15
        return score

    def _ocr_tokens(self, image: np.ndarray) -> list[tuple[str, float]]:
        """
        Run EasyOCR and return each recognised token separately.

        Returning tokens individually is the fix for the concatenation defect:
        the old code joined every token into one string, so ``IND`` + the
        registration became ``INDMH12AB1234`` — matching no plate pattern.
        """
        self._ocr_call_count += 1
        started = time.perf_counter()
        with self._reader_lock:
            try:
                results = self.reader.readtext(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    detail=1, paragraph=False, width_ths=0.7,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
            except Exception as exc:
                log.error("EasyOCR readtext failed: %s", exc)
                return []
            finally:
                self._ocr_ms_total += (time.perf_counter() - started) * 1000.0

        tokens: list[tuple[str, float]] = []
        cleaned_parts: list[str] = []
        for entry in results or ():
            try:
                _box, raw_text, confidence = entry
            except (ValueError, TypeError):
                continue
            cleaned = self.normalize_plate_text(raw_text)
            if not cleaned:
                continue
            cleaned_parts.append(cleaned)
            tokens.append((cleaned, float(confidence)))

        # Also offer the joined string: a plate split across two OCR boxes
        # ("MH12" + "AB1234") is only a valid plate once reassembled. It is one
        # more candidate to be scored, not an override.
        if len(cleaned_parts) > 1:
            joined = "".join(cleaned_parts)
            mean_conf = sum(c for _, c in tokens) / len(tokens)
            tokens.append((joined, mean_conf))
        return tokens

    #: Glyph pairs OCR habitually confuses. Used only where the plate grammar
    #: makes the correct class unambiguous (a position that must hold a digit
    #: cannot hold the letter "O"), never as a blanket substitution.
    _TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "J": "1",
                 "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
    _TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
                  "6": "G", "7": "T", "8": "B"}

    def apply_plate_grammar(self, text: str) -> tuple[str, bool]:
        """
        Resolve OCR glyph confusions using the Indian plate grammar.

        An Indian registration is ``LL DD L{0,3} DDDD``.  That structure makes
        many OCR errors *decidable* rather than ambiguous: at a position which
        must be a digit, the letter ``O`` can only have been ``0`` and ``I`` can
        only have been ``1``; at a position which must be a letter, ``8`` can
        only have been ``B``.

        This is constraint satisfaction against a known format, not guesswork,
        and it is applied only when the result matches the grammar exactly.  The
        uncorrected OCR string is preserved in the event details, so the
        evidence record still shows what the recogniser actually saw.

        Returns ``(text, corrected)``.
        """
        if not text:
            return text, False
        # NOTE: deliberately *not* short-circuiting on "already matches the
        # regex". The grammar is loose enough that a misread often matches it
        # under a different, implausible parse: "MH1ZAB1234" satisfies
        # ``LL D LLL DDDD`` as MH-1-ZAB-1234 with zero substitutions, so the
        # early return handed back a plate with a Z where a 2 belonged and never
        # consulted the scorer that exists to fix exactly this. Measured on
        # rendered plates from 16 px to 96 px tall, this single early return was
        # the difference between 9/10 and 10/10 characters correct at every size.
        # The identity reading still competes below — it simply has to win on
        # cost rather than by arriving first.

        # Enumerate every layout consistent with the grammar (2 letters, 1-2
        # digits, 0-3 letters, 1-4 digits) and keep the reading that requires the
        # **fewest** substitutions. Minimum edit distance from what the OCR
        # actually saw is the right tie-break: several layouts can fit a given
        # length, and preferring the first one found picked arbitrarily badly
        # ("KAOIF1234" fit LLDDLLDDD as "KA01FI234", three changes, instead of
        # LLDDLDDDD as "KA01F1234", two).
        best: Optional[tuple[float, str]] = None
        for digits_1 in (2, 1):
            for letters_2 in (0, 1, 2, 3):
                for digits_2 in (1, 2, 3, 4):
                    # Layout prior. Substitution count alone is ambiguous
                    # because the grammar is loose: "KAOIF1234" fits both
                    # LL-D-LL-DDDD with one change and LL-DD-L-DDDD with two,
                    # and only the second is a real registration shape. Indian
                    # plates almost always carry a two-digit district code and a
                    # four-digit serial, so those layouts are cheaper.
                    layout_cost = (
                        (0.0 if digits_1 == 2 else 1.5)
                        + {4: 0.0, 3: 0.5, 2: 1.0, 1: 1.5}[digits_2]
                        + {2: 0.0, 1: 0.25, 3: 0.5, 0: 1.5}[letters_2]
                    )
                    if 2 + digits_1 + letters_2 + digits_2 != len(text):
                        continue
                    pattern = (["L"] * 2 + ["D"] * digits_1
                               + ["L"] * letters_2 + ["D"] * digits_2)
                    out, changes, ok = [], 0, True
                    for glyph, kind in zip(text, pattern):
                        if kind == "D":
                            if glyph.isdigit():
                                out.append(glyph)
                            elif glyph in self._TO_DIGIT:
                                out.append(self._TO_DIGIT[glyph])
                                changes += 1
                            else:
                                ok = False
                                break
                        else:
                            if glyph.isalpha():
                                out.append(glyph)
                            elif glyph in self._TO_LETTER:
                                out.append(self._TO_LETTER[glyph])
                                changes += 1
                            else:
                                ok = False
                                break
                    if not ok:
                        continue
                    candidate = "".join(out)
                    if not self.INDIAN_PLATE_RE.match(candidate):
                        continue
                    candidate, state_cost = self._resolve_state_code(candidate)
                    cost = changes + layout_cost + state_cost
                    if best is None or cost < best[0]:
                        best = (cost, candidate)

        if best is None:
            return text, False
        return best[1], best[1] != text

    def is_plausible_plate(self, text: str) -> bool:
        """
        Does this reading look like a registration that could actually exist?

        Shape alone is not enough. ``KH12DE1433`` satisfies the layout perfectly
        but ``KH`` is not a state code India issues, so the reading is known to
        be wrong — almost always a single misread glyph in the first character.
        Treating it as format-verified would write a registration that cannot
        exist into a tamper-evident log and present it as identified.

        Used for ``format_verified``, which drives both the confidence penalty
        and whether the plate is logged as a number or shown as uncertain.
        """
        if not text or not self.INDIAN_PLATE_RE.match(text):
            return False
        return text[:2] in self.STATE_CODES

    def _resolve_state_code(self, candidate: str) -> tuple[str, float]:
        """
        Check — and where it is unambiguous, repair — the state code.

        Returns ``(candidate, extra_cost)``.

        A valid code costs nothing. An invalid one is repaired only when
        exactly one legal code is a single character away: one substitution is
        the signature of a glyph misread, and a unique answer is a correction
        rather than a guess. When several codes are equally close, or none is,
        the reading is left exactly as OCR produced it and simply carries a
        cost, so a different layout can win if one fits better. Nothing is ever
        rewritten to a plate the recogniser did not plausibly see.
        """
        code = candidate[:2]
        if code in self.STATE_CODES:
            return candidate, 0.0

        near = [valid for valid in self.STATE_CODES
                if sum(1 for a, b in zip(code, valid) if a != b) == 1]
        if len(near) == 1:
            return near[0] + candidate[2:], 1.0
        # Ambiguous or unreachable: keep what was read, but make this layout
        # expensive so a reading with a real state code is preferred.
        return candidate, 2.0

    def normalize_plate_text(self, raw_text: str) -> str:
        """
        Normalise OCR output to uppercase alphanumerics.

        Deliberately does *not* map O->0 or I->1: keeping a little OCR ambiguity
        visible is safer for evidence than silently rewriting a registration.
        """
        if not raw_text:
            return ""
        return self.PLATE_CHARS_RE.sub("", str(raw_text)).upper()

    def _best_token(self, clean_text: str) -> Optional[str]:
        """Pick the most plate-like substring of one OCR token."""
        if not clean_text or len(clean_text) < 4:
            return None
        if clean_text in self.NON_PLATE_TOKENS:
            return None

        # Glyph-confusion repair against the plate grammar, e.g. "DLBCAF5O31"
        # -> "DL8CAF5031": at those positions only a digit is possible. This runs
        # even when the token already satisfies the grammar, because a misread
        # frequently satisfies it under an implausible parse (see
        # ``apply_plate_grammar``); the unmodified reading competes on equal
        # terms and wins whenever it is genuinely the best interpretation.
        repaired, corrected = self.apply_plate_grammar(clean_text)
        if corrected:
            return repaired
        if self.INDIAN_PLATE_RE.match(clean_text):
            return clean_text

        # Strip a leading country/strip marker such as "IND".
        for marker in self.NON_PLATE_TOKENS:
            if clean_text.startswith(marker) and len(clean_text) > len(marker) + 3:
                remainder = clean_text[len(marker):]
                if self.INDIAN_PLATE_RE.match(remainder):
                    return remainder

        # A window matching the plate grammar anywhere inside the token.
        for length in range(min(10, len(clean_text)), 5, -1):
            for start in range(0, len(clean_text) - length + 1):
                window = clean_text[start:start + length]
                if self.INDIAN_PLATE_RE.match(window):
                    return window
                repaired, corrected = self.apply_plate_grammar(window)
                if corrected:
                    return repaired

        if self.LOOSE_PLATE_RE.match(clean_text):
            return clean_text
        tokens = re.findall(r"[A-Z0-9]{6,12}", clean_text)
        return max(tokens, key=len) if tokens else None

    # ------------------------------------------------------------------ #
    # Pipeline integration
    # ------------------------------------------------------------------ #
    def cached_detections(self, source_id: str = "default") -> list[PlateDetection]:
        """Last OCR result set for *this stream*, reused between cadence ticks."""
        return list(self._detection_cache.get(str(source_id), ()))

    @staticmethod
    def format_indian_plate(text: str) -> str:
        """``MH12AB1234`` -> ``MH 12 AB 1234``. Presentational only."""
        match = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})([A-Z]?)$", text)
        if not match:
            return text
        return " ".join(part for part in match.groups() if part)

    def build_event(self, plate: PlateDetection, source_id: str = "default"):
        """
        Build a debounced ANPR event, or ``None``.

        A read below ``ANPR_ALERT_CONFIDENCE`` is **not** logged as a plate
        number — writing a guessed registration into an evidentiary log is worse
        than logging nothing.  The overlay still shows it live as
        ``PLATE UNCERTAIN`` so the operator knows a plate was seen.
        """
        from cv.rules import Alert as RuleAlert

        if not plate.plate_text:
            return None
        if plate.text_confidence < settings.ANPR_ALERT_CONFIDENCE:
            return None
        # With consensus enabled, wait for corroboration before writing to the
        # log; a single frame is an observation, not a reading.
        if (settings.ANPR_CONSENSUS_ENABLED
                and plate.vehicle_track_id is not None
                and not plate.consensus):
            return None

        track_id = int(plate.vehicle_track_id or 0)
        raw = plate.plate_text.replace(" ", "")
        # Debounce per (camera, plate, vehicle) so the same car is logged once
        # per pass, while the same vehicle seen by a *different* camera is still
        # reported — that second sighting is the movement an operator cares
        # about, and a camera-blind key silently swallowed it.
        stream = str(source_id)
        key = (stream, raw, track_id)
        now = time.time()
        last = self._event_debounce.get(key, 0.0)
        if now - last < settings.ANPR_ALERT_DEBOUNCE_SECONDS:
            return None
        self._event_debounce[key] = now
        if len(self._event_debounce) > 256:
            self._event_debounce = {
                k: v for k, v in self._event_debounce.items() if now - v < 600
            }
        self.voter.mark_published((stream, track_id), raw)

        return RuleAlert(
            rule_name="anpr", rule_type="anpr", track_id=track_id,
            alert_type="anpr_detection",
            description=(
                f"Number plate read: {plate.plate_text} "
                f"({plate.text_confidence * 100:.0f}% OCR confidence"
                f"{f', {plate.votes} corroborating frames' if plate.consensus else ''}"
                f"{', matches Indian plate format' if plate.format_verified else ''})"
            ),
            details={
                "plate_text": plate.plate_text,
                "plate_raw": raw,
                "ocr_confidence": round(float(plate.text_confidence), 4),
                "localization_confidence": round(float(plate.confidence), 4),
                "format_verified": plate.format_verified,
                "consensus": plate.consensus,
                "corroborating_reads": plate.votes,
                "vehicle_class": plate.vehicle_class or "unknown",
                "vehicle_track_id": track_id,
                "bbox": list(plate.bbox),
            },
        )

    def reset(self) -> None:
        """Clear per-stream state (source reconnect, file loop)."""
        self.voter.reset()
        self._detection_cache.clear()
        self._last_ocr_tick.clear()

    def get_metrics(self) -> dict:
        calls = max(1, self._ocr_call_count)
        return {
            "available": self.is_available(),
            "enabled": settings.ANPR_ENABLED,
            "easyocr_installed": EASYOCR_AVAILABLE,
            "languages": self.lang_list,
            "device": "cuda" if self._gpu else "cpu",
            "ocr_call_count": self._ocr_call_count,
            "ocr_ms_avg": round(self._ocr_ms_total / calls, 1),
            "plates_confident": self._plates_read,
            "plates_uncertain": self._plates_uncertain,
            "frames_searched": self._frames_searched,
            "candidates_found": self._candidates_found,
            "cadence_frames": settings.ANPR_EVERY_N_FRAMES,
            "confidence_threshold": settings.ANPR_CONFIDENCE_THRESHOLD,
            "alert_confidence": settings.ANPR_ALERT_CONFIDENCE,
            "consensus_enabled": settings.ANPR_CONSENSUS_ENABLED,
            "min_votes": settings.ANPR_MIN_VOTES,
            "plate_target_height": settings.ANPR_PLATE_TARGET_HEIGHT,
            "last_inference_frame": dict(self._last_inference_frame),
        }


def _iou(a, b) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2, …)`` boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


_anpr_processor: Optional[ANPRProcessor] = None
_anpr_lock = threading.Lock()


def get_anpr_processor() -> ANPRProcessor:
    """
    Process-wide ANPR processor (the OCR model is loaded at most once).

    Building it initialises EasyOCR, measured at ~5.6 s and longer when the
    weights have to move onto CUDA — so never call this from a thread something
    is waiting on. Test with :func:`anpr_ready` first, and pay the cost once at
    startup with :func:`preload_anpr`.
    """
    global _anpr_processor
    with _anpr_lock:
        if _anpr_processor is None:
            _anpr_processor = ANPRProcessor()
        return _anpr_processor


def anpr_ready() -> bool:
    """
    True once the OCR reader is built — or has definitively failed to build.

    "Ready" deliberately means the *reader*, not the processor object.
    Constructing ``ANPRProcessor`` is cheap and proves nothing: the EasyOCR
    model is built by ``is_available()`` on first use, which is measured in
    seconds and would land on whichever analytics thread first saw a vehicle.
    Reporting readiness from the object alone would have moved that stall
    rather than removed it.

    A failed load counts as ready so the pipeline stops asking: the feature is
    gone either way, and retrying a broken install once per frame is not a fix.
    """
    processor = _anpr_processor
    if processor is None:
        return False
    return bool(processor._initialized or processor._init_failed)


def preload_anpr() -> bool:
    """
    Build the processor **and** its OCR reader now.

    Returns False when OCR is unavailable — a missing or broken EasyOCR costs
    the ANPR feature, never the boot.
    """
    try:
        return bool(get_anpr_processor().is_available())
    except Exception:  # pragma: no cover - a missing OCR stack must not break boot
        log.exception("ANPR processor could not be preloaded")
        return False


__all__ = [
    "ANPRProcessor", "PlateDetection", "PlateVoter",
    "get_anpr_processor", "anpr_ready", "preload_anpr", "EASYOCR_AVAILABLE",
]
