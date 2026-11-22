"""Find the webcam inset in a screen recording, and frame the two apart.

The case: one source, the whole screen (a game, a desktop, an editor) with the
person composited into a corner. It is how OBS records and how every stream VOD
looks, and it is the case screencast_layout gets wrong. There the speaker band
is a large crop taken AROUND the face, which on a full-screen source means the
band is mostly more screen: measured on an Excel walkthrough, the output showed
the same spreadsheet twice, once whole and once enlarged.

What this needs instead is the inset's own rectangle, so the two bands can hold
genuinely different things — screen above, person below.

Finding it exactly is harder than it looks. The inset is not always a rectangle
(circles and clipped polygons are common), it has no reliable border, and on
gameplay the background moves as much as the person does, which rules out
temporal-variance tricks. What IS reliable is that the inset is anchored to a
corner and that a person detector fires inside it. So: locate the person, decide
which corner they are nearest, and grow a box from that corner until it covers
them with margin.
"""
import os

from ffmpeg_utils import blurred_backdrop

CORNER_MARGIN = 0.20   # a subject this far from an edge (as a fraction of the
                       # frame) still counts as anchored to it

# An inset subject is SMALL and OFF-CENTRE; a presenter filling the shot is
# neither. Requiring the detection to actually touch two edges was tried first
# and rejected: YOLO returns an upper-body box that stops at the chest, so a
# webcam pinned to the right edge measured 18% clear of the bottom and got
# thrown out. Measured on four real clips, these two properties separate the
# cases where "touching a corner" did not.
MAX_SUBJECT_HEIGHT = 0.35   # fraction of frame height

# Horizontal offset from centre, as a fraction of frame width. HORIZONTAL
# specifically: a composited camera is pinned to the left or right side, while a
# talking head is centred left-to-right even when their face sits high in the
# shot. Accepting offset on either axis let four talking heads through, all of
# them with a face near the top edge. Measured on those nine candidates the two
# groups do not overlap: real insets sat 0.37-0.43 from centre, the talking
# heads 0.01-0.12.
MIN_OFFSET = 0.18

# How much bigger than the detected head/upper body the inset is assumed to be.
# A webcam frames head and shoulders, and detectors return the head or the upper
# body, so the box has to grow to reach the inset's real edges.
INSET_PADDING = float(os.environ.get("INSET_PADDING", "1.45"))

# Guard rails as a fraction of frame height. Below the floor there is nothing
# worth showing; above the ceiling it stopped being an inset and the layout
# should not be used at all.
MIN_INSET_HEIGHT = 0.10
MAX_INSET_HEIGHT = 0.38

# An inset is pinned to the same pixels for the whole recording; a presenter
# walks about. Spread of the detected centres, as a fraction of frame width,
# above which this is a person in a room rather than a composited camera.
# Measured on the two false positives this rule was written for: a real inset
# moved 3-11px across samples, a presenter 316px.
MAX_CENTRE_SPREAD = 0.05

# Room kept above the detection, as a fraction of its height (see inset_box).
FACE_HEADROOM = 0.35
BODY_HEADROOM = 0.10

# A webcam window frames head and shoulders: the face is about a third of its
# width. Used for overlays that float mid-screen, where inset_box's corner
# anchoring does not apply and pushed the crop up onto the ceiling, and to grow
# a face detection to the inset in a corner.
OVERLAY_FACE_WIDTHS = 3.0


def nearest_corner(box, frame_w, frame_h):
    """Which corner the subject sits in: (horizontal, vertical) as strings.

    Returns e.g. ("left", "bottom"). A subject in the middle of the frame is
    reported by its nearest edges anyway; callers use `is_cornered` to reject.
    """
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return ("left" if cx < frame_w / 2 else "right",
            "top" if cy < frame_h / 2 else "bottom")


def is_cornered(box, frame_w, frame_h, margin=CORNER_MARGIN):
    """True when the subject looks like a webcam inset rather than the shot.

    Small and off-centre, not "touches two edges": see MAX_SUBJECT_HEIGHT.

    This is a sanity filter, not the decision. Whether the video is a screen
    recording with a camera in it at all is answered upstream by layout_picker;
    a game character standing in a corner would pass this test, and is kept out
    by the fact that nobody asked for this layout on that video.
    """
    x, y, w, h = box
    if h > frame_h * MAX_SUBJECT_HEIGHT:
        return False

    cx = (x + w / 2.0) / frame_w
    off_centre = abs(cx - 0.5) >= MIN_OFFSET

    near_edge = (x <= frame_w * margin or (x + w) >= frame_w * (1 - margin)
                 or y <= frame_h * margin or (y + h) >= frame_h * (1 - margin))
    return off_centre and near_edge


INSET_ASPECT = 16 / 9.0


def _place(x, y, w, h, new_w, new_h, frame_w, frame_h, horizontal, vertical):
    """Top-left corner of a new_w x new_h box grown around the detection."""
    # Anchor: keep the edge the subject is already near, grow the other way.
    if horizontal == "left":
        new_x = max(0, min(x - (new_w - w) / 2.0, frame_w - new_w))
        if x <= frame_w * CORNER_MARGIN:
            new_x = 0
    else:
        new_x = max(0, min(x + w + (new_w - w) / 2.0 - new_w, frame_w - new_w))
        if (x + w) >= frame_w * (1 - CORNER_MARGIN):
            new_x = frame_w - new_w

    # Vertical growth is biased upwards: detectors return the head or the chest,
    # and a portrait needs headroom, not more torso. Splitting the growth evenly
    # cropped the top of the head off on real clips.
    grow = new_h - h
    if vertical == "top":
        new_y = max(0, min(y - grow * 0.6, frame_h - new_h))
        if y <= frame_h * CORNER_MARGIN:
            new_y = 0
    else:
        # Pin to the bottom edge: the inset is there, and the detection sits
        # somewhere inside it rather than at its top.
        new_y = frame_h - new_h
        if (y + h) < frame_h * (1 - CORNER_MARGIN):
            new_y = max(0, min(y - grow * 0.6, frame_h - new_h))
    return new_x, new_y


def inset_box(box, frame_w, frame_h, padding=None, face=False):
    """Estimated inset rectangle (x, y, w, h) around a detected subject.

    Grown from the corner the subject is anchored to, so the box hugs the same
    edges the inset does instead of floating around the face.

    The height comes from assuming a 16:9 inset rather than from scaling the
    detection, because the detection is often a torso: YOLO returned a 246x86
    box for a webcam whose real rectangle was about 325x180, and padding that
    shape upwards still cut the head off. Deriving the height from the width
    and pinning the result to the corner reaches the top of the inset instead.

    ``face`` says the detection is a face box, not YOLO's upper body. A face is
    a third of a webcam's width (OVERLAY_FACE_WIDTHS), not 1/1.45 of it: grown
    by the body padding, a 128px face on a 720p Excel walkthrough gave a
    181px box pinned to the bottom edge that started 41px below the top of
    the face, and the enlarged band showed a forehead-less face (17-oct-2026).
    """
    if padding is None:
        padding = OVERLAY_FACE_WIDTHS if face else INSET_PADDING
    x, y, w, h = box
    horizontal, vertical = nearest_corner(box, frame_w, frame_h)

    new_w = min(frame_w, w * padding)
    # A face box is square, so tripling its height too would make a portrait
    # crop; the 16:9 assumption alone sets the height.
    new_h = new_w / INSET_ASPECT if face else max(h * padding, new_w / INSET_ASPECT)
    new_h = min(frame_h, new_h)
    # Never start below the head. A face box stops at the brows, so the hair
    # needs about a third of its height on top; YOLO's box already starts at
    # the top of the head.
    top_limit = max(0.0, y - h * (FACE_HEADROOM if face else BODY_HEADROOM))

    new_x, new_y = _place(x, y, w, h, new_w, new_h, frame_w, frame_h,
                          horizontal, vertical)
    if new_y > top_limit:
        if new_y + new_h >= frame_h - 1:
            # Pinned to the bottom edge: grow upwards (and keep 16:9) rather
            # than slide off the edge the inset sits on.
            new_h = frame_h - top_limit
            new_w = min(frame_w, max(new_w, new_h * INSET_ASPECT))
            new_x, _ = _place(x, y, w, h, new_w, new_h, frame_w, frame_h,
                              horizontal, vertical)
        new_y = top_limit

    return (int(round(new_x)), int(round(new_y)),
            int(round(new_w)), int(round(new_h)))


def usable(box, frame_h, min_ratio=MIN_INSET_HEIGHT, max_ratio=MAX_INSET_HEIGHT):
    """Whether an inset of this size is worth building a layout around."""
    return min_ratio * frame_h <= box[3] <= max_ratio * frame_h


def detect(video_path, samples=10):
    """Median inset rectangle across sampled frames, or None.

    None means "no usable inset here", which callers must treat as "use another
    layout" rather than as an error.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or not frame_w:
        cap.release()
        return None

    boxes = []
    try:
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            # Faces first, body as fallback: a webcam inset is small, and on a
            # 1080p source the face inside it is often too few pixels for
            # BlazeFace even at full resolution, while YOLO still finds the
            # person.
            faces = screencast_layout.detect_faces_full_res(frame)
            if faces:
                box = max(faces, key=lambda c: c['score'])['box']
            else:
                box = m.detect_person_yolo(frame)
            if not box:
                continue
            if not is_cornered(box, frame_w, frame_h):
                continue
            boxes.append(inset_box(box, frame_w, frame_h, face=bool(faces)))
    finally:
        cap.release()

    if len(boxes) < max(3, samples // 3):
        return None

    return stable_box(boxes, frame_w, frame_h)


def stable_box(boxes, frame_w, frame_h, min_agree=3):
    """Median of the inset boxes that agree with each other, or None."""
    import numpy as np

    arr = np.array(boxes, dtype=float)

    # Stability gate. Without it, a presenter standing in front of a screen and
    # an ordinary talking head both come back as "insets" and the layout puts a
    # zoom of someone's face under a copy of the same shot.
    #
    # Measured against the plain min-to-max range first, which was far too
    # brittle: one sample landing on a face elsewhere in the frame killed three
    # genuine insets. So cluster around the median instead and require most
    # detections to agree, which tolerates the odd stray without tolerating a
    # subject that actually moves.
    centres_x = arr[:, 0] + arr[:, 2] / 2.0
    centres_y = arr[:, 1] + arr[:, 3] / 2.0
    mid_x, mid_y = np.median(centres_x), np.median(centres_y)
    tolerance = frame_w * MAX_CENTRE_SPREAD
    close = ((np.abs(centres_x - mid_x) <= tolerance)
             & (np.abs(centres_y - mid_y) <= tolerance))
    if close.sum() < max(min_agree, 0.6 * len(boxes)):
        return None

    arr = arr[close]
    median = tuple(int(v) for v in np.median(arr, axis=0))
    if not usable(median, frame_h):
        return None
    return median


def overlay_box(box, frame_w, frame_h, widen=OVERLAY_FACE_WIDTHS):
    """A 16:9 box centred on a detected face (or upper body), clamped."""
    x, y, w, h = box
    new_w = min(frame_w, w * widen)
    new_h = min(frame_h, new_w / INSET_ASPECT)
    cx, cy = x + w / 2.0, y + h / 2.0
    new_x = max(0, min(cx - new_w / 2.0, frame_w - new_w))
    new_y = max(0, min(cy - new_h * 0.45, frame_h - new_h))
    return (int(round(new_x)), int(round(new_y)),
            int(round(new_w)), int(round(new_h)))


def detect_in_scene(video_path, start_f, end_f, samples=5):
    """A presenter window laid over the screen in this one scene, or None.

    ``detect`` is the clip-wide check and insists on a corner, because nothing
    upstream of it says the frame is a screen. Here the shot check already did
    (this runs only on scenes Gemini called "screen"), so a small face that
    stays put is the presenter's camera wherever it floats: a QuickTime or Loom
    window over a document sits mid-left, not in a corner, and the screen-only
    layout shrank both into a 1080px-wide strip (Ty Myers tutorial,
    6-oct-2026). What still rules a subject out: being big (that is the shot,
    not an overlay) or moving between samples.

    The caller only asks where the shot check said a live presenter camera is
    laid over the screen (presenter_cam). That gate is what keeps game
    characters, photos on slides and cover art out; the YOLO body fallback
    stays because BlazeFace misses the webcam face in about half the samples
    of a dim or small window (ticker_VUqnFqEHy2E), and without it a real
    presenter window is lost.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    if end_f - 1 < start_f:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = []
    try:
        for f_idx in np.linspace(start_f, end_f - 1, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, frame = cap.read()
            if not ok:
                continue
            faces = screencast_layout.detect_faces_full_res(frame)
            if faces:
                box = max(faces, key=lambda c: c['score'])['box']
                widen = OVERLAY_FACE_WIDTHS
            else:
                box = m.detect_person_yolo(frame)
                widen = INSET_PADDING
            if not box or box[3] > frame_h * MAX_SUBJECT_HEIGHT:
                continue
            boxes.append(overlay_box(box, frame_w, frame_h, widen))
    finally:
        cap.release()
    need = samples // 2 + 1
    if len(boxes) < need:
        return None
    return stable_box(boxes, frame_w, frame_h, min_agree=need)


def _centre_inside(det, box, slack):
    cx = det[0] + det[2] / 2.0
    cy = det[1] + det[3] / 2.0
    x, y, w, h = box
    return (x - slack <= cx <= x + w + slack) and (y - slack <= cy <= y + h + slack)


def _person_in_region(m, frame, box, slack):
    """Person boxes (frame coordinates) found in ``box`` grown by ``slack``."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, int(box[0] - slack)), max(0, int(box[1] - slack))
    x1 = min(fw, int(box[0] + box[2] + slack))
    y1 = min(fh, int(box[1] + box[3] + slack))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    person = m.detect_person_yolo(frame[y0:y1, x0:x1].copy())
    if not person:
        return []
    return [(person[0] + x0, person[1] + y0, person[2], person[3])]


def present_in_scene(video_path, box, start_f, end_f, samples=3):
    """Whether the webcam really sits in ``box`` during this scene.

    ``detect`` finds the box once per clip, from samples spread over every
    scene, and a clip cut from a tutorial mixes shots: the bubble is in the
    corner during the screen recording and gone in the editor shot next to it.
    Before this check every screen scene of a clip with an inset anywhere got
    the INSET layout, and on a scene without the camera the bottom band was an
    enlarged crop of a toolbar (measured on a Canva + QuickTime tutorial,
    6-oct-2026). A face or a person whose centre falls in the box in most
    samples is the answer.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    if end_f - 1 < start_f:
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    slack = max(box[2], box[3]) * 0.1
    hits = 0
    try:
        for f_idx in np.linspace(start_f, end_f - 1, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, frame = cap.read()
            if not ok:
                continue
            dets = [c['box'] for c in screencast_layout.detect_faces_full_res(frame)]
            if not any(_centre_inside(d, box, slack) for d in dets):
                person = m.detect_person_yolo(frame)
                dets = [person] if person else []
            if not any(_centre_inside(d, box, slack) for d in dets):
                # A small corner inset: on the whole frame the detector picks
                # the game character, so look inside the box itself.
                dets = _person_in_region(m, frame, box, slack)
            if any(_centre_inside(d, box, slack) for d in dets):
                hits += 1
    finally:
        cap.release()
    return hits * 2 > samples


MAX_CAMERA_RATIO = 0.40

# Per-job layout for stream VODs (``layouts=["general"]`` / ``["streamer"]``,
# set by app.layout_env). On a stream the one face the classifier finds is the
# webcam, so the automatic TRACK crop follows it and throws the gameplay away.
#   general  - every scene on the blurred-background layout
#   streamer - webcam on top, gameplay centre crop below (STREAMER scenes),
#              GENERAL on the scenes the webcam is not in
STREAM_MODES = ("general", "streamer")
STREAM_MODE = os.environ.get("STREAM_LAYOUT", "").strip().lower()
if STREAM_MODE not in STREAM_MODES:
    STREAM_MODE = ""

# Share of the output height the webcam band takes in the STREAMER layout. At
# 0.5 the seam is mid-frame, which is where layout_ranges.SEAM_LAYOUTS puts the
# captions ({\an5}); any other value needs subtitles.py to place them itself.
STREAMER_CAMERA_RATIO = 0.50


# --------------------------------------------------------------------------- #
# Small corner webcam on gameplay.
#
# ``detect`` looks at the whole frame, and on gameplay the whole frame is the
# wrong place to look: the game character is the biggest person in it, so the
# person detector returns them, and the streamer's face in a 20%-wide inset is
# too few pixels for the face detector (measured on a GTA stream VOD: 8 of 8
# samples gave the game character, 0 the webcam). Searched one corner at a time
# the same detector finds the streamer in the same spot in every sample, while
# hits in the other corners wander with the game.
#
# The inset's real rectangle then comes from its border: a straight edge that
# stays in one place while the gameplay next to it changes. A box grown from
# the person would be a guess (16:9 around a face); the border is exact, so the
# crop can stay inside it instead of spilling gameplay into the camera band.
# --------------------------------------------------------------------------- #

CORNER_REGION_W = 0.40   # share of the frame searched at each corner
CORNER_REGION_H = 0.55
# A corner inset is at most this big; past it ``detect`` is the right tool.
MAX_CORNER_INSET_W = 0.50
MAX_CORNER_INSET_H = 0.60
MIN_CORNER_INSET_H = 0.12
# A border line is a peak this many times the profile's median, and at least
# this share of the strongest peak (the outermost such peak wins: lines inside
# the inset, a shelf or a door frame, are shorter and sit nearer the person).
BORDER_PEAK_X_MEDIAN = 4.0
BORDER_PEAK_SHARE = 0.6

_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _corner_view(frame, corner):
    """The frame flipped so ``corner`` is top-left (a view, no copy)."""
    if corner.endswith("right"):
        frame = frame[:, ::-1]
    if corner.startswith("bottom"):
        frame = frame[::-1]
    return frame


def _unflip(box, corner, frame_w, frame_h):
    x, y, w, h = box
    if corner.endswith("right"):
        x = frame_w - x - w
    if corner.startswith("bottom"):
        y = frame_h - y - h
    return (int(x), int(y), int(w), int(h))


def _outer_peak(profile, lo, hi):
    """Index of the outermost strong peak of ``profile`` in [lo, hi), or None."""
    import numpy as np

    hi = min(hi, len(profile))
    if hi - lo < 3:
        return None
    seg = profile[lo:hi]
    top = float(seg.max())
    floor = float(np.median(profile)) or 1e-6
    if top < BORDER_PEAK_X_MEDIAN * floor:
        return None
    strong = np.nonzero(seg >= BORDER_PEAK_SHARE * top)[0]
    if not len(strong):
        return None
    i = int(strong[-1])
    # A border is often two pixels wide: take the stronger of the pair.
    if i > 0 and seg[i - 1] > seg[i]:
        i -= 1
    return lo + i


def find_border(grays, seed):
    """(w, h) of an inset anchored at the top-left of these (already flipped)
    grey frames, from its persistent right and bottom edges, or None.

    ``seed`` is the person's box in the same coordinates: the border lies
    beyond it. Two passes: a first guess from the rows/columns that are surely
    inside the inset (the person's own band), then the same search over the
    inset's whole height/width, where the real border is the only line that
    runs the full length."""
    import numpy as np

    fh, fw = grays[0].shape
    sx, sy, sw, sh = seed
    max_w, max_h = int(fw * MAX_CORNER_INSET_W), int(fh * MAX_CORNER_INSET_H)

    def cols(r0, r1):
        return np.median([np.abs(np.diff(g[r0:r1, :max_w + 2], axis=1)).mean(axis=0)
                          for g in grays], axis=0)

    def rows(c0, c1):
        return np.median([np.abs(np.diff(g[:max_h + 2, c0:c1], axis=0)).mean(axis=1)
                          for g in grays], axis=0)

    x_edge = _outer_peak(cols(sy, sy + sh), sx + sw, max_w)
    y_edge = _outer_peak(rows(sx, sx + sw), sy + sh, max_h)
    if x_edge is None or y_edge is None:
        return None
    # Second pass over the full extent, a few pixels either side of the guess.
    full_c, full_r = cols(0, y_edge), rows(0, x_edge)
    x2 = _outer_peak(full_c, max(sx + sw, x_edge - 12), x_edge + 13)
    y2 = _outer_peak(full_r, max(sy + sh, y_edge - 12), y_edge + 13)
    w, h = (x2 if x2 is not None else x_edge) + 1, (y2 if y2 is not None else y_edge) + 1
    if not (MIN_CORNER_INSET_H * fh <= h <= max_h and w <= max_w and 0.6 <= w / h <= 2.4):
        return None
    return w, h


def corner_detect(video_path, samples=10):
    """A small webcam inset in a corner of gameplay: (x, y, w, h) or None.

    Finds the corner where a person sits still in most samples, then measures
    the inset from its border (``find_border``); when there is no clear
    border, a 16:9 box grown from the person (``inset_box``), as ``detect``
    would. Exact boxes are good for ``streamer_filtergraph(fit="inside")``.
    """
    import cv2
    import numpy as np
    import main as m

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or not frame_w:
        cap.release()
        return None
    rw, rh = int(frame_w * CORNER_REGION_W), int(frame_h * CORNER_REGION_H)
    hits = {c: [] for c in _CORNERS}
    grays = []
    try:
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
            for corner in _CORNERS:
                region = np.ascontiguousarray(_corner_view(frame, corner)[:rh, :rw])
                box = m.detect_person_yolo(region)
                if box and box[3] <= rh * 0.95:
                    hits[corner].append(tuple(box))
    finally:
        cap.release()

    need = max(3, int(np.ceil(len(grays) * 0.6)))
    best = None
    for corner, boxes in hits.items():
        if len(boxes) < need:
            continue
        arr = np.array(boxes, dtype=float)
        cx, cy = arr[:, 0] + arr[:, 2] / 2, arr[:, 1] + arr[:, 3] / 2
        tol = frame_w * MAX_CENTRE_SPREAD
        close = (np.abs(cx - np.median(cx)) <= tol) & (np.abs(cy - np.median(cy)) <= tol)
        if close.sum() < need:
            continue
        if best is None or close.sum() > best[1]:
            best = (corner, int(close.sum()), tuple(int(v) for v in np.median(arr[close], axis=0)))
    if best is None:
        return None
    corner, _n, seed = best

    flipped = [np.ascontiguousarray(_corner_view(g, corner)) for g in grays]
    size = find_border(flipped, seed)
    if size:
        box = _unflip((0, 0, size[0], size[1]), corner, frame_w, frame_h)
        print(f"   🎮 Corner webcam ({corner}) measured from its border: {box}")
        return box
    box = inset_box(_unflip(seed, corner, frame_w, frame_h), frame_w, frame_h)
    if not usable(box, frame_h, min_ratio=0.08, max_ratio=MAX_CORNER_INSET_H):
        return None
    print(f"   🎮 Corner webcam ({corner}), no clear border, estimated: {box}")
    return box


def parse_facecam_box(text):
    """``"x,y,w,h"`` in percent of the frame -> four floats, or None when the
    text is empty. Raises ValueError for anything else (API input)."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip().rstrip("%") for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("facecam_box needs four numbers: x,y,w,h (percent)")
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        raise ValueError("facecam_box values must be numbers (percent)")
    if not all(0 <= v <= 100 for v in (x, y, w, h)) or w < 2 or h < 2:
        raise ValueError("facecam_box values must be 0-100, width and height at least 2")
    if x + w > 100.01 or y + h > 100.01:
        raise ValueError("facecam_box must fit inside the frame (x+w and y+h at most 100)")
    return x, y, w, h


def forced_box(frame_w, frame_h, env=None):
    """FACECAM_BOX (x,y,w,h in percent) as a pixel box, or None when unset or
    malformed (the API validated it already; a bad value here is ignored)."""
    raw = (os.environ if env is None else env).get("FACECAM_BOX", "")
    try:
        pct = parse_facecam_box(raw)
    except ValueError:
        print(f"   ⚠️ Ignoring FACECAM_BOX={raw!r}: expected x,y,w,h in percent")
        return None
    if not pct:
        return None
    x, y, w, h = (int(round(v / 100.0 * d)) for v, d in
                  zip(pct, (frame_w, frame_h, frame_w, frame_h)))
    w, h = min(w, frame_w - x), min(h, frame_h - y)
    return (x - x % 2, y - y % 2, max(2, w - w % 2), max(2, h - h % 2))


def find_webcam(video_path, frame_w, frame_h):
    """``(box, fit, forced)`` of the stream's webcam for the STREAMER layout.

    FACECAM_BOX wins (forced: every scene uses it). Then ``detect`` (a large
    or clearly visible camera; its box is an estimate, so the band crop may
    grow past it) and, when that finds nothing, ``corner_detect`` (a small
    inset on gameplay; its box is measured, so the crop stays inside it).
    ``box`` is None when there is no webcam to stack."""
    box = forced_box(frame_w, frame_h)
    if box:
        return box, "inside", True
    box = detect(video_path)
    if box:
        return box, "grow", False
    box = corner_detect(video_path)
    return box, "inside", False


def _band_crop(box, orig_w, orig_h, target_aspect, inside=False):
    """``box`` widened (or heightened) to ``target_aspect`` around its centre,
    clamped to the frame and rounded to even numbers, so scaling it into a band
    of that aspect is uniform and never stretches a face.

    ``inside``: trim the box to the aspect instead of growing it, for a box
    that is the camera's exact rectangle (growing it would show the gameplay
    around the inset in the camera band). Trimmed from the bottom when
    shortened, where a webcam frames shoulders rather than the face."""
    x, y, w, h = box
    w, h = max(2, w), max(2, h)
    if inside:
        if w / float(h) < target_aspect:
            h = max(2, int(round(w / target_aspect)))
        else:
            new_w = max(2, int(round(h * target_aspect)))
            x = int(round(x + (w - new_w) / 2.0))
            w = new_w
        x = max(0, min(x, orig_w - w))
        y = max(0, min(y, orig_h - h))
        w -= w % 2
        h -= h % 2
        x += x % 2
        y += y % 2
        return x, y, w, h
    if w / float(h) < target_aspect:
        want_w = min(orig_w, int(round(h * target_aspect)))
        x = int(round(x + w / 2.0 - want_w / 2.0))
        w = want_w
    else:
        want_h = min(orig_h, int(round(w / target_aspect)))
        y = int(round(y + h / 2.0 - want_h / 2.0))
        h = want_h
    x = max(0, min(x, orig_w - w))
    y = max(0, min(y, orig_h - h))

    w -= w % 2
    h -= h % 2
    x -= x % 2
    y -= y % 2
    return x, y, w, h


def gameplay_crop(orig_w, orig_h, band_aspect, box):
    """(x, y, w, h) of the gameplay band: a full-height crop of ``band_aspect``
    from the centre of the frame, moved sideways off the webcam ``box`` when the
    centre crop would cut through it. Kept inside the frame, so a webcam too
    wide to step around still shows at the crop's edge rather than failing."""
    h = orig_h - orig_h % 2
    w = int(round(h * band_aspect))
    if w > orig_w:
        # A source narrower than the band (square, 4:5): full width, and the
        # height trimmed to the band's aspect instead of squashing it in.
        w = orig_w
        h = int(round(w / band_aspect))
        h -= h % 2
    w -= w % 2
    x = (orig_w - w) // 2
    bx, _by, bw, _bh = box
    if x < bx + bw and bx < x + w:
        if bx + bw / 2.0 >= orig_w / 2.0:
            x = bx - w        # webcam on the right: end the crop at its left edge
            x -= x % 2        # even, rounding away from the webcam
        else:
            x = bx + bw       # webcam on the left: start the crop at its right edge
            x += x % 2
    x = max(0, min(x, orig_w - w))
    x -= x % 2
    y = (orig_h - h) // 2
    y -= y % 2
    return x, y, w, h


def streamer_filtergraph(orig_w, orig_h, out_w, out_h, box,
                         camera_ratio=STREAMER_CAMERA_RATIO, fit="grow"):
    """Webcam on top, gameplay below, stacked with no filler.

    The opposite order from inset_filtergraph on purpose: there the screen is
    the subject and is kept whole; here the gameplay is cropped to fill its band
    and the streamer's face leads the frame, the way stream clips are cut.

    ``fit="inside"`` for a measured box (``find_webcam``): the camera crop is
    trimmed to the band instead of growing past the inset's edges.
    """
    cam_h = int(out_h * camera_ratio)
    cam_h -= cam_h % 2
    game_h = out_h - cam_h

    cx, cy, cw, ch = _band_crop(box, orig_w, orig_h, out_w / float(cam_h),
                                inside=(fit == "inside"))
    gx, gy, gw, gh = gameplay_crop(orig_w, orig_h, out_w / float(game_h), box)

    return (
        f"[0:v]split=2[ca][ga];"
        f"[ca]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{cam_h}[cam];"
        f"[ga]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{game_h}[game];"
        f"[cam][game]vstack=inputs=2,setsar=1[v]"
    )


def inset_filtergraph(orig_w, orig_h, out_w, out_h, box, camera_ratio=None):
    """Screen on top at full width, the webcam inset below.

    The screen keeps its whole width, which is the point: a game HUD or a
    desktop puts what matters at the edges. The inset is scaled up so the person
    reads at a size the source never gave them.

    The camera band takes its height from the INSET'S OWN aspect ratio rather
    than a fixed share of the frame. A fixed share was tried first and stretched
    every face sideways: a 16:9 inset forced into a 2:1 band is a 12% horizontal
    stretch, and it is immediately visible on a face.
    """
    box_w = max(2, box[2])
    box_h = max(2, box[3])

    if camera_ratio is not None:
        cam_h = int(out_h * camera_ratio)
    else:
        cam_h = int(round(out_w * box_h / float(box_w)))
    cam_h = min(cam_h, int(out_h * MAX_CAMERA_RATIO))
    cam_h -= cam_h % 2

    screen_h = int(round(out_w * orig_h / float(orig_w)))
    screen_h -= screen_h % 2
    screen_h = max(2, min(screen_h, out_h - cam_h - 2))

    # Widen (or heighten) the crop to the band's aspect so the scale below is
    # uniform. Clamped to the frame, so an inset hard against an edge simply
    # keeps whatever it can reach.
    x, y, w, h = _band_crop(box, orig_w, orig_h, out_w / float(cam_h))

    filler_h = out_h - screen_h - cam_h

    return (
        f"[0:v]split=3[bga][sa][ca];"
        # Blurred backdrop so the leftover strip is not a black bar. Scaled by
        # HEIGHT: scaling a 16:9 source to 1080 wide gives 608 tall, and there
        # is no 1920-tall crop to take out of that.
        f"[bga]{blurred_backdrop(out_w, out_h, 14)}[bg];"
        f"[sa]scale={out_w}:{screen_h}[screen];"
        f"[ca]crop=w={w}:h={h}:x={x}:y={y},scale={out_w}:{cam_h}[cam];"
        f"[bg][screen]overlay=x=0:y={filler_h // 2}[withscreen];"
        f"[withscreen][cam]overlay=x=0:y={filler_h // 2 + screen_h},"
        f"setsar=1[v]"
    )
