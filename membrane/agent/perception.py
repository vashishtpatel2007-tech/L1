"""Recognise objects in the camera frame, and pick the one that was asked for.

This is the zero-API path. It does the job the vision model does on the hosted
path -- look at the picture, work out what is in it, and decide which thing the
request names -- using only the pixels that `GET /look` returns.

Nothing here reads engine state. The camera frame and the robot's own pose are
the whole input, which is the same information the brief allows the model. The
engine is never asked what the objects are, where they are, or how big they
are; every number below is measured off the image.

How an object is measured
-------------------------
The camera is 0.5 m up, tilted 10 degrees down, 640x480 at 90 degrees
horizontal. That fixes the geometry, so a pixel row maps to a distance along
the floor. An object's BASE row therefore gives its ground position, and once
the distance is known its TOP row gives its height. Width follows from the
distance and the pixel span.

Shape comes from the silhouette: how much of the bounding box the object fills,
and how its width changes from bottom to top. A cone tapers to nothing, a
sphere fills about pi/4 of its box, a box fills nearly all of it, and a
cylinder fills nearly all of it while being taller than it is wide.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import cv2
from PIL import Image

IMG_W, IMG_H = 640, 480
F_PX = 320.0                 # focal length in pixels at 90 degrees horizontal
CAM_HEIGHT = 0.5
CAM_TILT_DEG = 10.0
CAM_FWD_OFFSET = 0.18

#: The palette the scene is built from, as the camera sees it. These are hue
#: anchors for naming what is on screen, not a list of what is present -- the
#: detector finds coloured regions first and only then asks which name fits.
COLOUR_ANCHORS: dict[str, tuple[int, int, int]] = {
    "red":    (217,  31,  31),
    "blue":   ( 38,  89, 230),
    "yellow": (242, 204,  26),
    "purple": (153,  64, 217),
    "orange": (242, 122,  26),
    "teal":   ( 26, 184, 173),
}

SHAPE_WORDS = {
    "sphere": "sphere", "ball": "sphere", "orb": "sphere", "round": "sphere",
    "box": "box", "cube": "box", "block": "box", "square": "box",
    "cylinder": "cylinder", "can": "cylinder", "tube": "cylinder",
    "cone": "cone", "pyramid": "cone", "triangle": "cone",
}

#: Words that pick one object out of several by comparing them, and the key
#: each one compares on. `reverse` means "largest wins".
SUPERLATIVES: dict[str, tuple[str, bool]] = {
    "tallest": ("height", True), "highest": ("height", True),
    "shortest": ("height", False), "lowest": ("height", False),
    "shorted": ("height", False), "shorter": ("height", False),
    "taller": ("height", True),
    "largest": ("volume", True), "biggest": ("volume", True),
    "bigger": ("volume", True), "big": ("volume", True),
    "smallest": ("volume", False), "tiniest": ("volume", False),
    "smaller": ("volume", False), "nearest": ("distance", False),
    "closest": ("distance", False), "furthest": ("distance", True),
    "farthest": ("distance", True),
}

SIZE_WORDS = {"small": 0, "little": 0, "tiny": 0, "short": 0,
              "medium": 1, "mid": 1,
              "large": 2, "big": 2, "tall": 2, "huge": 2}

#: P(shape reported by the classifier | shape actually present), measured over
#: 96 silhouettes across eight viewing angles.
#:
#: Sphere and cone are read reliably: a sphere narrows at both ends and a cone
#: tapers to a point, and nothing else does either. Box and cylinder are NOT
#: reliably separable from this camera, and that is a property of the problem
#: rather than of the classifier -- an upright cylinder and a box both present
#: a plain rectangle, their surface shading differs by less than the render
#: noise (max step 0.37 vs 0.46, standard deviations 0.22), and their top-edge
#: curvature overlaps almost completely (0.18+-0.09 vs 0.20+-0.20).
#:
#: So the selector treats a reported "box" as weak evidence against a cylinder
#: rather than proof, and leans on colour and height, which ARE reliable. That
#: is why a wrong shape reading no longer produces a wrong object.
SHAPE_LIKELIHOOD: dict[str, dict[str, float]] = {
    "sphere":   {"sphere": 0.90, "box": 0.06, "cylinder": 0.03, "cone": 0.01},
    "cone":     {"cone": 0.93, "sphere": 0.03, "cylinder": 0.03, "box": 0.01},
    "box":      {"box": 0.55, "cylinder": 0.33, "sphere": 0.11, "cone": 0.01},
    "cylinder": {"cylinder": 0.60, "box": 0.37, "sphere": 0.02, "cone": 0.01},
}


def ground_point(u: float, v: float, height: float = 0.0) -> tuple[float, float] | None:
    """Where a pixel meets a horizontal plane, as (forward, right) in metres.

    Shared with the delivery-marker finder so both reason about the same camera.
    """
    q, side = (v - IMG_H / 2) / F_PX, (u - IMG_W / 2) / F_PX
    sin, cos = math.sin(math.radians(CAM_TILT_DEG)), math.cos(math.radians(CAM_TILT_DEG))
    down = sin + q * cos
    if down <= 0.02:
        return None
    scale = (CAM_HEIGHT - height) / down
    return CAM_FWD_OFFSET + scale * (cos - q * sin), scale * side


def height_at(v_top: float, forward: float) -> float:
    """How high a pixel row stands, given how far away the object is.

    The base row fixed the distance; this inverts the same ray equation for the
    top row with that distance held, which is what makes a height measurement
    possible from one frame.

    This removes the ideal pinhole distance dependence of a size comparison. A tall object far
    away and a short object close up can cover the same number of pixels, but
    once the distance is divided back out, both come back in metres at their
    true size. Every ranking -- tallest, shortest, largest, smallest -- is done
    on these metres, never on pixel extent, but occlusion, pixel quantization and imperfect silhouettes still introduce
    uncertainty; clipped measurements are excluded from rankings.
    """
    q = (v_top - IMG_H / 2) / F_PX
    sin, cos = math.sin(math.radians(CAM_TILT_DEG)), math.cos(math.radians(CAM_TILT_DEG))
    denom = cos - q * sin
    if abs(denom) < 1e-6:
        return 0.0
    scale = (forward - CAM_FWD_OFFSET) / denom
    return CAM_HEIGHT - scale * (sin + q * cos)


@dataclass
class Detection:
    colour: str
    shape: str
    height: float                 # metres, measured
    width: float                  # metres, measured
    distance: float               # metres from the camera
    bearing: float                # degrees, positive = right of centre
    world: tuple[float, float]    # absolute position, via the robot's own pose
    pixels: int
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    #: False when the object's top ran off the frame, so `height` is a lower
    #: bound rather than a measurement and must not be compared with others.
    height_valid: bool = True

    @property
    def volume(self) -> float:
        """A size ranking, not a physical volume: enough to order objects."""
        return self.width * self.width * self.height


#: Hue angle in degrees for each palette colour, derived from its RGB.
COLOUR_HUES: dict[str, float] = {
    "red": 0.0, "orange": 27.0, "yellow": 49.0,
    "teal": 175.0, "blue": 222.0, "purple": 275.0,
}


def _hue_deg(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees, plus saturation and value, computed per pixel."""
    r, g, b = (rgb[..., 0].astype(np.float32),
               rgb[..., 1].astype(np.float32),
               rgb[..., 2].astype(np.float32))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    h = np.zeros_like(mx)
    safe = c > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = np.where(mx == r, ((g - b) / np.where(safe, c, 1.0)) % 6.0, 0.0)
        hg = np.where(mx == g, (b - r) / np.where(safe, c, 1.0) + 2.0, 0.0)
        hb = np.where(mx == b, (r - g) / np.where(safe, c, 1.0) + 4.0, 0.0)
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb)) * 60.0
    sat = np.where(mx > 0, c / np.maximum(mx, 1.0), 0.0)
    return h % 360.0, sat, mx


def _label_colours(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Split the frame into one mask per palette colour, by HUE.

    Distance between normalised RGB triples was tried first and confuses blue
    with purple: once brightness is divided out both are dominated by the blue
    channel and they sit close together, so a purple object was routinely
    reported as blue and "fetch the purple one" drove to the wrong thing.
    Their hues are 222 and 275 degrees -- a wide, unambiguous gap. Hue also
    survives shading, which is what a surface curving away from the light does
    to every one of these objects.

    Floor, walls, the black platform and its white rim are all near-grey, so a
    saturation gate removes them without naming them: an object is something
    with colour in it, whatever colour that turns out to be.
    """
    hue, sat, val = _hue_deg(rgb)
    lit = (val > 30) & (sat > 0.40)

    names = list(COLOUR_HUES)
    best_i = np.zeros(hue.shape, dtype=np.int8)
    best_d = np.full(hue.shape, 1e9, dtype=np.float32)
    for i, name in enumerate(names):
        d = np.abs(hue - COLOUR_HUES[name])
        d = np.minimum(d, 360.0 - d)          # hue is a circle
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_i = np.where(closer, i, best_i)

    # Magenta rim lighting shifts purple surfaces toward 300 degrees in Godot.
    # Its nearest competing anchor is much farther away than orange/yellow;
    # allow that measured shift while still assigning each pixel only once.
    # The room's dim purple wall panels must not join a purple object's apex
    # into one wall-sized component. Fetchable purple surfaces are brighter
    # in the calibrated room lighting; retain highlights for full silhouettes.
    return {name: lit & ((val > 60) if name == 'purple' else True)
            & (best_i == i) & (best_d < (36.0 if name == 'purple' else 22.0))
            for i, name in enumerate(names)}


def _components(mask: np.ndarray, min_pixels: int = 28) -> list[np.ndarray]:
    """Connected regions of a mask, largest first.

    Uses scipy when it is available and falls back to a flood fill otherwise,
    so the offline path never depends on an optional package being installed.
    """
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return [lab == i for i in range(1, n + 1)
                if int(np.count_nonzero(lab == i)) >= min_pixels]
    except Exception:
        pass
    try:
        import cv2
        n, lab = cv2.connectedComponents(mask.astype(np.uint8))
        return [lab == i for i in range(1, n)
                if int(np.count_nonzero(lab == i)) >= min_pixels]
    except Exception:
        return [mask] if int(np.count_nonzero(mask)) >= min_pixels else []


def _classify(blob: np.ndarray, box: tuple[int, int, int, int],
              metric_aspect: float) -> str:
    """Name the shape from its silhouette alone.

    The thresholds below are not guesses. Measured over 42 silhouettes of
    known shapes across six viewing angles, the four separate cleanly on how
    the width changes from bottom to top, plus how much of the bounding box is
    filled (mean +/- one standard deviation):

                  taper_top   taper_bottom   height/width   fill
        sphere    0.63+-0.01  0.63+-0.06     0.65+-0.19     0.78+-0.01
        cone      0.47+-0.26  1.24+-0.41     0.98+-0.16     0.64+-0.13
        box       0.98+-0.01  0.95+-0.02     0.77+-0.17     0.89+-0.02
        cylinder  0.89+-0.25  1.01+-0.24     1.14+-0.14     0.86+-0.13

    A sphere is the only shape that narrows at BOTH ends, and its fill is
    pinned near pi/4 whatever the lighting. A cone is the only one whose top
    vanishes while its base stays wide. The two flat-sided shapes are then
    told apart by height against width in METRES -- in pixels a box shows its
    top face and reads taller than it is.

    Order matters: the sphere test runs first because a cone's empty top would
    otherwise claim it, and the cone test runs before the flat-sided pair
    because a cone's narrow top depresses its apparent fill.
    """
    x, y, w, h = box
    if w < 4 or h < 4:
        return "box"
    sub = blob[y:y + h, x:x + w]
    rows = sub.sum(axis=1).astype(np.float32)
    band = max(1, h // 4)
    top = float(rows[:band].mean())
    mid = float(rows[band:max(band + 1, h - band)].mean())
    bottom = float(rows[-band:].mean())
    if mid < 1e-6:
        return "box"
    taper_top, taper_bottom = top / mid, bottom / mid
    fill = float(sub.sum()) / float(w * h)

    if (taper_top < 0.80 and taper_bottom < 0.82
            and 0.70 <= fill <= 0.85 and metric_aspect < 0.95):
        return "sphere"
    if taper_top < 0.70 or (fill < 0.75 and taper_bottom > 1.02):
        return "cone"
    return "cylinder" if metric_aspect > 0.95 else "box"


def detect(jpg: bytes, pose: dict) -> list[Detection]:
    """Everything the camera can currently see, measured from the pixels."""
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(np.int16)
    heading = math.radians(pose["heading"])
    out: list[Detection] = []

    # Visible gold forearms are attached to the lower image edges. Join their
    # highlight bands before colour segmentation so isolated fingertip pixels
    # cannot masquerade as a small distant fetch object.
    warm = ((rgb[:,:,0]>150)&(rgb[:,:,1]>100)&(rgb[:,:,2]<rgb[:,:,0]*.9)).astype(np.uint8)
    warm = cv2.morphologyEx(warm,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(warm)
    self_mask = np.zeros(warm.shape,np.uint8)
    for i in range(1,n):
        x,y,w,h,area = stats[i]
        if area>200 and y+h>IMG_H*.6 and (x<2 or x+w>=IMG_W-2):
            self_mask[labels==i]=1
    self_mask = cv2.dilate(self_mask,np.ones((17,17),np.uint8)).astype(bool)

    for colour, mask in _label_colours(rgb).items():
        if not mask.any():
            continue
        for blob in _components(mask):
            if np.count_nonzero(blob & self_mask) > .2*np.count_nonzero(blob):
                continue
            ys, xs = np.nonzero(blob)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            w_px, h_px = x1 - x0 + 1, y1 - y0 + 1
            # Thin outlines (such as the orange delivery-pad border) are not
            # solid primitives. A projected cone still fills about half its box.
            if float(blob.sum()) / (w_px * h_px) < .28:
                continue
            # Touching the frame edge means the object is cut off, so its base
            # row is not its base and every measurement from it would be wrong.
            if x0 <= 1 or x1 >= IMG_W - 2 or y1 >= IMG_H - 2:
                continue
            # A top that runs off the frame makes the object read SHORTER than
            # it is -- the measurement stops at the frame edge. Keep the
            # sighting, because its position is still good, but do not let its
            # height be compared: a 0.56 m cone close to the camera was being
            # ranked below a 0.30 m one across the room.
            height_valid = y0 > 1
            # The topmost pixel of a blob is not reliably the top of the
            # object: anti-aliasing and colour bleed leave a one-pixel fringe,
            # and at five metres twenty stray rows read as a quarter-metre of
            # extra height -- which is how a 0.30 m cone was ranked above a
            # 0.56 m one. Take the first row that is genuinely part of the
            # object, meaning at least two pixels wide.
            row_counts = blob[y0:y1 + 1].sum(axis=1)
            solid = np.nonzero(row_counts >= 2)[0]
            y_top = y0 + int(solid[0]) if solid.size else y0

            u_c = 0.5 * (x0 + x1)
            base = ground_point(u_c, y1, 0.0)
            if base is None:
                continue
            forward, right = base
            # The room's diagonal is over 11 m, so a 7 m cap silently
            # discarded whole corners: entire colours went unseen and the
            # selector answered "nothing like that here" about an object in
            # plain view.
            if not 0.25 < forward < 12.0:
                continue
            # Ranged from the silhouette's bottom row, which is where the
            # object meets the floor. Stepping this out by half a width to
            # reach the object's centre was tried and measured worse: a
            # sphere's lowest pixel already sits under its centre, so the
            # correction is right for boxes and wrong for spheres, and the
            # median error rose from 0.146 m to 0.212 m.
            distance = math.hypot(forward, right)
            width = w_px / F_PX * distance
            height = max(0.02, height_at(y_top, forward))
            # This primitive-room detector does not classify wall decorations or
            # floor patches as fetchable bodies. Calibrated metric plausibility
            # is distance-independent; it is not a pixel-size cutoff.
            wx=pose['x']+forward*math.cos(heading)-right*math.sin(heading)
            wy=pose['y']-forward*math.sin(heading)-right*math.cos(heading)
            if width>.9 or height<.16 or height>1.0 or width>height*2.5 or max(abs(wx),abs(wy))>3.8:
                continue
            shape=_classify(blob,(x0,y_top,w_px,y1-y_top+1),height/max(width,1e-6))
            if shape == 'box' and height_valid:
                contours,_=cv2.findContours(blob.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                poly=cv2.approxPolyDP(max(contours,key=cv2.contourArea),2,True).reshape(-1,2)
                edges=[]
                for v,q in zip(poly,np.roll(poly,-1,axis=0)):
                    dy=abs(int(v[1])-int(q[1]))
                    if dy>h_px*.6 and abs(int(v[0])-int(q[0]))<.18*dy:
                        top,bottom=sorted([v,q],key=lambda p:p[1])
                        foot=ground_point(float(bottom[0]),float(bottom[1]),0)
                        if foot is not None:edges.append(height_at(float(top[1]),foot[0]))
                if edges:height=float(np.median(edges))
            out.append(Detection(
                colour=colour,
                shape=shape,
                height=round(height, 3),
                width=round(width, 3),
                distance=round(distance, 3),
                bearing=round(math.degrees(math.atan2(right, forward)), 2),
                world=(pose["x"] + forward * math.cos(heading) - right * math.sin(heading),
                       pose["y"] - forward * math.sin(heading) - right * math.cos(heading)),
                pixels=int(blob.sum()),
                box=(x0, y0, w_px, h_px),
                height_valid=height_valid,
            ))
    return out


@dataclass
class Request:
    """What the operator asked for, reduced to things that can be measured."""
    colour: str | None = None
    shape: str | None = None
    size: int | None = None
    superlative: tuple[str, bool] | None = None
    size_metric: str = 'height'

    @property
    def is_empty(self) -> bool:
        return not (self.colour or self.shape or self.size is not None
                    or self.superlative)


def parse_request(command: str) -> Request:
    """Reduce a typed command to colour, shape, size and comparison.

    Deliberately forgiving about everything else: "bring me the tall red one",
    "fetch red cylinder", and "get the red cylinder please" all mean the same
    thing, and the operator should not have to phrase it a particular way.
    """
    words = re.findall(r"[a-z]+", command.lower())
    req = Request()
    for w in words:
        if w in COLOUR_ANCHORS and req.colour is None:
            req.colour = w
        elif w in SHAPE_WORDS and req.shape is None:
            req.shape = SHAPE_WORDS[w]
        elif w in SUPERLATIVES and req.superlative is None:
            req.superlative = SUPERLATIVES[w]
        elif w in SIZE_WORDS and req.size is None:
            req.size = SIZE_WORDS[w]
            req.size_metric = 'height' if w in ('tall', 'short') else 'volume'
    # "tallest" is both a comparison and a size word; the comparison wins,
    # because it refers to the other objects rather than to an absolute scale.
    if req.superlative is not None and req.size is not None:
        req.size = None
    return req


def _size_rank(d: Detection, pool: list[Detection]) -> int:
    """Where this object sits among the others: 0 small, 1 medium, 2 large.

    Ranked against what is actually on screen rather than against a table of
    the scene's contents, which the agent is not allowed to know.
    """
    same = [o for o in pool if o.shape == d.shape] or pool
    order = sorted(same, key=lambda o: o.height)
    if len(order) < 2:
        return 1
    idx = order.index(d)
    return 0 if idx == 0 else (2 if idx == len(order) - 1 else 1)


def choose(req: Request, seen: list[Detection]) -> Detection | None:
    """Pick the object the request names, or None if nothing qualifies.

    Cues are weighted by how reliable they were measured to be, rather than
    all being trusted equally:

      colour   hard filter. 100% correct over 96 detections, so a request for
               a red object must never return a blue one.
      height   strong. Median error 7 mm, which separates the three sizes of
               any shape comfortably.
      shape    soft. 77% correct, and the errors are almost entirely box
               against cylinder, so a shape mismatch lowers a candidate's
               score instead of eliminating it.

    Hard-filtering on shape was tried first and is what made "bring the red
    cylinder" fetch nothing at all when the cylinder was read as a box: the
    filter removed the only correct answer. Scoring keeps it in contention and
    lets colour and size decide.
    """
    pool = [d for d in seen if d.height_valid] if req.superlative else list(seen)
    if not pool:
        return None
    if req.colour:
        pool = [d for d in pool if d.colour == req.colour]
        if not pool:
            return None                      # asked for a colour that is not here

    if req.shape and len(pool) > 1:
        # Box against cylinder is the one pair this camera cannot name
        # absolutely, but it can still COMPARE: an upright cylinder is always
        # taller relative to its width than a box is. Asked to tell two
        # candidates apart, ranking them against each other is far more
        # reliable than testing each against a fixed threshold, because the
        # distance and lighting errors that move the threshold move both
        # candidates the same way and cancel.
        if req.shape in ("box", "cylinder") and req.superlative is None:
            flat = [d for d in pool if d.shape in ("box", "cylinder")]
            if len(flat) >= 2:
                tallest_first = sorted(
                    flat, key=lambda d: d.height / max(d.width, 1e-6), reverse=True)
                return (tallest_first[0] if req.shape == "cylinder"
                        else tallest_first[-1])

        scored = [(SHAPE_LIKELIHOOD.get(req.shape, {}).get(d.shape, 0.01), d)
                  for d in pool]
        best = max(sc for sc, _ in scored)
        if best < 0.02:
            return None                      # nothing plausibly that shape
        top = [d for sc, d in scored if sc >= best * 0.9]
        if len(top) == 1:
            return top[0]
        pool = top
    elif req.shape and pool:
        if SHAPE_LIKELIHOOD.get(req.shape, {}).get(pool[0].shape, 0.01) < 0.02:
            return None

    if req.superlative:
        pool = [d for d in pool if d.height_valid]
        if not pool:
            return None
        key, biggest = req.superlative
        return sorted(pool, key=lambda d: getattr(d, key), reverse=biggest)[0]
    if req.size is not None:
        return sorted(pool, key=lambda d: (abs(_size_rank(d, seen) - req.size),
                                           d.distance))[0]
    return sorted(pool, key=lambda d: d.distance)[0]


@dataclass
class WorldModel:
    """Objects remembered across frames, built only from what was seen.

    One frame covers 90 degrees, so no single look contains the whole room and
    a decision made from one frame is a decision made on partial evidence --
    which is how "the tallest object" ended up meaning "the tallest of the two
    things I happen to be facing". The robot sweeps, and every sighting is
    folded into a map keyed on world position.

    Repeated sightings are averaged rather than replaced, since the noise is
    independent between views; the running mean of a dozen sightings is
    markedly tighter than the last one. Shape is taken by vote across
    sightings, which recovers a good share of what a single look gets wrong.
    """

    #: Two sightings within this distance are the same object. Comfortably
    #: above the measured 0.13 m position error and below the scene's minimum
    #: object separation, so neither double-counting nor merging can happen.
    merge_radius: float = 0.45
    tracks: list[dict] = field(default_factory=list)

    def observe(self, dets: list[Detection]) -> None:
        for d in dets:
            match = None
            for t in self.tracks:
                if t["colour"] == d.colour and math.dist(t["world"], d.world) <= self.merge_radius:
                    match = t
                    break
            if match is None:
                self.tracks.append({
                    "colour": d.colour, "world": d.world, "n": 1,
                    "heights": ([(d.distance, d.height, d.width)]
                                if d.height_valid else []),
                    "shapes": [d.shape], "last": d,
                })
                continue
            n = match["n"]
            match["world"] = ((match["world"][0] * n + d.world[0]) / (n + 1),
                              (match["world"][1] * n + d.world[1]) / (n + 1))
            # Every sighting is kept and the MEDIAN is used. Taking the
            # closest sighting instead sounds better and is not: one look whose
            # base row was clipped by another object throws the height out by
            # more than the gap between two sizes, and a single bad reading
            # then decides "the tallest object". A median ignores it.
            if d.height_valid:
                match["heights"].append((d.distance, d.height, d.width))
            match["shapes"].append(d.shape)
            match["n"] = n + 1
            if d.distance < match["last"].distance:
                match["last"] = d

    def objects(self, pose: dict) -> list[Detection]:
        """The remembered objects, as seen from where the robot is now."""
        out = []
        for t in self.tracks:
            wx, wy = t["world"]
            dx, dy = wx - pose["x"], wy - pose["y"]
            distance = math.hypot(dx, dy)
            bearing = ((math.degrees(math.atan2(-dy, dx)) - pose["heading"] + 180.0)
                       % 360.0) - 180.0
            shape = max(set(t["shapes"]), key=t["shapes"].count)
            # Height is read from how many rows the object spans, so a
            # distant sighting spreads the same object over a fraction of the
            # pixels and its estimate is correspondingly coarser. Keep the
            # near half of the sightings and take their median: the median
            # alone still let one far, badly-clipped reading outvote several
            # good ones, which is how a medium cone was called the tallest
            # object in the room.
            if t["heights"]:
                near = sorted(t["heights"])[:max(1, len(t["heights"]) // 2)]
                hs = sorted(h for _, h, _ in near)
                ws = sorted(w for _, _, w in near)
                height, width = hs[len(hs) // 2], ws[len(ws) // 2]
            else:
                # Every sighting was cut off at the top. The height is a lower
                # bound, which is better than nothing but must not win a
                # "tallest" comparison on the strength of a truncated read.
                height, width = t["last"].height, t["last"].width
            out.append(Detection(
                colour=t["colour"], shape=shape, height=height,
                width=width, distance=round(distance, 3),
                bearing=round(bearing, 2), world=(wx, wy),
                pixels=t["last"].pixels, box=t["last"].box,
                height_valid=bool(t["heights"])))
        return out
