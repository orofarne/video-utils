<!-- meshfox:canvas -->
# Blur licence plates in a video
<!-- meshfox:node id="root" -->

<!-- meshfox:comment -->
> This is a [meshfox](https://meshfox.orofarne.net/) document — open it
> with `meshfox view` (or `meshfox tui`) for the interactive canvas.
> This note is only visible here, in a plain Markdown viewer.
<!-- /meshfox:comment -->

Finds every car licence plate in a video and blurs it (or pixelates it, or
covers it with a black box). Detection is a small YOLOv9 model run through ONNX
Runtime — on the GPU/Neural Engine through CoreML on a Mac, CUDA when
available, otherwise on the CPU. This canvas is the whole tool: the
environment, the code, its tests and both pipeline steps live in the nodes
below. It needs `ffmpeg` on the `PATH`; the Python packages and the model
(about 30 MB, downloaded from GitHub on the first run) are fetched by the
first run itself.

```button name="main" default deps="render/render"
▶ Blur the plates
```

The pipeline is **Detect → Render**. *Detect* runs the model on **every**
frame and writes the boxes to `WORK_DIR` (default: this canvas's temporary
directory); *Render* turns them into regions, blurs them and encodes the
result. Re-rendering with another `BLUR_STYLE`, `PADDING` or `HOLD` does not
detect again, only a new video, model or `CONFIDENCE` does.

Detection is the slow part. At 720×1280 the default model takes about 12 ms a
frame with CoreML (a 45 s, 1339-frame clip: ≈ 30 s to detect and ≈ 30 s to
render on an Apple-silicon Mac) and about 135 ms on a CPU, so a CPU-only
machine needs `MODEL=fast` (≈ 40 ms) for anything long.

Because a missed plate is worse than a needless blur, the regions are made
generous:

- **`HOLD`** — a plate seen in a frame is also blurred in the `HOLD` frames
  before and after it. The detector most often misses a plate in the first and
  last frames, when it is half out of the picture or smeared by motion; looking
  at both directions is possible because the whole video is already analysed.
- **`PADDING`** — every box grows by this % of its size on each side, so the
  plate's frame and a slightly late box still disappear.
- Overlapping boxes in a frame are merged into one.

Try `BLUR_STYLE=outline` first on a new kind of footage: it only draws green
frames on the video, so you can see what is found (and what is not) before the
real render. `CONFIDENCE` lower than the default finds more, at the price of
more false frames.

Non-interactively:

```console
meshfox README.canvas.md run main --set VIDEO=in.mov --set OUTPUT=out/in.mp4 \
  --set BLUR_STYLE=blur
```

**Limits.** The model was exercised on close, large plates of a handheld video
(about 360 px wide in a 720 px frame). It is an ordinary detector: very small,
far-away, strongly tilted or night-time plates may be missed — check the result
(`outline` is for that) before publishing anything. HDR video is re-encoded as
plain 8-bit SDR, audio is copied unchanged.

<!-- meshfox:var name="VIDEO" type="file" ext="mp4,mkv,mov,webm,m4v" prompt="Video to process" -->
<!-- meshfox:var name="OUTPUT" type="file" mode="save" ext="mp4" prompt="Where to write the result (.mp4)" -->
<!-- meshfox:var name="BLUR_STYLE" type="select" choices="blur,pixelate,black,outline" default="blur" prompt="What to do with a plate: blur, pixelate, cover with a black box, or outline (a preview: green frames, nothing hidden)" -->
<!-- meshfox:var name="MODEL" type="select" choices="accurate,fast" default="accurate" prompt="Detector: accurate (YOLOv9-s, 608 px) or fast (YOLOv9-t, 640 px; about 3x faster on a CPU)" -->
<!-- meshfox:var name="CONFIDENCE" default="0.25" prompt="Detection threshold, 0.01-0.99 (lower finds more plates and more false ones)" -->
<!-- meshfox:var name="PADDING" default="15" prompt="Grow every box by this % of its size on each side (0-100)" -->
<!-- meshfox:var name="HOLD" default="4" prompt="Also blur a plate in this many frames before and after it is detected (0-60)" -->
<!-- meshfox:var name="WORK_DIR" type="dir" default="@tmp/work" prompt="Where to keep the detected boxes; default: this canvas's temporary directory" -->
<!-- meshfox:var name="PYTHON" from="env/setup" -->
<!-- meshfox:var name="BOXES" type="file" mode="save" from="paths/paths" -->

## Pipeline
<!-- meshfox:node id="pipeline" type="group" fold="false" -->

### Detect
<!-- meshfox:node id="detect" x=0 y=0 w=360 h=60 -->

Decodes the video with OpenCV (which applies a rotation flag, so the frames
are upright) and runs the plate detector on every frame. The result is
`BOXES`: the frame size, the frame rate and, for each frame, a list of
`[x1, y1, x2, y2, confidence]` in pixels. The ONNX Runtime provider is CUDA if
present, else CoreML, else the CPU.

```python name="detect-defs" extends="lib/core-defs"
MODELS = {
    "accurate": "yolo-v9-s-608-license-plate-end2end",
    "fast": "yolo-v9-t-640-license-plate-end2end",
}


def parse_confidence(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"CONFIDENCE must be a number between 0.01 and 0.99, got {text!r}")
    if not 0.01 <= value <= 0.99:
        raise SystemExit(f"CONFIDENCE must be between 0.01 and 0.99, got {value}")
    return value


def pick_providers(available: list[str]) -> list[str]:
    """The best ONNX Runtime provider that is installed, with the CPU behind it."""
    for name in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
        if name in available:
            return [name, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def quiet_empty_frame_warnings() -> None:
    """With CoreML, a frame with no plate at all makes ONNX Runtime complain
    that an output "has a dynamic shape ... zero elements". The detector turns
    that into "no detections", which is the right answer (checked frame by
    frame against the CPU provider), but logs it twice per frame. Hide exactly
    that Python warning (the matching ORT stderr line is silenced in `detect`);
    every other warning and error stays visible."""
    import logging

    class Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "has a dynamic shape" not in record.getMessage()

    for name in list(logging.root.manager.loggerDict):   # a logger's filter is not applied to its children
        if name.startswith("open_image_models"):
            logging.getLogger(name).addFilter(Drop())


def detect(video: str, boxes_path: str, model: str, confidence: float) -> None:
    import cv2
    import onnxruntime as ort
    from open_image_models import create_detector

    if model not in MODELS:
        raise SystemExit(f"MODEL must be one of {', '.join(MODELS)}, got {model!r}")
    ort.set_default_logger_severity(4)   # before the first inference: ORT's own stderr lines
    detector = create_detector(MODELS[model], conf_thresh=confidence,
                               providers=pick_providers(ort.get_available_providers()))
    quiet_empty_frame_warnings()   # after create_detector: that is what creates the loggers
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[list[list[float]]] = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        frames.append([[b.bounding_box.x1, b.bounding_box.y1, b.bounding_box.x2, b.bounding_box.y2,
                        round(float(b.confidence), 3)] for b in detector.predict(frame)])
        if len(frames) % 100 == 0:
            print(f"  {len(frames)} / {total} frames", flush=True)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames could be read from {video}")
    hits = sum(1 for f in frames if f)
    print(f"{len(frames)} frames, plates in {hits} of them")
    write_json(boxes_path, {"video": video, "model": model, "confidence": confidence,
                            "width": width, "height": height, "fps": fps, "frames": frames})
```

```python name="detect" extends="detect-defs" env="VIDEO,BOXES,MODEL,CONFIDENCE" inputs="$VIDEO" outputs="$BOXES"
E = os.environ
detect(E["VIDEO"], E["BOXES"], E["MODEL"], parse_confidence(E["CONFIDENCE"]))
```

Tests for the option parsing and the provider choice (the detection needs a
video and the model, so it was exercised end to end):

```python name="detect-test" extends="detect-defs"
import unittest


class DetectTests(unittest.TestCase):
    def test_confidence(self):
        self.assertEqual(parse_confidence("0.25"), 0.25)
        for bad in ("", "x", "0", "1", "-0.3"):
            with self.assertRaises(SystemExit):
                parse_confidence(bad)

    def test_provider_preference(self):
        cpu = "CPUExecutionProvider"
        self.assertEqual(pick_providers([cpu]), [cpu])
        self.assertEqual(pick_providers(["CoreMLExecutionProvider", cpu]), ["CoreMLExecutionProvider", cpu])
        self.assertEqual(pick_providers(["CoreMLExecutionProvider", "CUDAExecutionProvider", cpu])[0],
                         "CUDAExecutionProvider")
        self.assertEqual(pick_providers([]), [cpu])

    def test_every_model_name_is_a_registered_one(self):
        from open_image_models.detection.core.hub import DETECTION_MODELS
        for name in MODELS.values():
            self.assertIn(name, DETECTION_MODELS)


unittest.main(argv=["detect-test"], verbosity=2)
```

### Render
<!-- meshfox:node id="render" x=480 y=0 w=360 h=60 -->
<!-- meshfox:edge from="detect" style="solid" -->

`BOXES` → regions → blurred frames → H.264.

1. **Regions.** Each frame gets the boxes of the frames `HOLD` before and after
   it, every box grows by `PADDING`, and overlapping ones are merged.
2. **Hiding.** The region of the frame is replaced in place: a blur (the
   region is shrunk to about 12 px on its short side, blurred and scaled back
   up, so the characters are gone, not just soft, and it stays cheap), a
   pixelation into a few big cells, or black. Only pixels
   inside the region are read, so nothing leaks in from the surroundings.
3. **Encoding.** The frames go to `ffmpeg` through a pipe (H.264, CRF 18,
   `yuv420p`) while the original audio track is copied unchanged.

The frame count of the render must equal the detection's, otherwise it stops
instead of guessing which frame a box belongs to.

```python name="render-defs" extends="lib/core-defs"
import subprocess

STYLES = ("blur", "pixelate", "black", "outline")
BLUR_SMALL_SIDE = 12   # px: a plate's short side while it is being blurred


def parse_padding(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"PADDING must be a number between 0 and 100, got {text!r}")
    if not 0 <= value <= 100:
        raise SystemExit(f"PADDING must be between 0 and 100, got {value}")
    return value


def parse_hold(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(f"HOLD must be a whole number of frames between 0 and 60, got {text!r}")
    if not 0 <= value <= 60:
        raise SystemExit(f"HOLD must be between 0 and 60, got {value}")
    return value


def merge_overlapping(boxes: list[list[float]]) -> list[list[float]]:
    """Replace boxes that overlap by their union, until none overlap."""
    boxes = [list(b[:4]) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[list[float]] = []
        for b in boxes:
            for o in out:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[0], o[1] = min(o[0], b[0]), min(o[1], b[1])
                    o[2], o[3] = max(o[2], b[2]), max(o[3], b[3])
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def pad_box(box: list[float], pct: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow a box by `pct` % of its size on every side, clamp to the frame, round outwards."""
    x1, y1, x2, y2 = box[:4]
    dx, dy = (x2 - x1) * pct / 100, (y2 - y1) * pct / 100
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
            min(width, int(x2 + dx) + 1), min(height, int(y2 + dy) + 1))


def plan_regions(frames: list[list[list[float]]], width: int, height: int,
                 padding_pct: float, hold: int) -> list[list[tuple[int, int, int, int]]]:
    """For each frame the rectangles to hide: the boxes of the `hold` frames
    before and after it too, padded and merged. Empty rectangles are dropped."""
    n = len(frames)
    regions = []
    for i in range(n):
        near = [b for j in range(max(0, i - hold), min(n, i + hold + 1)) for b in frames[j]]
        padded = [pad_box(b, padding_pct, width, height) for b in near]
        regions.append([tuple(int(v) for v in r) for r in merge_overlapping([list(p) for p in padded])
                        if r[2] > r[0] and r[3] > r[1]])
    return regions


def hide(img, region: tuple[int, int, int, int], style: str) -> None:
    """Hide one rectangle of a BGR image, in place. Reads only pixels inside it."""
    import cv2

    x1, y1, x2, y2 = region
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h < 1 or w < 1:
        return
    if style == "blur":
        # Shrink to ~12 px on the short side, blur that, scale back: the same look
        # as a huge Gaussian kernel (the characters are gone), at a hundredth of the cost.
        k = min(1.0, BLUR_SMALL_SIDE / min(w, h))
        small = cv2.resize(roi, (max(2, round(w * k)), max(2, round(h * k))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif style == "pixelate":
        cell = max(6, h // 3)
        small = cv2.resize(roi, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif style == "black":
        roi[:] = 0
    elif style == "outline":
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), max(2, h // 25))
    else:
        raise SystemExit(f"BLUR_STYLE must be one of {', '.join(STYLES)}, got {style!r}")


def ffmpeg_command(video: str, output: str, width: int, height: int, fps: float) -> list[str]:
    """Raw BGR frames on stdin, the audio (if any) copied from the original."""
    return ["ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
            "-i", video, "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest", "-movflags", "+faststart", output]


def render(video: str, boxes_path: str, output: str, *, style: str, padding_pct: float, hold: int) -> None:
    import cv2

    if style not in STYLES:
        raise SystemExit(f"BLUR_STYLE must be one of {', '.join(STYLES)}, got {style!r}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on the PATH")
    data = read_json(boxes_path)
    width, height, fps = data["width"], data["height"], data["fps"]
    regions = plan_regions(data["frames"], width, height, padding_pct, hold)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(ffmpeg_command(video, output, width, height, fps), stdin=subprocess.PIPE)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise SystemExit(f"frame size changed to {frame.shape[1]}x{frame.shape[0]}; run Detect again")
            if count >= len(regions):
                raise SystemExit("the video has more frames than were detected; run Detect again")
            for region in regions[count]:
                hide(frame, region, style)
            proc.stdin.write(frame.tobytes())
            count += 1
        if count != len(regions):
            raise SystemExit(f"read {count} frames but {len(regions)} were detected; run Detect again")
    except BrokenPipeError:
        pass   # ffmpeg died; its return code is reported below
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
    if code:
        raise SystemExit(f"ffmpeg failed (exit code {code})")
    print(f"{count} frames, {sum(1 for r in regions if r)} with something hidden")
```

```python name="render" extends="render-defs" env="VIDEO,BOXES,OUTPUT,BLUR_STYLE,PADDING,HOLD" inputs="$VIDEO,$BOXES" outputs="$OUTPUT"
import shutil

E = os.environ
render(E["VIDEO"], E["BOXES"], E["OUTPUT"], style=E["BLUR_STYLE"],
       padding_pct=parse_padding(E["PADDING"]), hold=parse_hold(E["HOLD"]))
print(f"rendered -> {E['OUTPUT']}")
```

Tests for the region planning, the hiding and the `ffmpeg` command (the
encoding itself is exercised end to end):

```python name="render-test" extends="render-defs"
import shutil
import unittest

import numpy as np


class RegionTests(unittest.TestCase):
    def test_options(self):
        self.assertEqual(parse_padding("15"), 15.0)
        self.assertEqual(parse_hold("4"), 4)
        for bad in ("", "x", "-1", "101"):
            with self.assertRaises(SystemExit):
                parse_padding(bad)
        for bad in ("", "1.5", "-1", "61"):
            with self.assertRaises(SystemExit):
                parse_hold(bad)

    def test_pad_grows_every_side_and_stays_in_the_frame(self):
        self.assertEqual(pad_box([100, 100, 200, 140], 10, 1000, 1000), (90, 96, 211, 145))
        self.assertEqual(pad_box([0, 0, 50, 50], 50, 60, 60), (0, 0, 60, 60))

    def test_merge_overlapping(self):
        merged = merge_overlapping([[0, 0, 10, 10], [5, 5, 20, 20], [100, 100, 110, 110]])
        self.assertEqual(sorted(merged), [[0, 0, 20, 20], [100, 100, 110, 110]])
        # a chain: the third touches only the union of the first two
        self.assertEqual(merge_overlapping([[0, 0, 10, 10], [20, 0, 30, 10], [8, 0, 22, 10]]), [[0, 0, 30, 10]])
        self.assertEqual(merge_overlapping([[0, 0, 5, 5], [5, 0, 10, 5]]), [[0, 0, 5, 5], [5, 0, 10, 5]])  # touching is not overlapping

    def test_hold_reaches_back_and_forward(self):
        box = [10, 10, 20, 20, 0.9]
        regions = plan_regions([[], [], [box], [], [], [], []], 100, 100, 0, hold=1)
        self.assertEqual([bool(r) for r in regions], [False, True, True, True, False, False, False])
        regions = plan_regions([[], [], [box], [], []], 100, 100, 0, hold=0)
        self.assertEqual([bool(r) for r in regions], [False, False, True, False, False])

    def test_moving_plate_is_covered_in_every_frame_it_sweeps(self):
        frames = [[[10 + 20 * i, 10, 30 + 20 * i, 30, 0.9]] for i in range(3)]
        (frame_one,) = plan_regions(frames, 200, 100, 0, hold=2)[1]
        self.assertEqual((frame_one[0], frame_one[2]), (10, 71))   # the union of all three positions

    def test_no_boxes_no_regions(self):
        self.assertEqual(plan_regions([[], []], 100, 100, 15, 4), [[], []])


class HideTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.img = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)   # noise: every detail visible
        self.region = (40, 30, 160, 90)

    def hidden(self, style):
        img = self.img.copy()
        hide(img, self.region, style)
        return img

    def test_nothing_outside_the_region_changes(self):
        x1, y1, x2, y2 = self.region
        for style in ("blur", "pixelate", "black"):
            out = self.hidden(style)
            mask = np.ones(out.shape[:2], bool)
            mask[y1:y2, x1:x2] = False
            self.assertTrue((out[mask] == self.img[mask]).all(), style)
            self.assertFalse((out[~mask] == self.img[~mask]).all(), style)

    def test_detail_is_destroyed_not_just_softened(self):
        x1, y1, x2, y2 = self.region
        before = self.img[y1:y2, x1:x2].astype(float).std()
        for style in ("blur", "pixelate"):
            self.assertLess(self.hidden(style)[y1:y2, x1:x2].astype(float).std(), before / 4, style)
        self.assertEqual(self.hidden("black")[y1:y2, x1:x2].max(), 0)

    def test_outline_hides_nothing_inside(self):
        x1, y1, x2, y2 = self.region
        out = self.hidden("outline")
        self.assertTrue((out[y1 + 10:y2 - 10, x1 + 10:x2 - 10] == self.img[y1 + 10:y2 - 10, x1 + 10:x2 - 10]).all())
        self.assertFalse((out == self.img).all())

    def test_unknown_style_is_refused(self):
        with self.assertRaises(SystemExit):
            hide(self.img.copy(), self.region, "sparkle")


class CommandTests(unittest.TestCase):
    def test_ffmpeg_command(self):
        cmd = ffmpeg_command("/v/a b (1).mov", "/o/out.mp4", 720, 1280, 29.97003)
        self.assertEqual(cmd[cmd.index("-s") + 1], "720x1280")
        self.assertEqual(cmd[cmd.index("-r") + 1], "29.970030")
        self.assertEqual(cmd[cmd.index("-i", cmd.index("-i") + 1) + 1], "/v/a b (1).mov")   # one argv item
        self.assertIn("1:a?", cmd)        # audio is optional
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")
        self.assertEqual(cmd[-1], "/o/out.mp4")


unittest.main(argv=["render-test"], verbosity=2)
```

### Result (OUTPUT)
<!-- meshfox:node id="result-output" type="file" x=960 y=0 w=360 -->
<!-- meshfox:edge from="render" -->

[OUTPUT]($OUTPUT)

The finished video, written by **Render** where you asked for it.

## Tech details
<!-- meshfox:node id="tech-details" -->

The environment, shared code, path handling and tests.

### Python environment
<!-- meshfox:node id="env" -->

`open-image-models` brings the YOLOv9 plate detectors (MIT) and downloads
the chosen model on first use; ONNX Runtime executes it (the `[onnx]` extra);
OpenCV decodes the frames and draws; NumPy is its array type. On a Mac the
ONNX Runtime wheel already contains the CoreML provider; on a Linux machine with an
NVIDIA GPU install `onnxruntime-gpu` into this environment instead for CUDA.

The block is a `pyproject.toml`: `@python_venv` installs it into an isolated
venv under `.meshfox/` and publishes its interpreter as `$PYTHON`.
`requires-python` stops at 3.13 because ONNX Runtime has no wheels for newer
interpreters yet, so `python3` on the `PATH` must be 3.10-3.13 (for example
`brew install python@3.13`).

```toml name="setup" interpreter="@python_venv"
[project]
name = "blur-plates-env"
version = "0.0.0"
requires-python = ">=3.10,<3.14"
dependencies = [
  "open-image-models[onnx]",
  "opencv-python-headless",
  "numpy",
]
```

### Shared helpers
<!-- meshfox:node id="lib" -->

Reading the environment, handing values on to meshfox through
`MESHFOX_VARS_OUT`, JSON. This block is only borrowed from: a step that
writes `extends="lib/core-defs"` runs with this text in front of its own, so
what is defined here is simply there for it. It begins with
`from __future__ import annotations`, so it always comes first in an
`extends=` list.

```python name="core-defs" interpreter="$PYTHON -u"
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def vars_out(**kv: str) -> None:
    """Publish values to meshfox (`meshfox:var ... from=`), or print them."""
    target = os.environ.get("MESHFOX_VARS_OUT")
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(lines)
    sys.stdout.write(lines)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
```

### Paths
<!-- meshfox:node id="paths" -->

Derives the boxes file name from `OUTPUT` and `WORK_DIR` (never next to the
result) and publishes it as `BOXES`.

```python name="paths-defs" extends="lib/core-defs"
import hashlib


def derive_paths(output: str, work_dir: str) -> dict[str, str]:
    """Named after OUTPUT's stem plus a short hash of its full path, so two
    outputs called `clip.mp4` do not share state."""
    out = Path(output).expanduser().resolve()
    tag = hashlib.sha1(str(out).encode("utf-8")).hexdigest()[:8]
    return {"BOXES": str(Path(work_dir).expanduser().resolve() / f"{out.stem}-{tag}.boxes.json")}
```

```python name="paths" extends="paths-defs" env="OUTPUT,WORK_DIR"
E = os.environ
Path(E["WORK_DIR"]).mkdir(parents=True, exist_ok=True)
vars_out(**derive_paths(E["OUTPUT"], E["WORK_DIR"]))
```

```python name="paths-test" extends="paths-defs"
import unittest


class PathsTests(unittest.TestCase):
    def test_boxes_path_derives_from_output(self):
        p = derive_paths("/o/clip (1).mp4", "/w")
        self.assertRegex(p["BOXES"], r"^/w/clip \(1\)-[0-9a-f]{8}\.boxes\.json$")

    def test_same_stem_in_different_folders_does_not_collide(self):
        self.assertNotEqual(derive_paths("/a/clip.mp4", "/w"), derive_paths("/b/clip.mp4", "/w"))


unittest.main(argv=["paths-test"], verbosity=2)
```

### Tests
<!-- meshfox:node id="tests" -->

Every function has unit tests next to it (`paths-test`, `detect-test`,
`render-test`). Each runs on its own from its node; this block runs all of
them. They need no video: the detection and the encoding were exercised end to
end.

```bash name="tests" default deps="paths/paths-test,detect/detect-test,render/render-test"
echo "all blur-plates tests passed"
```
