#!/usr/bin/env python3

"""A small browser tool for fixing the esophagus/aorta split by eye.

    python relabel_gui.py --source_dir data/segthor_part1

then open http://localhost:8000 .  Click a blob to give it the right label,
step through the slices, press Save, and it writes your decisions into
corrections.py -- which apply_corrections.py then turns into a corrected
GT_corrected.nii.gz.  Nothing is written until you press Save.

Only the standard library plus what the project already needs (numpy, nibabel,
scipy, PIL) is used, so there is nothing extra to install.

Clicking assigns a whole connected blob rather than painting voxel by voxel.
That matches how the reconstruction actually fails: it hands a whole tube
cross-section to the wrong organ.  Where instead the *border* between two
touching tubes is in the wrong place, painting it by hand is not worth the
effort -- mark the slice "unsure" and let MarginalCrossEntropy handle it.

The view is the radiological convention: anterior at the top, the patient's
left on the right of the image.  The descending aorta is the big round vessel
just left of the spine; the esophagus is the smaller, squashier one in front
of it.
"""

import argparse
import base64
import io
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ESOPHAGUS, HEART, TRACHEA, AORTA, AMBIGUOUS = 1, 2, 3, 4, 5
PAIR = (ESOPHAGUS, AORTA)
REGION = PAIR + (AMBIGUOUS,)
STRUCT_2D = ndi.generate_binary_structure(2, 1)

COLOURS = {ESOPHAGUS: (235, 64, 64), HEART: (54, 190, 90),
           TRACHEA: (224, 208, 52), AORTA: (72, 128, 255), AMBIGUOUS: (222, 122, 222)}
MARGIN, ZOOM, WINDOW = 28, 3, (-160.0, 240.0)   # crop margin px, upscale, HU window

STATE: dict = {}          # filled in by main()


# --------------------------------------------------------------------- data --
class Patient:
    """One patient's volume, its crop box, and the slices worth looking at."""

    def __init__(self, folder: Path, gt_name: str):
        self.pid = folder.name
        self.folder = folder
        nii = nib.load(str(folder / gt_name))
        self.gt = np.asarray(nii.dataobj).astype(np.uint8)
        self.spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
        ct = np.asarray(nib.load(str(folder / f"{self.pid}.nii.gz")).dataobj).astype(np.float32)
        lo, hi = WINDOW
        self.ct = np.clip((ct - lo) / (hi - lo), 0, 1)

        region = np.isin(self.gt, REGION)
        zs = np.where(region.any((0, 1)))[0]
        self.z0, self.z1 = int(zs.min()), int(zs.max())

        # One crop box for the whole patient, so the picture does not jump
        # around as you step through the slices.
        ii, jj = np.where(region.any(2))
        self.box = (max(0, int(ii.min()) - MARGIN), min(self.gt.shape[0], int(ii.max()) + MARGIN),
                    max(0, int(jj.min()) - MARGIN), min(self.gt.shape[1], int(jj.max()) + MARGIN))
        self.suspect = self._suspect_slices()

    def _suspect_slices(self) -> list[dict]:
        """Cheap checks for the places the reconstruction usually goes wrong."""
        out: list[dict] = []
        eso, aor = self.gt == ESOPHAGUS, self.gt == AORTA
        za = np.where(aor.any((0, 1)))[0]
        for z in range(self.z0, self.z1 + 1):
            plane = np.isin(self.gt[:, :, z], REGION)
            if not plane.any():
                continue
            why = []
            if za.size and z in (int(za.min()), int(za.min()) + 1, int(za.max()), int(za.max()) - 1):
                why.append("end of the aorta tube")
            dt = ndi.distance_transform_edt(plane, sampling=self.spacing[:2])
            re_ = dt[eso[:, :, z]].max() if eso[:, :, z].any() else 0
            ra = dt[aor[:, :, z]].max() if aor[:, :, z].any() else 0
            if re_ and ra and re_ > ra:
                why.append("esophagus wider than the aorta")
            elif re_ > 11:
                why.append("unusually wide esophagus")
            _, k = ndi.label(plane, STRUCT_2D)
            if k == 1 and eso[:, :, z].any() and aor[:, :, z].any():
                why.append("the two organs are fused here")
            if why:
                out.append({"z": int(z), "why": "; ".join(why)})
        return out

    def slice_labels(self, z: int) -> np.ndarray:
        """This slice's labels with the pending edits applied, in order."""
        plane = self.gt[:, :, z].copy()
        ops = STATE["edits"].get(self.pid, {}).get(z, [])
        # Actions first, then voxel edits -- the same order apply_corrections.py
        # replays them in, so what you see here is what you will get.
        for op in ops:
            if op["kind"] == "action":
                apply_action(plane, op["action"], op["target"])
        for op in ops:
            if op["kind"] == "paint":
                apply_paint(plane, op["voxels"], op["label"])
        return plane

    def render(self, z: int) -> tuple[str, list[dict]]:
        i0, i1, j0, j1 = self.box
        labels = self.slice_labels(z)
        grey = (self.ct[i0:i1, j0:j1, z] * 255).astype(np.uint8)
        rgb = np.stack([grey] * 3, -1).astype(np.float32)
        crop = labels[i0:i1, j0:j1]
        for k, colour in COLOURS.items():
            m = crop == k
            if m.any():   # keep the CT texture visible through the overlay
                rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(colour, np.float32)

        # rows = j (anterior at top), cols = i (patient's left on the right)
        img = Image.fromarray(np.transpose(rgb, (1, 0, 2)).astype(np.uint8))
        img = img.resize((img.width * ZOOM, img.height * ZOOM), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, "PNG")

        blobs = []
        lab, k = ndi.label(np.isin(labels, REGION), STRUCT_2D)
        for i in range(1, k + 1):
            m = lab == i
            vals, counts = np.unique(labels[m], return_counts=True)
            ci, cj = ndi.center_of_mass(m)
            blobs.append({"size": int(m.sum()), "label": int(vals[np.argmax(counts)]),
                          "i": int(round(ci)), "j": int(round(cj))})
        blobs.sort(key=lambda b: -b["size"])
        return base64.b64encode(buf.getvalue()).decode(), blobs


def apply_paint(plane: np.ndarray, voxels, label: int) -> None:
    """Set individual voxels.  Only ever inside the esophagus/aorta region, so a
    stray stroke cannot eat into the heart, the trachea or the background."""
    if not voxels:
        return
    idx = np.asarray(voxels, dtype=np.intp)
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < plane.shape[0])
          & (idx[:, 1] >= 0) & (idx[:, 1] < plane.shape[1]))
    idx = idx[ok]
    if not idx.size:
        return
    ii, jj = idx[:, 0], idx[:, 1]
    inside = np.isin(plane[ii, jj], REGION)
    plane[ii[inside], jj[inside]] = label


def lasso_voxels(shape: tuple[int, int], points) -> list[tuple[int, int]]:
    """The voxels a freehand outline encloses, as (i, j) pairs.

    PIL indexes images (x, y) while the label array is [i, j], so the polygon
    goes in as (j, i) and the rasterised mask comes back the right way round."""
    if len(points) < 3:
        return []
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([(int(j), int(i)) for i, j in points], fill=1)
    ii, jj = np.nonzero(np.asarray(img, dtype=bool))
    return list(zip(ii.tolist(), jj.tolist()))


def in_region(plane: np.ndarray, voxels) -> list[tuple[int, int]]:
    """Keep only the voxels that are esophagus, aorta or unsure right now."""
    if not voxels:
        return []
    idx = np.asarray(voxels, dtype=np.intp)
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < plane.shape[0])
          & (idx[:, 1] >= 0) & (idx[:, 1] < plane.shape[1]))
    idx = idx[ok]
    if not idx.size:
        return []
    keep = np.isin(plane[idx[:, 0], idx[:, 1]], REGION)
    return [tuple(v) for v in idx[keep].tolist()]


def apply_action(plane: np.ndarray, action: str, target) -> None:
    """The same actions apply_corrections.py understands, on one slice."""
    region = np.isin(plane, REGION)
    if not region.any():
        return
    if target is None:
        sel = region
    else:
        lab, _ = ndi.label(region, STRUCT_2D)
        comp = int(lab[target[0], target[1]])
        if not comp:
            return
        sel = lab == comp

    match action:
        case "aorta":
            plane[sel] = AORTA
        case "esophagus":
            plane[sel] = ESOPHAGUS
        case "ambiguous" | "drop":
            plane[sel] = AMBIGUOUS
        case "swap":
            e, a = sel & (plane == ESOPHAGUS), sel & (plane == AORTA)
            plane[e], plane[a] = AORTA, ESOPHAGUS


# ---------------------------------------------------------------- saving ----
def ranges(zs: list[int]) -> str:
    """[26, 27, 28, 40] -> '26-28,40'"""
    zs = sorted(set(zs))
    out, start = [], zs[0]
    for a, b in zip(zs, zs[1:] + [None]):
        if b != a + 1:
            out.append(str(start) if start == a else f"{start}-{a}")
            start = b
    return ",".join(out)


def to_source() -> str:
    """Turn the pending blob/slice actions into the body of corrections.py.

    Freehand paint cannot be written as a (slice, action, target) triple, so it
    goes to corrections_paint.npz instead and apply_corrections.py lays it on
    top afterwards.  Keeping the two apart means the readable file stays
    readable: it says what you decided, not which pixels you touched."""
    lines = ["CORRECTIONS: dict[str, list[tuple]] = {"]
    for pid in sorted(STATE["edits"]):
        per_slice = STATE["edits"][pid]
        grouped: dict[tuple, list[int]] = {}
        for z, ops in sorted(per_slice.items()):
            for op in ops:
                if op["kind"] != "action":
                    continue
                target = tuple(op["target"]) if op["target"] else None
                grouped.setdefault((op["action"], target), []).append(z)
        if not grouped:
            continue
        lines.append(f'    "{pid}": [')
        for (action, target), zs in grouped.items():
            spec = f'"{ranges(zs)}"'
            tail = f', {target}' if target else ''
            lines.append(f'        ({spec}, "{action}"{tail}),')
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines) + "\n"


def paint_arrays() -> dict[str, np.ndarray]:
    """Every painted voxel as one (N, 4) int32 array per patient: z, i, j, label."""
    out: dict[str, np.ndarray] = {}
    for pid, per_slice in STATE["edits"].items():
        rows = [(z, i, j, op["label"])
                for z, ops in per_slice.items() for op in ops
                if op["kind"] == "paint" for i, j in op["voxels"]]
        if rows:
            out[pid] = np.asarray(rows, dtype=np.int32)
    return out


def save() -> str:
    path = Path("corrections.py")
    head = path.read_text().split("CORRECTIONS")[0]   # keep the docstring as written
    path.write_text(head + to_source())

    n_act = sum(1 for per in STATE["edits"].values() for ops in per.values()
                for op in ops if op["kind"] == "action")
    msg = f"{n_act} action(s) -> {path}"

    paint = paint_arrays()
    brush = Path("corrections_paint.npz")
    if paint:
        np.savez_compressed(brush, **paint)
        msg += f"; {sum(len(v) for v in paint.values())} painted voxel(s) -> {brush}"
    elif brush.exists():
        brush.unlink()
        msg += f"; removed {brush} (no paint left)"
    return msg + f"  [{len(STATE['edits'])} patient(s)]"


# ------------------------------------------------------------------ server --
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console quiet
        pass

    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}

        if url.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")

        if url.path == "/api/patients":
            return self._send(json.dumps([
                {"pid": p.pid, "z0": p.z0, "z1": p.z1, "suspect": p.suspect}
                for p in STATE["patients"]]).encode())

        if url.path == "/api/slice":
            p = STATE["by_id"][q["pid"]]
            z = int(q["z"])
            png, blobs = p.render(z)
            i0, _, j0, _ = p.box
            return self._send(json.dumps({
                "png": png, "blobs": blobs, "zoom": ZOOM, "i0": i0, "j0": j0,
                "edits": len(STATE["edits"].get(p.pid, {}).get(z, [])),
                "source": to_source()}).encode())

        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        url = urlparse(self.path)

        if url.path == "/api/edit":
            pid, z = body["pid"], int(body["z"])
            per = STATE["edits"].setdefault(pid, {}).setdefault(z, [])
            match body["op"]:
                case "add":
                    tgt = tuple(body["target"]) if body.get("target") else None
                    per.append({"kind": "action", "action": body["action"], "target": tgt})
                case "paint" | "lasso":
                    # A lasso becomes voxels here, so undo, saving and replay all
                    # treat it exactly like a brush stroke.
                    patient = STATE["by_id"][pid]
                    if body["op"] == "lasso":
                        vox = lasso_voxels(patient.gt.shape[:2], body["points"])
                    else:
                        vox = [tuple(v) for v in body["voxels"]]
                    # Drop anything outside the esophagus/aorta region now rather
                    # than at apply time, so the saved record holds only voxels
                    # that actually change something.
                    vox = in_region(patient.gt[:, :, z], vox)
                    if vox:   # a stroke entirely on the background records nothing
                        per.append({"kind": "paint", "label": int(body["label"]),
                                    "voxels": vox})
                case "undo":
                    if per:
                        per.pop()
                case "clear":
                    per.clear()
            if not per:
                STATE["edits"][pid].pop(z, None)
            if not STATE["edits"].get(pid):
                STATE["edits"].pop(pid, None)
            return self._send(json.dumps({"ok": True}).encode())

        if url.path == "/api/save":
            return self._send(json.dumps({"message": save()}).encode())

        self.send_error(404)


PAGE = r"""
<title>SegTHOR relabel</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c313a; --ink:#e7eaf0; --dim:#98a1b0; }
  body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; margin:0; }
  .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .side { width:290px; flex:none; max-height:94vh; overflow:auto; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim);
       margin:0 0 8px; font-weight:600; }
  select, button { background:#262b33; color:var(--ink); border:1px solid var(--line);
       border-radius:6px; padding:6px 10px; font:inherit; cursor:pointer; }
  button:hover { background:#2f3540; }
  button.on { background:#3b6ef5; border-color:#3b6ef5; }
  .row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
  #stage { position:relative; display:inline-block; }
  #img { display:block; image-rendering:pixelated; border-radius:6px; cursor:crosshair; }
  .tag { position:absolute; color:#8e97a6; font-size:11px; pointer-events:none; }
  .legend span { display:inline-flex; align-items:center; gap:5px; margin-right:12px; font-size:12px; }
  .sw { width:11px; height:11px; border-radius:3px; display:inline-block; }
  .sus { padding:5px 7px; border-radius:6px; cursor:pointer; margin-bottom:3px; font-size:12px; }
  .sus:hover { background:#262b33; }
  .sus b { color:#ffd479; }
  pre { background:#111318; border:1px solid var(--line); border-radius:8px; padding:10px;
        font-size:11.5px; overflow:auto; max-height:230px; color:#c8d0dc; }
  kbd { background:#2a2f38; border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:11px; }
  .hint { color:var(--dim); font-size:12px; }
</style>

<div class="wrap">
  <div class="side panel">
    <h2>Patient</h2>
    <select id="pid" style="width:100%"></select>

    <h2 style="margin-top:16px">Paint with</h2>
    <div class="row">
      <button data-mode="aorta" class="on">Aorta</button>
      <button data-mode="esophagus">Esophagus</button>
      <button data-mode="ambiguous">Unsure</button>
    </div>

    <h2>Tool</h2>
    <div class="row">
      <button data-tool="blob" class="on">Whole blob</button>
      <button data-tool="brush">Brush</button>
      <button data-tool="lasso">Lasso</button>
    </div>
    <div class="row" id="brushrow" style="display:none; align-items:center">
      <span class="hint">size</span>
      <input id="bsize" type="range" min="1" max="10" value="3" style="width:150px">
      <span id="bsizelab" class="hint">3 px</span>
    </div>

    <h2>Whole slice</h2>
    <div class="row">
      <button data-whole="swap">Swap the two</button>
      <button data-whole="aorta">All aorta</button>
      <button data-whole="esophagus">All esophagus</button>
      <button data-whole="ambiguous">All unsure</button>
    </div>
    <div class="row">
      <button id="undo">Undo</button>
      <button id="clear">Reset slice</button>
    </div>

    <h2 style="margin-top:16px">Worth a look</h2>
    <div id="suspect"></div>

    <h2 style="margin-top:16px">corrections.py</h2>
    <pre id="src">(nothing yet)</pre>
    <button id="save" style="width:100%">Save to corrections.py</button>
    <div id="msg" class="hint" style="margin-top:6px"></div>
  </div>

  <div class="panel">
    <div class="row" style="align-items:center">
      <button id="prev">&larr;</button>
      <input id="slider" type="range" style="width:340px">
      <button id="next">&rarr;</button>
      <span id="zlab" class="hint"></span>
    </div>
    <div id="stage">
      <img id="img" draggable="false">
      <canvas id="ghost" style="position:absolute;left:0;top:0;display:none;pointer-events:none"></canvas>
      <div class="tag" style="top:4px;left:50%;transform:translateX(-50%)">anterior</div>
      <div class="tag" style="bottom:4px;left:50%;transform:translateX(-50%)">posterior (spine)</div>
      <div class="tag" style="left:4px;top:50%;transform:translateY(-50%) rotate(-90deg)">patient right</div>
      <div class="tag" style="right:4px;top:50%;transform:translateY(-50%) rotate(90deg)">patient left</div>
    </div>
    <div class="legend" style="margin-top:10px">
      <span><i class="sw" style="background:#eb4040"></i>esophagus</span>
      <span><i class="sw" style="background:#4880ff"></i>aorta</span>
      <span><i class="sw" style="background:#36be5a"></i>heart</span>
      <span><i class="sw" style="background:#e0d034"></i>trachea</span>
      <span><i class="sw" style="background:#de7ade"></i>unsure</span>
    </div>
    <div class="hint" style="margin-top:8px">
      <kbd>&larr;</kbd><kbd>&rarr;</kbd> slice &nbsp; <kbd>1</kbd> aorta
      <kbd>2</kbd> esophagus <kbd>3</kbd> unsure &nbsp; <kbd>b</kbd> next tool
      <kbd>[</kbd><kbd>]</kbd> brush size &nbsp; <kbd>s</kbd> swap
      <kbd>u</kbd> undo &nbsp; <span id="blobinfo"></span>
    </div>
  </div>
</div>

<script>
let P = [], cur = null, z = 0, mode = "aorta", meta = null;
let tool = "blob", bsize = 3, painting = false, stroke = new Set(), ghostLast = null;
let lasso = [], lassoScreen = [];
const LABEL = {aorta: 4, esophagus: 1, ambiguous: 5};

const $ = s => document.querySelector(s);

async function boot() {
  P = await (await fetch("/api/patients")).json();
  $("#pid").innerHTML = P.map(p => `<option>${p.pid}</option>`).join("");
  pick(P[0]);
}

function pick(p) {
  cur = p; z = p.z0;
  $("#slider").min = p.z0; $("#slider").max = p.z1;
  $("#suspect").innerHTML = p.suspect.length
    ? p.suspect.map(s => `<div class="sus" data-z="${s.z}"><b>z ${s.z}</b> — ${s.why}</div>`).join("")
    : '<div class="hint">nothing flagged automatically</div>';
  draw();
}

async function draw() {
  $("#slider").value = z;
  $("#zlab").textContent = `slice ${z} of ${cur.z0}–${cur.z1}`;
  const r = await (await fetch(`/api/slice?pid=${cur.pid}&z=${z}`)).json();
  meta = r;
  $("#img").src = "data:image/png;base64," + r.png;
  $("#src").textContent = r.source.trim() || "(nothing yet)";
  $("#blobinfo").textContent = r.blobs.map(b =>
      `${b.size}px ${({1:"eso",4:"aorta",5:"unsure"})[b.label] || b.label}`).join(" · ");
}

async function edit(body) {
  await fetch("/api/edit", {method:"POST", headers:{"Content-Type":"application/json"},
                            body: JSON.stringify({pid: cur.pid, z, ...body})});
  draw();
}

// the picture is labels[i, j] transposed: screen x runs along i, screen y along j
function voxelAt(e) {
  const r = $("#img").getBoundingClientRect();
  return [meta.i0 + Math.floor((e.clientX - r.left) / meta.zoom),
          meta.j0 + Math.floor((e.clientY - r.top)  / meta.zoom)];
}

// a round brush of the chosen radius, in voxels
function disc(i0, j0) {
  const out = [], r = bsize - 1;
  for (let di = -r; di <= r; di++)
    for (let dj = -r; dj <= r; dj++)
      if (di*di + dj*dj <= r*r + 0.5) out.push([i0 + di, j0 + dj]);
  return out.length ? out : [[i0, j0]];
}

let lastVox = null;

// a Set keyed by "i,j": overlapping stamps along a stroke would otherwise send
// the same voxel dozens of times
function stamp(i, j) { for (const [a, b] of disc(i, j)) stroke.add(a + "," + b); }

function paintAt(e) {
  const [i, j] = voxelAt(e);
  // The mouse only reports a handful of positions during a quick drag, so walk
  // the line from the previous one -- otherwise a fast stroke comes out dotted.
  if (lastVox) {
    const [pi, pj] = lastVox;
    const n = Math.max(Math.abs(i - pi), Math.abs(j - pj));
    for (let s = 1; s <= n; s++)
      stamp(Math.round(pi + (i - pi) * s / n), Math.round(pj + (j - pj) * s / n));
  } else {
    stamp(i, j);
  }
  lastVox = [i, j];

  // draw locally so it feels immediate; the server redraws the truth on release
  const r = $("#img").getBoundingClientRect();
  const c = $("#ghost").getContext("2d");
  c.strokeStyle = c.fillStyle = {4:"#4880ff", 1:"#eb4040", 5:"#de7ade"}[LABEL[mode]];
  c.globalAlpha = 0.75;
  c.lineWidth = bsize * meta.zoom * 1.4;
  c.lineCap = c.lineJoin = "round";
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (ghostLast) { c.beginPath(); c.moveTo(ghostLast[0], ghostLast[1]); c.lineTo(x, y); c.stroke(); }
  else { c.beginPath(); c.arc(x, y, c.lineWidth / 2, 0, 7); c.fill(); }
  ghostLast = [x, y];
}

function armGhost() {
  const g = $("#ghost");
  g.width = $("#img").width; g.height = $("#img").height; g.style.display = "block";
  return g.getContext("2d");
}

function lassoAt(e) {
  const v = voxelAt(e);
  // one point per voxel stepped over, so the outline stays a sane size
  if (!lasso.length || lasso.at(-1)[0] !== v[0] || lasso.at(-1)[1] !== v[1]) lasso.push(v);
  const r = $("#img").getBoundingClientRect();
  lassoScreen.push([e.clientX - r.left, e.clientY - r.top]);

  const c = $("#ghost").getContext("2d");
  c.clearRect(0, 0, c.canvas.width, c.canvas.height);
  c.strokeStyle = c.fillStyle = {4:"#4880ff", 1:"#eb4040", 5:"#de7ade"}[LABEL[mode]];
  c.lineWidth = 2; c.setLineDash([5, 4]);
  c.beginPath();
  lassoScreen.forEach(([x, y], n) => n ? c.lineTo(x, y) : c.moveTo(x, y));
  c.globalAlpha = 1; c.stroke();
  c.globalAlpha = 0.25; c.closePath(); c.fill();
}

$("#img").onmousedown = e => {
  if (tool === "blob") { edit({op:"add", action: mode, target: voxelAt(e)}); return; }
  painting = true; lastVox = null; ghostLast = null;
  stroke = new Set(); lasso = []; lassoScreen = [];
  armGhost();
  (tool === "lasso" ? lassoAt : paintAt)(e);
  e.preventDefault();
};
onmousemove = e => { if (painting) (tool === "lasso" ? lassoAt : paintAt)(e); };
onmouseup = async () => {
  if (!painting) return;
  painting = false; lastVox = null; ghostLast = null;
  $("#ghost").style.display = "none";
  if (tool === "lasso") {
    if (lasso.length >= 3) await edit({op:"lasso", label: LABEL[mode], points: lasso});
  } else if (stroke.size) {
    await edit({op:"paint", label: LABEL[mode],
                voxels: [...stroke].map(s => s.split(",").map(Number))});
  }
  stroke = new Set(); lasso = []; lassoScreen = [];
};

document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => {
  mode = b.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(x => x.classList.toggle("on", x === b));
});
document.querySelectorAll("[data-whole]").forEach(b => b.onclick = () =>
  edit({op:"add", action: b.dataset.whole, target: null}));

function setTool(t) {
  tool = t;
  document.querySelectorAll("[data-tool]").forEach(x => x.classList.toggle("on", x.dataset.tool === t));
  $("#brushrow").style.display = t === "brush" ? "flex" : "none";
  $("#img").style.cursor = {brush: "cell", lasso: "copy"}[t] || "crosshair";
}
document.querySelectorAll("[data-tool]").forEach(b => b.onclick = () => setTool(b.dataset.tool));
$("#bsize").oninput = e => { bsize = +e.target.value; $("#bsizelab").textContent = bsize + " px"; };

$("#undo").onclick  = () => edit({op:"undo"});
$("#clear").onclick = () => edit({op:"clear"});
$("#prev").onclick  = () => { if (z > cur.z0) { z--; draw(); } };
$("#next").onclick  = () => { if (z < cur.z1) { z++; draw(); } };
$("#slider").oninput = e => { z = +e.target.value; draw(); };
$("#pid").onchange = e => pick(P.find(p => p.pid === e.target.value));
$("#suspect").onclick = e => {
  const d = e.target.closest(".sus");
  if (d) { z = +d.dataset.z; draw(); }
};
$("#save").onclick = async () => {
  const r = await (await fetch("/api/save", {method:"POST"})).json();
  $("#msg").textContent = r.message;
};

onkeydown = e => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === "ArrowLeft")  { $("#prev").click(); e.preventDefault(); }
  if (k === "ArrowRight") { $("#next").click(); e.preventDefault(); }
  if (k === "1") document.querySelector('[data-mode="aorta"]').click();
  if (k === "2") document.querySelector('[data-mode="esophagus"]').click();
  if (k === "3") document.querySelector('[data-mode="ambiguous"]').click();
  if (k === "b") setTool({blob: "brush", brush: "lasso", lasso: "blob"}[tool]);
  if (k === "[" || k === "]") {
    bsize = Math.min(10, Math.max(1, bsize + (k === "]" ? 1 : -1)));
    $("#bsize").value = bsize; $("#bsizelab").textContent = bsize + " px";
  }
  if (k === "s") edit({op:"add", action:"swap", target:null});
  if (k === "u") $("#undo").click();
};

boot();
</script>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source_dir", type=str, required=True)
    p.add_argument("--gt_name", type=str, default="GT_fixed.nii.gz")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--patients", type=str, nargs="*", default=[],
                   help="only load these, e.g. --patients Patient_14 Patient_15")
    p.add_argument("--no_browser", action="store_true")
    args = p.parse_args()

    folders = sorted((Path(args.source_dir) / "train").glob("Patient_*"))
    if args.patients:
        folders = [f for f in folders if f.name in args.patients]
    assert folders, "no patients found"

    print(f">> loading {len(folders)} patients (a few seconds each)")
    patients = []
    for f in folders:
        patients.append(Patient(f, args.gt_name))
        print(f"   {f.name}  slices {patients[-1].z0}-{patients[-1].z1}"
              f"  {len(patients[-1].suspect)} flagged")

    STATE.update(patients=patients, by_id={p.pid: p for p in patients}, edits={})

    url = f"http://localhost:{args.port}"
    print(f"\n>> open {url}   (ctrl-c to stop; nothing is written until you press Save)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
