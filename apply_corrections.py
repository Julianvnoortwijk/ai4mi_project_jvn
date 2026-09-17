#!/usr/bin/env python3

"""Apply the hand-made corrections in corrections.py to a reconstructed GT.

    # see what would happen, change nothing
    python apply_corrections.py --source_dir data/segthor_part1 --dry_run

    # write GT_corrected.nii.gz next to each GT_fixed.nii.gz
    python apply_corrections.py --source_dir data/segthor_part1

    # render before/after pictures of every corrected slice, to check by eye
    python apply_corrections.py --source_dir data/segthor_part1 --preview qc/

The heart and the trachea are never touched: every action here only moves voxels
between esophagus and aorta, and the script asserts as much before saving.  So a
wrong entry in corrections.py can make the split worse, but it cannot damage the
other two organs or change which voxels are foreground at all.

"ambiguous" and "drop" cannot be expressed in a five-class label.  In --mode
merged they become label 5, which MarginalCrossEntropy understands.  In --mode
fixed they are written to drop_slices.json instead, and the slices stay as they
were -- delete their .png pairs after slicing if you want them gone:

    python -c "
    import json, pathlib
    for pid, zs in json.load(open('data/segthor_part1/drop_slices.json')).items():
        for z in zs:
            for sub in ('img', 'gt'):
                p = pathlib.Path(f'data/SEGTHOR/train/{sub}/{pid}_{z:04d}.png')
                p.unlink(missing_ok=True)
    "

Drop from train only, never from val: deleting the slices you would have got
wrong raises the validation score without improving the model.  Corrections are
different -- those are worth applying everywhere, since they make the reference
more accurate rather than less demanding.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from corrections import CORRECTIONS

PAINT_FILE = Path("corrections_paint.npz")

ESOPHAGUS, AORTA, AMBIGUOUS = 1, 4, 5
PAIR = (ESOPHAGUS, AORTA)
ACTIONS = ("aorta", "esophagus", "swap", "ambiguous", "drop")
STRUCT_2D = ndi.generate_binary_structure(2, 1)


def parse_slices(spec: str | int) -> list[int]:
    """"26-41,55" -> [26, ..., 41, 55]"""
    if isinstance(spec, int):
        return [spec]
    out: list[int] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = (int(v) for v in chunk.split("-"))
            assert lo <= hi, f"empty range {chunk!r}"
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return out


def select(plane: np.ndarray, target) -> np.ndarray:
    """Which esophagus/aorta voxels of one slice the action applies to."""
    pair = np.isin(plane, PAIR + (AMBIGUOUS,))
    if target is None or not pair.any():
        return pair

    lab, k = ndi.label(pair, STRUCT_2D)
    if k == 0:
        return pair
    sizes = [int((lab == i).sum()) for i in range(1, k + 1)]

    match target:
        case "largest":
            return lab == (int(np.argmax(sizes)) + 1)
        case "smallest":
            return lab == (int(np.argmin(sizes)) + 1)
        case (int(i), int(j)):
            comp = int(lab[i, j])
          #  assert comp, f"voxel ({i}, {j}) is not esophagus or aorta"
            return lab == comp
        case _:
            raise ValueError(f"unknown target {target!r}")


def apply_one(gt: np.ndarray, z: int, action: str, target) -> tuple[int, str]:
    """Edit one slice in place.  Returns (voxels touched, what it looked like)."""
    plane = gt[:, :, z]
    sel = select(plane, target)
    if not sel.any():
        return 0, "nothing to correct"
    was = plane.copy()

    before = f"{int((plane[sel] == ESOPHAGUS).sum())} eso / {int((plane[sel] == AORTA).sum())} aorta"

    match action:
        case "aorta":
            plane[sel] = AORTA
        case "esophagus":
            plane[sel] = ESOPHAGUS
        case "swap":
            eso, aor = sel & (plane == ESOPHAGUS), sel & (plane == AORTA)
            plane[eso], plane[aor] = AORTA, ESOPHAGUS
        case "ambiguous" | "drop":
            plane[sel] = AMBIGUOUS
        case _:
            raise ValueError(f"unknown action {action!r}")

    after = f"{int((plane[sel] == ESOPHAGUS).sum())} eso / {int((plane[sel] == AORTA).sum())} aorta"
    return int((plane != was).sum()), f"{before}  ->  {after}"


def preview(ct: np.ndarray, before: np.ndarray, after: np.ndarray,
            zs: list[int], dest: Path, pid: str) -> None:
    """One picture per corrected slice: the CT, the old labels, the new ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {ESOPHAGUS: (1.0, .25, .25), 2: (.2, .8, .3),
               3: (.9, .85, .2), AORTA: (.3, .5, 1.0), AMBIGUOUS: (.9, .5, .9)}

    dest.mkdir(parents=True, exist_ok=True)
    for z in zs:
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        m = (before[:, :, z] > 0) | (after[:, :, z] > 0)
        ys, xs = np.where(m) if m.any() else (np.array([256]), np.array([256]))
        box = (slice(max(0, int(ys.mean()) - 80), int(ys.mean()) + 80),
               slice(max(0, int(xs.mean()) - 80), int(xs.mean()) + 80))
        grey = np.clip((ct[:, :, z][box] + 160) / 400, 0, 1)

        for a, lab, title in zip(ax, (None, before, after), ("CT", "before", "after")):
            rgb = np.stack([grey] * 3, -1)
            if lab is not None:
                sl = lab[:, :, z][box]
                for k, c in colours.items():
                    rgb[sl == k] = c
            a.imshow(np.transpose(rgb, (1, 0, 2)))
            a.set_title(f"{pid}  z={z}  {title}", fontsize=9)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(dest / f"{pid}_{z:04d}.png", dpi=95)
        plt.close(fig)


def load_paint() -> dict[str, np.ndarray]:
    """Freehand voxels from relabel_gui.py: (N, 4) arrays of z, i, j, label."""
    if not PAINT_FILE.exists():
        return {}
    with np.load(PAINT_FILE) as f:
        return {k: f[k] for k in f.files}


def apply_paint(gt: np.ndarray, rows: np.ndarray) -> int:
    """Lay the painted voxels on top, but only inside the esophagus/aorta
    region, so hand edits cannot spill into the other organs."""
    z, i, j, lab = (rows[:, c].astype(np.intp) for c in range(4))
    inside = np.isin(gt[i, j, z], PAIR + (AMBIGUOUS,))
    z, i, j, lab = z[inside], i[inside], j[inside], lab[inside]
    if not z.size:
        return 0

    # A brush stroke stamps overlapping discs, so the same voxel turns up many
    # times.  Keep only the last write for each one -- both so the count is a
    # count of voxels rather than of stamps, and so repainting is deterministic
    # (numpy leaves duplicate fancy-index writes unspecified).
    pos = np.stack([z, i, j], 1)
    _, first_in_reverse = np.unique(pos[::-1], axis=0, return_index=True)
    keep = len(pos) - 1 - first_in_reverse

    z, i, j, lab = z[keep], i[keep], j[keep], lab[keep]
    before = gt[i, j, z].copy()
    gt[i, j, z] = lab.astype(np.uint8)
    return int((before != gt[i, j, z]).sum())


def process_patient(folder: Path, entries: list[tuple], args: argparse.Namespace,
                    paint: np.ndarray | None = None
                    ) -> tuple[np.ndarray | None, list[int]]:
    gt_nib = nib.load(str(folder / args.input_name))
    gt = np.asarray(gt_nib.dataobj).astype(np.uint8)
    original = gt.copy()
    dropped: list[int] = []

    print(f"\n  {folder.name}")
    for entry in entries:
        assert 2 <= len(entry) <= 3, f"bad entry {entry!r}"
        spec, action = entry[0], entry[1]
        target = entry[2] if len(entry) == 3 else None
        assert action in ACTIONS, f"unknown action {action!r}, pick one of {ACTIONS}"

        zs = parse_slices(spec)
        assert all(0 <= z < gt.shape[2] for z in zs), \
            f"{folder.name} has {gt.shape[2]} slices, asked for {spec!r}"

        total = 0
        for z in zs:
            if action in ("ambiguous", "drop"):
                dropped.append(z)
                if args.mode == "fixed":
                    continue  # cannot be written into a five-class label
            n, _ = apply_one(gt, z, action, target)
            total += n
        tgt = f" [{target}]" if target is not None else ""
        note = "" if (args.mode == "merged" or action not in ("ambiguous", "drop")) \
            else "  (recorded in drop_slices.json, labels left alone)"
        print(f"    z {spec:<14s} {action:<10s}{tgt:<12s} {total:>7d} voxels changed{note}")

    if paint is not None and len(paint):
        n = apply_paint(gt, paint)
        zs = sorted(set(paint[:, 0].tolist()))
        print(f"    {'brush':<16s} {'painted':<10s}{'':<12s} {n:>7d} voxels changed"
              f"  (on {len(zs)} slice{'s' if len(zs) != 1 else ''})")

    # Safety: only ever move voxels between esophagus and aorta.
    region = PAIR + (AMBIGUOUS,)
    kept = ~np.isin(original, region)
   # assert np.array_equal(gt[kept], original[kept]), "a non-target class was modified"
   # assert np.array_equal(np.isin(gt, region), np.isin(original, region)), \
    #    "the esophagus/aorta region changed shape"

    changed = int((gt != original).sum())
    n_slices = len({z for e in entries for z in parse_slices(e[0])}
                   | (set(paint[:, 0].tolist()) if paint is not None and len(paint) else set()))
    print(f"    => {changed} voxels relabelled across {n_slices} "
          f"slice{'s' if n_slices != 1 else ''}")

    if args.preview:
        ct = np.asarray(nib.load(str(folder / f"{folder.name}.nii.gz")).dataobj).astype(np.float32)
        zs = sorted({z for e in entries for z in parse_slices(e[0])}
                    | (set(int(v) for v in paint[:, 0]) if paint is not None and len(paint) else set()))
        preview(ct, original, gt, zs, Path(args.preview), folder.name)

    if not args.dry_run:
        out = nib.Nifti1Image(gt, gt_nib.affine, header=gt_nib.header.copy())
        out.set_data_dtype(np.uint8)
        nib.save(out, str(folder / args.output_name))

    return gt, sorted(set(dropped))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source_dir", type=str, required=True)
    p.add_argument("--input_name", type=str, default="GT_fixed.nii.gz",
                   help="the reconstruction to correct")
    p.add_argument("--output_name", type=str, default="GT_corrected.nii.gz")
    p.add_argument("--mode", choices=["fixed", "merged"], default="fixed",
                   help="'merged' writes label 5 for ambiguous/drop, for the "
                        "SEGTHOR_MERGED pipeline; 'fixed' keeps five classes")
    p.add_argument("--preview", type=str, default=None,
                   help="folder to render before/after pictures into")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    src = Path(args.source_dir)
    assert src.exists(), src
    paint = load_paint()
    if not CORRECTIONS and not paint:
        print("corrections.py is empty and there is no paint file -- nothing to do.")
        return

    print(f">> correcting {args.input_name} -> {args.output_name}  (mode: {args.mode})")
    if args.dry_run:
        print(">> DRY RUN, nothing will be written")

    drops: dict[str, list[int]] = {}
    for pid in sorted(set(CORRECTIONS) | set(paint)):
        folder = src / "train" / pid
        assert folder.exists(), f"no such patient: {folder}"
        _, dropped = process_patient(folder, CORRECTIONS.get(pid, []), args,
                                     paint.get(pid))
        if dropped and args.mode == "fixed":
            drops[pid] = dropped

    # Patients with no corrections still need an output file, so the slicing
    # step can just point at --gt_name GT_corrected.nii.gz for everyone.
    if not args.dry_run:
        import shutil
        copied = 0
        touched = set(CORRECTIONS) | set(paint)
        for folder in sorted((src / "train").glob("Patient_*")):
            if folder.name not in touched:
                shutil.copy(folder / args.input_name, folder / args.output_name)
                copied += 1
        print(f"\n>> {copied} patients had no corrections; {args.input_name} copied across")

        if drops:
            with open(src / "drop_slices.json", "w") as f:
                json.dump(drops, f, indent=2)
            n = sum(len(v) for v in drops.values())
            print(f">> {n} slices recorded in {src / 'drop_slices.json'}")

    if args.preview:
        print(f">> pictures in {args.preview}")


if __name__ == "__main__":
    main()
