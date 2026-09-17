#!/usr/bin/env python3

"""Recover the 5-class SegTHOR encoding from the merged part-1 ground truth.

In segthor_part1 the aorta is folded into the esophagus label, so GT.nii.gz only
holds {0, 1, 2, 3}.  This writes GT_fixed.nii.gz with the intended encoding:

    0 background   1 esophagus   2 heart   3 trachea   4 aorta

Where a real GT2.nii.gz exists (Patient_07 only) it is copied verbatim.

    python fix_label_encoding_minimal.py --source_dir data/segthor_part1

"""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

MERGED_LABEL, ESOPHAGUS, AORTA = 1, 1, 4
R_AORTA = 7.0                                        # mm; stable anywhere in 5-9

STRUCT_3D = ndi.generate_binary_structure(3, 3)     
STRUCT_2D_LABEL = ndi.generate_binary_structure(2, 1)
STRUCT_2D_GROW = ndi.generate_binary_structure(2, 2)


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, STRUCT_3D)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def split_merged_class(merged: np.ndarray,
                       spacing: tuple[float, float, float],
                       r_aorta: float = R_AORTA) -> tuple[np.ndarray, np.ndarray]:
    """Split the merged class into (esophagus, aorta) using calibre."""
    dist = ndi.distance_transform_edt(merged, sampling=spacing)

    # The thickest part of the esophagus also survives the threshold, but as a
    # separate blob.  The aorta is one connected tube, so keeping only the largest
    # component drops that contaminant.
    aorta_seed = largest_component(dist > r_aorta)
    if not aorta_seed.any():
        return merged.copy(), np.zeros_like(merged)

    esophagus_seed = np.zeros_like(merged)
    for z in range(merged.shape[2]):
        plane = merged[:, :, z]
        if not plane.any():
            continue
        lab, k = ndi.label(plane, STRUCT_2D_LABEL)
        for i in range(1, k + 1):
            comp = lab == i
            if (aorta_seed[:, :, z] & comp).any():
                continue                     # this blob is (part of) the aorta

            # A thin blob sitting where the aorta runs in the neighbouring slices
            # is the aorta narrowing, not the esophagus.
            neigh = np.zeros_like(comp)
            for dz in (-1, 1):
                if 0 <= z + dz < merged.shape[2]:
                    neigh |= aorta_seed[:, :, z + dz]
            if (comp & ndi.binary_dilation(neigh, STRUCT_2D_GROW, 3)).sum() > 0.5 * comp.sum():
                continue

            esophagus_seed[:, :, z] |= comp

    if not esophagus_seed.any():
        return np.zeros_like(merged), merged.copy()

    markers = np.zeros(merged.shape, np.int32)
    markers[esophagus_seed] = 1
    markers[aorta_seed] = 2
    labels = watershed(-dist, markers, mask=merged)

    eso, aor = largest_component(labels == 1), largest_component(labels == 2)

    # Whatever the cleanup orphaned goes to the nearer class, so no annotated
    # voxel is silently dropped.
    orphan = merged & ~(eso | aor)
    if orphan.any():
        d_e = ndi.distance_transform_edt(~eso, sampling=spacing)
        d_a = ndi.distance_transform_edt(~aor, sampling=spacing)
        eso = eso | (orphan & (d_e <= d_a))
        aor = aor | (orphan & (d_a < d_e))

    assert not (eso & aor).any()
    assert np.array_equal(eso | aor, merged)
    return eso, aor


def process_patient(folder: Path, output_name: str) -> None:
    gt_nib = nib.load(str(folder / "GT.nii.gz"))
    gt = np.asarray(gt_nib.dataobj).astype(np.uint8)
    spacing = tuple(float(v) for v in gt_nib.header.get_zooms()[:3])

    reference = folder / "GT2.nii.gz"
    if reference.exists():
        shutil.copy(reference, folder / output_name)
        return

    merged = gt == MERGED_LABEL
    if AORTA in np.unique(gt) or not merged.any():
        return

    eso, aor = split_merged_class(merged, spacing)

    fixed = gt.copy()
    fixed[merged] = 0
    fixed[eso] = ESOPHAGUS
    fixed[aor] = AORTA

    assert np.array_equal(fixed > 0, gt > 0)
    for k in (2, 3):
        assert np.array_equal(fixed == k, gt == k)

    out = nib.Nifti1Image(fixed, gt_nib.affine, header=gt_nib.header.copy())
    out.set_data_dtype(np.uint8)
    nib.save(out, str(folder / output_name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source_dir", type=str, required=True)
    p.add_argument("--output_name", type=str, default="GT_fixed.nii.gz")
    args = p.parse_args()

    for folder in sorted((Path(args.source_dir) / "train").glob("Patient_*")):
        process_patient(folder, args.output_name)


if __name__ == "__main__":
    main()
