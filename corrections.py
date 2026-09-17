#!/usr/bin/env python3

"""Manual corrections to the reconstructed esophagus/aorta split.

This file is the record of what a human decided by eye.  `apply_corrections.py`
reads it and writes a corrected ground truth; nothing else touches it.  Keeping
the decisions here rather than making them by hand means the whole pipeline can
be re-run from the zip, and the manual pass can be described in the report.

Each entry is  (slices, action)  or  (slices, action, target).

slices   "42"          a single slice
         "26-41"       an inclusive range
         "26-41,55"    several of those, comma separated
         The number is the z index of the nifti array, which is also the number
         in the sliced filenames: Patient_14_0101.png is z = 101.

action   "aorta"       label the targeted voxels aorta (4)
         "esophagus"   label them esophagus (1)
         "swap"        exchange the two labels -- for when the split is right
                       but the two organs were named the wrong way round
         "ambiguous"   give up on the split and mark the voxels "esophagus or
                       aorta" (label 5).  Only meaningful for the SEGTHOR_MERGED
                       pipeline, where MarginalCrossEntropy can use it; in the
                       ordinary pipeline these slices are reported as a drop list
         "drop"        the slice is unusable; report it in the drop list

target   omitted       every esophagus/aorta voxel in the slice
         "largest"     only the biggest blob in the slice (usually the aorta)
         "smallest"    only the smallest blob
         (i, j)        only the blob containing that voxel

Pick the action that matches what you see:

  one blob, whole thing named wrong          -> "aorta" or "esophagus"
  two blobs, names the wrong way round       -> "swap"
  two blobs, only one named wrong            -> "aorta"/"esophagus" + a target
  the border between them is in the wrong    -> "ambiguous" (or "drop")
  place and redrawing it is not worth it

Example of what a filled-in entry looks like:

    "Patient_14": [
        ("101-103", "aorta"),               # aorta tube called esophagus
        ("110", "swap"),                    # the two are the wrong way round
        ("78-83", "ambiguous"),             # border wrong where they touch
    ],
"""

CORRECTIONS: dict[str, list[tuple]] = {
    "Patient_01": [
        ("154", "esophagus", (268, 238)),
        ("155", "esophagus", (277, 257)),
    ],
    "Patient_04": [
        ("94", "aorta", (277, 267)),
    ],
    "Patient_06": [
        ("139", "aorta", (286, 247)),
        ("140", "aorta", (281, 241)),
    ],
    "Patient_08": [
        ("109", "aorta", (294, 272)),
        ("109", "aorta", (280, 276)),
        ("110", "aorta", (284, 276)),
    ],
    "Patient_14": [
        ("93", "esophagus", (262, 297)),
        ("96", "esophagus", (263, 293)),
        ("97", "esophagus", (269, 291)),
        ("102", "esophagus", (268, 286)),
        ("104", "esophagus", (261, 253)),
        ("104", "esophagus", (275, 284)),
        ("105", "esophagus", (267, 278)),
        ("106", "esophagus", (267, 282)),
        ("107", "esophagus", (280, 280)),
        ("108", "esophagus", (274, 271)),
        ("109-119", "esophagus"),
    ],
    "Patient_15": [
        ("26", "esophagus", (278, 216)),
        ("27", "esophagus", (276, 216)),
        ("28", "esophagus", (272, 219)),
        ("30-31", "esophagus", (271, 229)),
        ("32-36", "esophagus", (280, 245)),
        ("37", "esophagus", (280, 255)),
        ("38-39", "esophagus", (282, 255)),
        ("40-41", "esophagus", (283, 255)),
        ("42", "esophagus", (288, 259)),
        ("43", "esophagus", (292, 260)),
        ("45", "esophagus", (302, 268)),
        ("46-49", "esophagus", (302, 267)),
        ("52", "esophagus", (300, 268)),
        ("56", "esophagus", (307, 270)),
        ("80,83", "aorta", (292, 304)),
    ],
    "Patient_17": [
        ("104", "aorta", (280, 245)),
        ("108", "aorta", (270, 244)),
    ],
    "Patient_19": [
        ("100", "aorta", (254, 271)),
    ],
}
