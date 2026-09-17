# AI for medical imaging — Fall 2026 group project

<!-- MarkdownTOC autolink="true" autoanchor="true" -->

- [Project overview](#project-overview)
- [Codebase features](#codebase-features)
- [Codebase use](#codebase-use)
    - [Setting up the environment](#setting-up-the-environment)
        - [Setting up the environment - Some troubleshooting for windows users](#setting-up-the-environment---some-troubleshooting-for-windows-users)
    - [Getting the data](#getting-the-data)
    - [The merged aorta label](#the-merged-aorta-label)
        - [Step 1 - reconstruct the split automatically](#step-1-reconstruct-the-split)
        - [Step 2 - correct the rest by hand](#step-2-correct-the-rest-by-hand)
        - [Step 3 - slice to png](#step-3-slice-to-png)
        - [Step 4 - train on the corrected data](#step-4-train)
    - [Training a base network](#training-a-base-network)
    - [Viewing the results](#viewing-the-results)
        - [2D viewer](#2d-viewer)
        - [3D viewers](#3d-viewers)
    - [Plotting the metrics](#plotting-the-metrics)
- [Submission and scoring](#submission-and-scoring)
    - [Packing the code](#packing-the-code)
    - [Saving the best model](#saving-the-best-model)
    - [Archiving everything for submission](#archiving-everything-for-submission)
- [Known issues](#known-issues)
    - [Cannot pickle lambda in the dataloader](#cannot-pickle-lambda-in-the-dataloader)
    - [Pytorch not compiled for Numpy 2.0](#pytorch-not-compiled-for-numpy-20)
    - [Viewer on Windows](#viewer-on-windows)

<!-- /MarkdownTOC -->


<a id="project-overview"></a>
## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.


<a id="codebase-features"></a>
## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`; **To re-implement as assignment 01**
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files; **To re-implement as assignment 03**
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

<a id="codebase-use"></a>
## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

<a id="setting-up-the-environment"></a>
### Setting up the environment
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

<a id="setting-up-the-environment---some-troubleshooting-for-windows-users"></a>
#### Setting up the environment - Some troubleshooting for windows users
These steps assume you are using Git Bash + Anaconda + an IDE (e.g., PyCharm).

Open git bash and run:
Step 1:
```
$ git clone https://github.com/HKervadec/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```
Step 2:
```
# 1) Create a fresh conda env with Python 3.10+ (matches project note)
$ conda create -n ai4mi python=3.10 -y

# 2) Activate it
$ conda activate ai4mi

# 3) (Optional but nice) make sure pip is present/updated
$ python -m pip install --upgrade pip

# 4) From the repo folder, install dependencies with pip
$ python -m pip install -r requirements.txt
```

Some common troubleshooting for windows users:

In case in bash u got - conda: command not found

Open Anaconda Prompt:
```
conda init bash

#Find where conda is installed
where conda
```
Yous should get sth like - C:\Users\<YourName>\anaconda3

Close and open git bash - change CONDA_HOME in the code below

```
$ CONDA_HOME="/c/Users/<YourName>/anaconda3"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    . "$CONDA_HOME/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_HOME:$CONDA_HOME/Scripts:$CONDA_HOME/Library/bin:$PATH"
fi
```
You can also create new conda environment in anaconda prompt 

<a id="getting-the-data"></a>
### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_part1.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/IQBJLXRY5wedSYEuofqRtuylAWiiHp2ciems5XSCu3DFMkA?e=qa3Ujf) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```


For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

<a id="the-merged-aorta-label"></a>
### The merged aorta label

Part 1 of the data shipped with the aorta folded into the esophagus label: every
`GT.nii.gz` holds only `{0, 1, 2, 3}`, and a class-1 voxel means "esophagus **or**
aorta". Only `Patient_07` has the intended five-class encoding, in `GT2.nii.gz`.

Recovering the split takes three steps: reconstruct it automatically, fix the few
slices the reconstruction gets wrong by hand, then slice the result to `.png`.


<a id="step-1-reconstruct-the-split"></a>
#### Step 1 - reconstruct the split automatically

```
$ python fix_label_encoding.py --source_dir data/segthor_part1
```

This writes `GT_fixed.nii.gz` beside each patient's `GT.nii.gz`. It separates the
two organs by calibre: the largest disk that fits inside the aorta is 10-14 mm in
plane, the esophagus 4-10 mm. Against `Patient_07`'s real `GT2.nii.gz` it scores
esophagus DSC 0.9982 and aorta DSC 0.9995, and it is flat for any `--r_aorta`
between 6 and 11 mm. Patients 02, 03, 06, 09 and 16 were scanned with contrast
(aorta 137-209 HU against an esophagus at 31-51 HU), so an intensity threshold
gives five further, fully independent references; agreement there is DSC
0.995-1.000.

It is still a reconstruction, and wherever it is wrong those errors become the
labels the network trains on. In practice it fails on a handful of slices per
patient, in predictable places: the ends of the aorta tube, and slices where the
two organs touch and merge into a single blob.

<a id="step-2-correct-the-rest-by-hand"></a>
#### Step 2 - correct the rest by hand

NOTE: corrections.py already contains the corrections, so running 
this is not necessary and may erase the corrections --> needs fix

```
$ python relabel_gui.py --source_dir data/segthor_part1
```

This opens a browser tool at `http://localhost:8000` for stepping through the
slices. It is easy to tell the two apart by eye: the descending aorta is the big
round vessel just left of the spine, the esophagus the smaller, squashier one in
front of it. Three tools, all of which only ever move voxels between the esophagus
and the aorta:

* **Whole blob** - click a blob to relabel all of it. This is the common case: the
  reconstruction hands an entire tube cross-section to the wrong organ.
* **Brush** - drag to paint individual voxels, size on a slider or `[` / `]`.
* **Lasso** - draw a loop, everything inside takes the current label. Use this
  where the *border* between two touching tubes sits in the wrong place.

Plus whole-slice buttons (swap the two, all aorta, all esophagus), `Undo`, and a
**worth a look** list that jumps to the slices most likely to be wrong, with the
reason. That list keeps the review to a couple of dozen slices per patient rather
than all 130. Keys: arrows for slices, `1` `2` `3` for the labels, `b` for the next
tool, `s` swap, `u` undo.

**Pressing Save writes your decisions to `corrections.py`** - a plain dict of
patient, slice range and action, so the manual pass is a readable, re-runnable part
of the pipeline rather than an undocumented step:

```python
CORRECTIONS: dict[str, list[tuple]] = {
    "Patient_14": [
        ("93", "esophagus", (262, 297)),   # one blob, named wrong
        ("109-119", "esophagus"),          # whole slice
    ],
}
```

Any freehand brush or lasso voxels go to `corrections_paint.npz` alongside it,
since a voxel set cannot be written as a readable slice-and-action line. Nothing is
written until you press Save. You can also edit `corrections.py` by hand - the
actions are `aorta`, `esophagus`, `swap`, `ambiguous` and `drop`, and the file's
docstring documents them.

Then apply them. Check first, look at the pictures, and only then write:

```
$ python apply_corrections.py --source_dir data/segthor_part1 --dry_run
$ python apply_corrections.py --source_dir data/segthor_part1 --preview qc/
```

That writes `GT_corrected.nii.gz` per patient and drops a three-panel picture (CT,
before, after) for every slice it touched into `qc/`. Flipping through those is the
cheapest check that a click landed where you meant it.

Both scripts assert before saving that only esophagus and aorta voxels moved, so a
wrong entry can make the split worse but cannot touch the heart, the trachea, or
which voxels are foreground at all.


<a id="step-3-slice-to-png"></a>
#### Step 3 - slice to `.png`

```
$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python slice_segthor.py --source_dir data/segthor_part1 --dest_dir data/SEGTHOR_tmp \
      --gt_name GT_corrected.nii.gz --shape 256 256 --retain 5 -p -1
$ mv data/SEGTHOR_tmp data/SEGTHOR
```

Note that `make data/SEGTHOR` slices `GT_fixed.nii.gz`, the *uncorrected*
reconstruction, so use the command above once you have `GT_corrected.nii.gz` — or
point the recipe at the corrected file. The `--retain 5` split is drawn with a
fixed seed, so the same command always produces the same train/val division.


<a id="step-4-train"></a>
#### Step 4 - train on the corrected data

```
$ python -O main.py --dataset SEGTHOR --mode full --epochs 25 \
      --dest results/segthor_corrected/ce --gpu
```

Ordinary cross-entropy over the five classes. Drop `-O` for the first epoch or two
if you want the sanity assertions checked, and drop `--gpu` to run on the CPU.

One caveat when reading the numbers: `dice_val.npy` averages over every slice,
including the many that contain none of the organ at all. Those score close to 1.0
through the smoothing term, so the per-class means come out higher than the model
deserves — on one run the heart read 0.898 where the figure over organ-bearing
slices only was 0.680. Compare runs against each other freely, but restrict to the
slices where a class is present before quoting an absolute DSC.

<a id="training-a-base-network"></a>
### Training a base network
Running a training
```
$ python main.py --help
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--gpu] [--debug]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights).
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
$ python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

The codebase uses a lot of assertions for control and self-documentation, they can easily be disabled with the `-O` option (for faster training) once everything is known to be correct (for instance run the previous command for 1/2 epochs, then kill it and relaunch it):
```
$ python -O main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

<a id="viewing-the-results"></a>
### Viewing the results
<a id="2d-viewer"></a>
#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back):
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


```
$ python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
    --legend --class_names background esophagus heart trachea aorta
```
<!-- ![Example of the viewer on SegTHOR](viewer_segthor.png) -->

<a id="3d-viewers"></a>
#### 3D viewers
To look at the results in 3D, it is necessary to reconstruct the 3D volume from the individual 2D predictions saved as images.
To stitch the `.png` back to a nifti file:
```
$ python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
```

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](3dslicer_zoom.png)


<a id="plotting-the-metrics"></a>
### Plotting the metrics
There are some facilities to plot the metrics saved by [`main.py`](main.py):
```
$ python plot.py --help
usage: plot.py [-h] --metric_file METRIC_MODE.npy [--dest METRIC_MODE.png] [--headless]

Plot data over time

options:
  -h, --help            show this help message and exit
  --metric_file METRIC_MODE.npy
                        The metric file to plot.
  --dest METRIC_MODE.png
                        Optional: save the plot to a .png file
  --headless            Does not display the plot and save it directly (implies --dest to be provided.
$ python plot.py --metric_file results/segthor/ce/dice_val.npy --dest results/segthor/ce/dice_val.png
```
![Validation DSC](dice_val.png)


<a id="submission-and-scoring"></a>
## Submission and scoring
Groups will have to submit:
* archive of the git repo with the whole project, which includes:
    * slicing (if any) and any other pre-processing;
    * training;
    * post-processing when applicable;
    * inference;
    * metrics computation/scripts to run the metrics submodule;
* the best trained model;
* predictions on the test set (`sha256sum -c data/test.zip.sha256` as optional checksum);
* predictions on the group's internal validation set, the labels of their validation set, and the metrics they computed (akin to Assignment 3).

The main criterions for scoring will include (listed here only for convenience, please see Canvas for reference rubric):
* improvement or lack thereof of performances over baseline;
* code quality/clear [git use](git.md);
* the [final choice of metrics](https://metrics-reloaded.dkfz.de/) (they need to be in 3D);
* correctness of the computed metrics (on the validation set);
* oral presentation.


<a id="packing-the-code"></a>
### Packing the code
`$ git bundle create group-XX.bundle master`

<a id="saving-the-best-model"></a>
### Saving the best model
`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

<a id="archiving-everything-for-submission"></a>
### Archiving everything for submission
All files should be grouped in single folder with the following structure
```
group-XX/
    test/
        pred/
            Patient_41.nii.gz
            Patient_42.nii.gz
            ...
    val/
        pred/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        gt/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        metric01.npz
        metric02.npz
        ...
    group-XX.bundle
    bestmodel-group-XX.pkl
```
The metrics should be a `.npz` archives, that maps patient ID (e.g., `Patient_21`) to a `ndarray` with shape `KxD` (or `K` if `D = 1`), with `K` the number of classes and `D` the eventual dimensionality of the metric (can be simply 1). Ultimately it is the same format as Distorch from Assignment 3.


The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:
```
Example using Zstandard:
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
Example using gunzip:
$ tar cf group-XX.tar.gz - group-XX/
```

<a id="known-issues"></a>
## Known issues
<a id="cannot-pickle-lambda-in-the-dataloader"></a>
### Cannot pickle lambda in the dataloader
Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

<a id="pytorch-not-compiled-for-numpy-20"></a>
### Pytorch not compiled for Numpy 2.0
It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

<a id="viewer-on-windows"></a>
### Viewer on Windows
Windows has different paths names (`\` in stead of `/`), so the default regex in the viewer needs to be changed to `--id_regex=".*\\\\(.*).png"`.
