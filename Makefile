red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

## Recover the aorta (class 4) that part 1 merged into the esophagus label.
## Writes GT_fixed.nii.gz beside each patient's GT.nii.gz; the stamp file exists
## because the real outputs are spread over all 20 patient folders.
data/.labels_fixed: data/segthor_part1 fix_label_encoding.py
	$(info $(magenta)python $(CFLAGS) fix_label_encoding.py$(reset))
	python $(CFLAGS) fix_label_encoding.py --source_dir data/segthor_part1
	touch $@

data/SEGTHOR: data/.labels_fixed
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--gt_name GT_fixed.nii.gz --shape 256 256 --retain 5
	mv $@_tmp $@
