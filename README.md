# CNN-Based Image Reconstruction Method for Ultrafast Ultrasound Imaging: Code

## Summary

This repository contains the code related to the following paper:

D. Perdios, M. Vonlanthen, F. Martinez, M. Arditi and J.-P. Thiran,
“CNN-Based Image Reconstruction Method for Ultrafast Ultrasound Imaging,”
in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
vol. 69, no. 4, pp. 1154-1168, April 2022,
doi: [10.1109/TUFFC.2021.3131383][tuffc-doi].

The accepted version of this paper is also available on arXiv:
[arXiv:2008.12750][arxiv-doi].

The corresponding data is available online at [10.21227/vn0e-cw64][data-doi].

### Content

- [Installation](#installation):
Installation instructions
- [Data](#data):
Detailed description of the corresponding data
(training, test, trained models, etc.)
- [Scripts](#scripts):
Description of the scripts to reproduce the results
- [License](#license)
- [Acknowledgments](#acknowledgments)

### Paper Abstract

> Ultrafast ultrasound (US) revolutionized biomedical imaging with its capability of acquiring full-view frames at over 1 kHz, unlocking breakthrough modalities such as shear-wave elastography and functional US neuroimaging. Yet, it suffers from strong diffraction artifacts, mainly caused by grating lobes, side lobes, or edge waves. Multiple acquisitions are typically required to obtain a sufficient image quality, at the cost of a reduced frame rate. To answer the increasing demand for high-quality imaging from single unfocused acquisitions, we propose a two-step convolutional neural network (CNN)-based image reconstruction method, compatible with real-time imaging. A low-quality estimate is obtained by means of a backprojection-based operation, akin to conventional delay-and-sum beamforming, from which a high-quality image is restored using a residual CNN with multiscale and multichannel filtering properties, trained specifically to remove the diffraction artifacts inherent to ultrafast US imaging. To account for both the high dynamic range and the oscillating properties of radio frequency US images, we introduce the mean signed logarithmic absolute error (MSLAE) as a training loss function. Experiments were conducted with a linear transducer array, in single plane-wave (PW) imaging. Trainings were performed on a simulated dataset, crafted to contain a wide diversity of structures and echogenicities. Extensive numerical evaluations demonstrate that the proposed approach can reconstruct images from single PWs with a quality similar to that of gold-standard synthetic aperture imaging, on a dynamic range in excess of 60 dB. In vitro and in vivo experiments show that trainings carried out on simulated data perform well in experimental settings.

[data-doi]: https://dx.doi.org/10.21227/vn0e-cw64
[arxiv-doi]: https://doi.org/10.48550/arXiv.2008.12750
[tuffc-doi]: https://doi.org/10.1109/TUFFC.2021.3131383

## Installation

Simply clone or download this repository.

To obtain a Python environment with all dependencies,
the easiest is to create a [Conda] environment
(e.g., after having installed [Miniconda]).

GPU version (with a CUDA-capable device):

```bash
conda create -n dui-ultrafast python=3 tensorflow-gpu=1.14 gast=0.2.2
```

CPU-only version:

```bash
conda create -n dui-ultrafast python=3 tensorflow-gpu=1.14 gast=0.2.2
```

Most dependencies will be automatically installed
(e.g., `numpy`, `scipy`, and `h5py`).

Some other dependencies will need to be installed.
First, activate the newly created environment:

```bash
conda activate dui-ultrafast
```

Install the remaining dependencies:

```bash
conda install matplotlib pandas scikit-image
```

Note that specifying the `gast` version to `0.2.2` avoids a known
[issue][gast-issue].
(See this [comment][gast-cmt-1] and this [comment][gast-cmt-2].)

[conda]: https://docs.conda.io/en/latest/
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[gast-issue]: https://github.com/tensorflow/tensorflow/issues/32319
[gast-cmt-1]: https://github.com/tensorflow/tensorflow/issues/32319#issuecomment-537568070
[gast-cmt-2]: https://github.com/tensorflow/tensorflow/issues/32319#issuecomment-597946520

## Data

The data can be downloaded at [10.21227/vn0e-cw64][data-doi],
including training and test datasets, trained models, and results
(predictions and metrics).
Once downloaded, they should be organized as follows for direct use with
the code provided.

### Training Data

The training data is composed of four HDF5 files that should be accessible
from the directory `data/datasets/train`:

```text
20200304-ge9ld-random-phantom.hdf5
20200304-ge9ld-random-phantom.images.hq.hdf5
20200304-ge9ld-random-phantom.images.lq.hdf5
20200304-ge9ld-random-phantom.images.uq.hdf5
```

`20200304-ge9ld-random-phantom.hdf5` is the main HDF5 file.
It provides access to the complete simulated training dataset composed
of 31000 images for the three imaging configuration considered
(i.e., LQ, HQ, and UQ).
This dataset is described in Section III-B of the [paper][tuffc-doi].

The main HDF5 file contains three [HDF5 virtual datasets][hdf5-vds],
accessible identically to standard [HDF5 datasets][hdf5-dset].
Each dataset links to a corresponding HDF5 files with the following keys:

- `images/lq` -> `20200304-ge9ld-random-phantom.images.lq.hdf5`
- `images/hq` -> `20200304-ge9ld-random-phantom.images.hq.hdf5`
- `images/uq` -> `20200304-ge9ld-random-phantom.images.uq.hdf5`

To avoid breaking relative links, these four HDF5 file must be accessible
from the same directory.

[hdf5-dset]: https://docs.h5py.org/en/stable/high/dataset.html
[hdf5-vds]: https://docs.h5py.org/en/stable/vds.html

### Test Data

The test data is composed of 22 HDF5 files that should be accessible
from the directory `data/datasets/test`:

```text
20200304-ge9ld-numerical-test-phantom.hdf5
20200304-ge9ld-numerical-test-phantom.images.hq.hdf5
20200304-ge9ld-numerical-test-phantom.images.lq.hdf5
20200304-ge9ld-numerical-test-phantom.images.uq.hdf5
20200304-ge9ld-numerical-test-phantom-predictions.hdf5
20200304-ge9ld-random-phantom-test-set.hdf5
20200304-ge9ld-random-phantom-test-set.images.hq.hdf5
20200304-ge9ld-random-phantom-test-set.images.lq.hdf5
20200304-ge9ld-random-phantom-test-set.images.uq.hdf5
20200304-ge9ld-random-phantom-test-set.inclusions.amp.hdf5
20200304-ge9ld-random-phantom-test-set.inclusions.ang.hdf5
20200304-ge9ld-random-phantom-test-set.inclusions.pos.hdf5
20200304-ge9ld-random-phantom-test-set.inclusions.semiaxes.hdf5
20200304-ge9ld-random-phantom-test-set-predictions.hdf5
20200527-ge9ld-experimental-test-set-carotid-long.hdf5
20200527-ge9ld-experimental-test-set-carotid-long.images.hq.hdf5
20200527-ge9ld-experimental-test-set-carotid-long.images.lq.hdf5
20200527-ge9ld-experimental-test-set-carotid-long-predictions.hdf5
20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.hdf5
20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.images.hq.hdf5
20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.images.lq.hdf5
20200527-ge9ld-experimental-test-set-cirs054gs-hypo2-predictions.hdf5
```

The test data is composed of four main HDF5 files:

- `20200304-ge9ld-numerical-test-phantom.hdf5`:
Contains the numerical test phantom dataset (300 independent realizations).
It is described in Section III-D of the [paper][tuffc-doi].
- `20200304-ge9ld-random-phantom-test-set.hdf5`:
Contains 300 additional samples simulated identically to the training dataset.
- `20200527-ge9ld-experimental-test-set-carotid-long.hdf5`:
Contains the *in vivo* test dataset acquired on the carotid of a volunteer
(60 frames, longitudinal view).
It is described in Section III-E of the [paper][tuffc-doi].
- `20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.hdf5`:
Contains the *in vitro* test dataset acquired on a [CIRS model 054GS][cirs-054gs].
It is described in Section III-E of the [paper][tuffc-doi].

[cirs-054gs]: https://www.cirsinc.com/products/ultrasound/zerdine-hydrogel/general-purpose-ultrasound-phantom/

Similarly to the training dataset, each test dataset is composed of a main
HDF5 (described above) composed of virtual datasets that link to the
accompanying HDF5 files.
For example, the main HDF5 file `20200304-ge9ld-numerical-test-phantom.hdf5`
links to the corresponding files using the following keys:

- `images/lq` -> `20200304-ge9ld-numerical-test-phantom.images.lq.hdf5`
- `images/hq` -> `20200304-ge9ld-numerical-test-phantom.images.hq.hdf5`
- `images/uq` -> `20200304-ge9ld-numerical-test-phantom.images.uq.hdf5`

These accompanying HDF5 files must be accessible from the same directory
as the main one (to avoid breaking relative links).

Predictions obtained using the different trained models of each test dataset
are also made available:

- `20200304-ge9ld-numerical-test-phantom-predictions.hdf5`:
Contains the predictions of the numerical test phantom experiment
reported in Section IV-A (in particular Fig. 4).
- `20200304-ge9ld-random-phantom-test-set-predictions.hdf5`:
Contains the predictions of the additional simulated test dataset
used to compare visually (Supplementary Material, Fig. S4) the results obtained
from trainings performed with different image representations
(Supplementary Material, Section S-III-A)
and with different reference image configurations
(Supplementary Material, Section S-III-B).
- `20200527-ge9ld-experimental-test-set-carotid-long-predictions.hdf5`:
Contains the predictions of the *in vivo* experiment (sequence of a carotid)
reported in Section IV-B [in particular Fig. 6(d)-6(f)].
- `20200527-ge9ld-experimental-test-set-cirs054gs-hypo2-predictions.hdf5`:
Contains the predictions of the *in vitro* experiment reported in Section IV-B
[in particular Fig. 6(a)-6(c) and Table III].

### Trained Models

The archive `trained-models.tar.gz` contains the 25 trained models reported
in the [paper][tuffc-doi].
They should be made accessible from the directory `data/trained-models`:

```text
inp-lq-rf-out-uq-rf/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-env-out-uq-env/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-bm-out-uq-bm/mae-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-hq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mae-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mse-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-false-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-concat-rb-false-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-8-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-32-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-00200/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-00409/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-00837/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-01713/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-03504/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-07168/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-14664/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotnormal-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-heuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-henormal-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-1.0e-03-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-5.0e-04-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-1.0e-04-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
inp-lq-iq-out-uq-iq/mslae62-bs-2-adam-lr-1.0e-05-seed-5250-ts-30000/gunet-ch-16-ln-5-bs-2-ks-3-mf-2-af-relu-sc-add-rb-true-bki-glorotuniform-bbi-zeros-cki-glorotuniform-cbi-zeros
```

The naming convention of sub-directories is based on
the learned mapping, the training settings, and the network architecture
as `<mapping>/<training>/<network>`.

Each trained model directory contains:

- `checkpoints/`:
A directory to access two trained models: best and last.
- `configs/`:
A directory with all configuration files.
- `training.log`
A file containing all metrics computed during training.

### Metrics

The archive `metrics.tar.gz` is composed of 4 files that should be accessible
from the directory `data/metrics`:

```text
20200304-ge9ld-numerical-test-phantom-results.pickle
20200527-ge9ld-experimental-test-set-cirs054gs-hypo2-results.pickle
experimental-phantom-metrics.html
numerical-phantom-metrics.html
```

These files contain intermediate results and final metrics of the main two
quantitative experiments of the [paper][tuffc-doi],
namely,
the numerical test phantom (Section IV-A)
and the experimental test phantom (Section IV-B).
They are provided to avoid having to recompute them if one only wants to
reproduce specific figures of paper (see below for the corresponding scripts).

## Scripts

The following scripts can be used to reproduce part of the experiments
and results of the [paper][tuffc-doi]:

- `compute_predictions.py`:
Script to compute the predictions from the different test datasets
using the appropriate trained models.
Test datasets should be accessible from `data/datasets/test`
and trained models should be accessible from `data/trained-models`.
Prediction will be saved in `data/datasets/test`.
- `experimental_invitro_results.py`:
Script to compute the *in vitro* metrics (Table III).
It will also export the results as a Python pickle file in `data/metrics`
and as an HTML table (for direct visualization).
- `experimental_results_figure.py`:
Script to reproduce Fig. 6.
- `explore_training_data.py`:
Script to display (B-mode) a selected sample of the training dataset
with the different imaging configurations considered (i.e., LQ, HQ, and UQ).
The default sample is the one shown in Fig. 3.
- `hyperparameter_search_results.py`:
Script to reproduce the validation metric curves
of Figures S3, S5-S9 (Supplementary Material, Section S-III).
- `hyperparameter_search_image_comp.py`:
Script to reproduce Fig. S4.
- `numerical_test_phantom_results.py`:
Script to compute the metrics of the numerical test phantom results (Table II).
It will also export the results as a Python pickle file in `data/metrics`
and as an HTML table (for direct visualization).
- `numerical_test_phantom_figures.py`:
Script to reproduce Fig. 4, 5, and S10.
- `stats_rayleigh_confidence_interval.py`:
Script to reproduce Fig. S1.
- `train.py`:
Script to train the 25 models in `data/trained-models`.

To run them, first activate the Python environment previously installed:

```bash
conda activate dui-ultrafast
```

To run a specific script, simply execute `python <script_name.py>`.
For example, to run `stats_rayleigh_confidence_interval.py`:

```bash
python stats_rayleigh_confidence_interval.py
```

Note that the figures will not be identical, in terms of style,
to the corresponding ones shown in the paper but the data are identical.
Because trainings and predictions were computed on a GPU,
exact reproduction cannot be guaranteed.
This is the reason why all trained models and predictions are also provided
(see [Trained Models](#trained-models) and [Test Data](#test-data)).

## License

Code released under the terms of the [3-Clause BSD License](./LICENSE.txt).

If you are using this code and/or data,
please cite the corresponding [paper][tuffc-doi].

## Acknowledgments

This work was supported in part by the Swiss National Science
Foundation under Grant 205320_175974 and Grant 206021_170758.
