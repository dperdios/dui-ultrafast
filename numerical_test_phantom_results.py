#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from utils.datasets import load_and_preprocess_images
from utils.signal import convert_lin_to_db
from utils.training.run_configs import get_numerical_test_phantom_managers
from utils.metrics.standard import compare_psnr_batch, compare_ssim_batch
from utils.metrics.phantoms.numericalphantom import NumericalPhantom

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
datasets_dir = Path('./data/datasets/test')
dset_path = datasets_dir.joinpath('20200304-ge9ld-numerical-test-phantom.hdf5')
group_key = 'images'
samples_slicer = slice(None)

# Global scale factor
#  Note: scale factor calibrated on the linear gradient. This factor was also
#  applied before inference (in `compute_predictions.py`)
scale_factor = 3

# Trained model managers
trained_models_dir = Path('./data/trained-models')
model_managers = get_numerical_test_phantom_managers(trained_models_dir)

# Data ranges
vmin, vmax = clp_min, clp_max = -62, 36

# Sample index for figure
plt_ind = 0

# Save
save_dir = Path("./data/metrics")
if not save_dir.is_dir():
    raise NotADirectoryError(f"Directory '{save_dir}' does not exist.")

results_fn = dset_path.stem + '-results'
results_fp = save_dir.joinpath(results_fn).with_suffix('.pickle')
if results_fp.is_file():
    input_str_seq = [
        f"File '{results_fp}' already exists and will be overwritten.\n"
        f"Do you want to continue (y / [n]): "
    ]
    input_str = '\n'.join(input_str_seq)
    usr_answer = input(input_str).lower() or 'n'
    if usr_answer != 'y':
        raise InterruptedError()

# -----------------------------------------------------------------------------
# Load images: inputs and targets
# -----------------------------------------------------------------------------
env_dict = {}

image_keys = 'lq', 'hq', 'uq'
input_signal = 'iq'
output_signal = 'env'

dset_path = dset_path.resolve(strict=True)

# Load images
for k in image_keys:
    # Build dataset key
    dset_name = '/'.join([group_key, k])
    print(f"Loading '{dset_name}' from '{dset_path}'")

    # Load images as envelope to compute metrics
    images, image_axes = load_and_preprocess_images(
        path=dset_path,
        name=dset_name,
        input_signal=input_signal,
        input_factor='0db',
        output_signal=output_signal,
        samples_slicer=samples_slicer,
    )

    # Additional scale factor specific to numerical test phantom
    images *= scale_factor

    # Store
    env_dict[k] = np.copy(images)

# -----------------------------------------------------------------------------
# Load images: predictions
# -----------------------------------------------------------------------------
predictions_dir = datasets_dir
pred_suffix = 'predictions'
pred_path = predictions_dir.joinpath(dset_path.stem + '-' + pred_suffix)
pred_path = pred_path.with_suffix(dset_path.suffix)
pred_path = pred_path.resolve(strict=True)

# Load images
pred_dict = {}
dset_path = pred_path
for k, mgr in model_managers.items():
    # Build dataset key
    dset_name = '/'.join([group_key, k])
    print(f"Loading '{dset_name}' from '{dset_path}'")

    # Convert prediction signal (CNN output) to envelope for computing metrics
    input_signal = mgr.run_config.mapping_config.output_signal
    images, image_axes = load_and_preprocess_images(
        path=dset_path,
        name=dset_name,
        input_signal=input_signal,
        output_signal=output_signal,
        samples_slicer=samples_slicer
    )

    # Store
    env_dict[k] = images

# -----------------------------------------------------------------------------
# B-mode conversions for metrics and display
# -----------------------------------------------------------------------------
bm_dict = {}
for k, env in env_dict.items():
    eps = np.spacing(1, dtype=env.dtype)
    bm_dict[k] = convert_lin_to_db(env, x_min=eps)

# -----------------------------------------------------------------------------
# Compute results for metrics and figures
# -----------------------------------------------------------------------------
results_dict = {}

# Phantom metrics helper
phantom = NumericalPhantom()

# B-mode clip settings and references
clp_kwargs = {'a_min': clp_min, 'a_max': clp_max}
clp_range = clp_max - clp_min
bm_ref_clp = np.clip(bm_dict['uq'], **clp_kwargs)

for k in env_dict.keys():

    # Extract envelope and B-mode images
    env = env_dict[k]
    bm = bm_dict[k]

    # Phantom metrics
    print(f"Computing phantom metrics for '{k}'")
    metrics = phantom.compute_metrics(images=env, image_axes=image_axes)

    # B-mode image comparison metrics
    print(f"Computing image comparison metrics for '{k}'")
    bm_clp = np.clip(bm, **clp_kwargs)
    bm_psnr = compare_psnr_batch(
        true=bm_ref_clp, test=bm_clp, data_range=clp_range
    )
    bm_ssim = compare_ssim_batch(
        true=bm_ref_clp, test=bm_clp, data_range=clp_range
    )
    image_metrics = {'psnr': bm_psnr, 'ssim': bm_ssim}
    metrics['image'] = image_metrics

    # Results dict
    res = {
        'metrics': metrics,
        'figures': {
            'bmode': bm[plt_ind],
            'bmode_mean': np.mean(bm, axis=0),
            'bmode_axes': image_axes,
            'vmin': clp_min,
            'vmax': clp_max,
        }
    }

    # Store results globally
    results_dict[k] = res

# Save results
with open(results_fp, mode='wb') as fp:
    pickle.dump(results_dict, file=fp, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Successfully saved '{results_fp}'.")

# -----------------------------------------------------------------------------
# Results table
# -----------------------------------------------------------------------------
# Construct table data array (mean and std) with unit transformations
table_arr = []
for k, res in results_dict.items():
    metrics = res['metrics']
    col_seq = []
    # Contrast
    values = convert_lin_to_db(metrics['inclusion']['contrast'])
    col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # Artifact: GL, SL, EW
    for ak in list(metrics['artifacts']):
        values = convert_lin_to_db(metrics['artifacts'][ak])
        col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # ----
    # Speckle: SNR
    values = metrics['speckle']['snr']
    col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # Speckle: FWHM ACF lateral and axial
    values = 1e6 * metrics['speckle']['fwhm']
    m = np.array([np.mean(values, axis=0), np.std(values, axis=0)])
    col_seq += m.T.tolist()
    # ----
    # FWHM lateral: p0, p1, p2, p3
    for pk in list(metrics['resolution']):
        fwhm = 1e6 * metrics['resolution'][pk]['fwhm']
        values = fwhm[:, 0]
        col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # ----
    # FWHM axial: p0, p1, p2, p3
    for pk in list(metrics['resolution']):
        fwhm = 1e6 * metrics['resolution'][pk]['fwhm']
        values = fwhm[:, 1]
        col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # ----
    # Image metrics: PSNR
    values = metrics['image']['psnr']
    col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # Image metrics: SSIM
    values = metrics['image']['ssim']
    col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    # ----
    # Store column
    table_arr.append(col_seq)

# Reshape array
table_arr = np.array(table_arr)
table_arr = np.swapaxes(table_arr, axis1=0, axis2=1)
table_arr = np.reshape(table_arr, newshape=(table_arr.shape[0], -1))

# Assign NaNs to image metrics computed on itself (i.e., UQ case)
table_arr[15:17, 4:6] = np.nan

# Create DataFrame
table_index = [
    'C (dB)', 'GL (dB)', 'SL (dB)', 'EW (dB)',
    'SNR', 'ACF lat. (µm)', 'ACF ax. (µm)',
    'FWHM lat. p0 (µm)', 'FWHM lat. p1 (µm)', 'FWHM lat. p2 (µm)', 'FWHM lat. p3 (µm)',
    'FWHM ax. p0 (µm)', 'FWHM ax. p1 (µm)', 'FWHM ax. p2 (µm)', 'FWHM ax. p3 (µm)',
    'PSNR (dB)', 'SSIM'
]
col_names = ['LQ', 'HQ', 'UQ', 'MSE-16', 'MAE-16', 'MSLAE-16', 'MSLAE-32']
sub_col_names = ['mean', 'std']

df = pd.DataFrame(
    data=table_arr,
    index=pd.Index(table_index),
    columns=pd.MultiIndex.from_product([col_names, sub_col_names]))


# Export as HTML
def fmt_func(x):
    if np.isnan(x):
        return '×'
    else:
        return f"{x:.2f}".replace('-', '−')


html_formatters = dict.fromkeys(list(df.columns), fmt_func)
table_fp = Path('./data/metrics/numerical-phantom-metrics.html')
df.to_html(
    table_fp,
    na_rep='×',
    formatters=html_formatters,
    classes='table table-striped text-center',
    bold_rows=True,
    border=0)

# Append minimal bootstrap styles
html_head = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">'
html_bot = '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.min.js"></script'
with open(table_fp, 'r') as fr:
    html_table = fr.read()
html_table = '\n'.join([html_head, html_table, html_bot])
with open(table_fp, 'w') as fw:
    fw.write(html_table)

print(f"Successfully exported '{table_fp}'.")
