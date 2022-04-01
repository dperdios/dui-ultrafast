#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from utils.datasets import load_and_preprocess_images
from utils.signal import convert_lin_to_db
from utils.training.run_configs import get_experimental_test_managers
from utils.metrics.phantoms.cirsmodel054gshypo2 import CIRSModel054GSHypo2

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
datasets_dir = Path('./data/datasets/test')
dset_path = datasets_dir.joinpath(
    '20200527-ge9ld-experimental-test-set-cirs054gs-hypo2.hdf5')
group_key = 'images'
samples_slicer = slice(None)

# Trained model managers
trained_models_dir = Path('./data/trained-models')
model_managers = get_experimental_test_managers(trained_models_dir)

# Image cropping
zlim = 15e-3, 50e-3  # pht-hypo2

# Data ranges
vmin, vmax = clp_min, clp_max = -42, 36

# Sample index for figure
plt_ind = 0  # pht-hypo2

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

image_keys = 'lq', 'hq'
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

    # Crop
    xaxis, zaxis = image_axes
    zmin_crop, zmax_crop = zlim
    zmin_ind = np.where(zaxis < zmin_crop)[0][-1]
    zmax_ind = np.where(zaxis > zmax_crop)[0][0]
    slice_crop_z = slice(zmin_ind, zmax_ind + 1)
    slice_crop = Ellipsis, slice_crop_z
    zaxis = zaxis[slice_crop_z]
    image_axes = xaxis, zaxis
    images = images[slice_crop]

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

    # Crop
    xaxis, zaxis = image_axes
    zmin_crop, zmax_crop = zlim
    zmin_ind = np.where(zaxis < zmin_crop)[0][-1]
    zmax_ind = np.where(zaxis > zmax_crop)[0][0]
    slice_crop_z = slice(zmin_ind, zmax_ind + 1)
    slice_crop = Ellipsis, slice_crop_z
    zaxis = zaxis[slice_crop_z]
    image_axes = xaxis, zaxis
    images = images[slice_crop]

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
# Metrics: in vitro phantom
# -----------------------------------------------------------------------------
phantom = CIRSModel054GSHypo2()

results_dict = {k: {'metrics': {}} for k in env_dict.keys()}

for k, env in env_dict.items():

    # Phantom metrics
    print(f"Computing phantom metrics for '{k}'")
    metrics = phantom.compute_metrics(images=env, image_axes=image_axes)

    results_dict[k]['metrics'] = metrics

    # Display
    for k, m in metrics['inclusions'].items():
        contrast = convert_lin_to_db(m['contrast'])
        print(f"contrast-{k}: {np.mean(contrast, axis=0):+.2f}")

    print(f"SNR: {np.mean(metrics['speckle']['snr'], axis=0)}")

# Save results
with open(results_fp, mode='wb') as fp:
    pickle.dump(results_dict, file=fp, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Successfully saved '{results_fp}'.")

# -----------------------------------------------------------------------------
# Results table
# -----------------------------------------------------------------------------
table_arr = []
for k, res in results_dict.items():
    metrics = res['metrics']
    col_seq = []

    # Contrast (inclusions a, b, and c)
    for ii, mm in metrics['inclusions'].items():
        values = convert_lin_to_db(mm['contrast'])
        # col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
        col_seq.append(np.mean(values, axis=0))

    # Speckle: SNR
    values = metrics['speckle']['snr']
    # col_seq.append([np.mean(values, axis=0), np.std(values, axis=0)])
    col_seq.append(np.mean(values, axis=0))

    # Speckle: FWHM ACF lateral and axial
    values = 1e6 * metrics['speckle']['fwhm']
    # m = np.array([np.mean(values, axis=0), np.std(values, axis=0)])
    m = np.mean(values, axis=0)
    col_seq += m.T.tolist()

    # Store column
    table_arr.append(col_seq)

# Reshape array "as rows"
table_arr = np.array(table_arr)
table_arr = np.transpose(table_arr)

# Create DataFrame
table_index = [
    'C_A (dB)', 'C_B (dB)', 'C_C (dB)',
    'SNR', 'ACF lat. (µm)', 'ACF ax. (µm)',
]
col_names = ['LQ', 'HQ', 'MSLAE-16']

df = pd.DataFrame(
    data=table_arr,
    index=pd.Index(table_index),
    columns=pd.Index(col_names))

# Export as HTML
def fmt_func(x):
    if np.isnan(x):
        return '×'
    else:
        return f"{x:.2f}".replace('-', '−')

html_formatters = dict.fromkeys(list(df.columns), fmt_func)
table_fp = Path('./data/metrics/experimental-phantom-metrics.html')
df.to_html(
    table_fp,
    na_rep='×',
    formatters=html_formatters,
    classes='table table-striped text-center',
    justify='center',
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
