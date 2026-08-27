import numpy as np
import h5py
import pandas as pd
import os
from tqdm import tqdm
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI


def generate_lens(is_lens=True, numPix=100, deltaPix=0.05, seed=None):
    """
    Generate a single simulated galaxy scene, either a true gravitational
    lens or a non-lensed pair of galaxies, using randomized physical parameters.

    :param is_lens: if True, applies SIE gravitational lensing to the source.
                     if False, renders the same kind of scene with no lensing applied.
    :param numPix: image size in pixels (square).
    :param deltaPix: pixel scale in arcsec/pixel.
    :param seed: optional random seed, for reproducible generation.
    :return: dict containing the image array, the label, and all physical
             parameters used to generate it.
    """
    if seed is not None:
        np.random.seed(seed)

        # Lens mass model (SIE) — always present, since even "non-lens" images
    # still have a real foreground galaxy with real mass; we just won't
    # apply its lensing effect to the source when is_lens=False.
    theta_E = np.random.uniform(0.8, 2.0)
    e1_lens = np.random.uniform(-0.2, 0.2)
    e2_lens = np.random.uniform(-0.2, 0.2)

    kwargs_lens = [{
        'theta_E': theta_E,
        'e1': e1_lens,
        'e2': e2_lens,
        'center_x': 0.0,
        'center_y': 0.0
    }]

        # Source light model (Sersic) — the background galaxy being lensed.
    # A small random offset from the lens center is what produces
    # asymmetric arcs rather than a perfectly uniform ring.
    source_offset_r = np.random.uniform(0.3, 0.8)
    source_offset_angle = np.random.uniform(0, 2 * np.pi)
    source_x = source_offset_r * np.cos(source_offset_angle)
    source_y = source_offset_r * np.sin(source_offset_angle)
    source_amp = np.random.uniform(50, 150)

    R_sersic_source = np.random.uniform(0.1, 0.4)
    n_sersic_source = np.random.uniform(1.0, 4.0)
    e1_source = np.random.uniform(-0.3, 0.3)
    e2_source = np.random.uniform(-0.3, 0.3)

    kwargs_source = [{
        'amp': source_amp,
        'R_sersic': R_sersic_source,
        'n_sersic': n_sersic_source,
        'e1': e1_source,
        'e2': e2_source,
        'center_x': source_x,
        'center_y': source_y
    }]

        # Lens light model (Sersic) — the visible foreground galaxy itself.
    # Present in every image, lens or not, since a real foreground galaxy
    # doesn't stop existing just because it isn't lensing anything.
    R_sersic_lens_light = np.random.uniform(0.5, 1.2)
    amp_lens_light = np.random.uniform(200, 500)

    kwargs_lens_light = [{
        'amp': amp_lens_light,
        'R_sersic': R_sersic_lens_light,
        'n_sersic': 4,
        'e1': e1_lens,
        'e2': e2_lens,
        'center_x': 0.0,
        'center_y': 0.0
    }]

        # The core of the lens/non-lens distinction: same source, same lens
    # galaxy, but lensing is only applied when is_lens=True.
    if is_lens:
        lens_model_list = ['SIE']
        lens_kwargs_for_render = kwargs_lens
    else:
        lens_model_list = []
        lens_kwargs_for_render = []

    source_model_list = ['SERSIC_ELLIPSE']
    lens_light_model_list = ['SERSIC_ELLIPSE']

    kwargs_data = {
        'pixel_scale': deltaPix,
        'exposure_time': 100,
        'magnitude_zero_point': 25.0,
        'sky_brightness': 22.0,
        'read_noise': 4,
        'ccd_gain': 2.5,
        'seeing': 0.1,
        'psf_type': 'GAUSSIAN'
    }

    sim_api = SimAPI(numpix=numPix, kwargs_single_band=kwargs_data, kwargs_model={
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list
    })

    image_model = sim_api.image_model_class()
    image = image_model.image(
        kwargs_lens=lens_kwargs_for_render,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light
    )

    metadata = {
        'theta_E': theta_E,
        'e1_lens': e1_lens,
        'e2_lens': e2_lens,
        'source_x': source_x,
        'source_y': source_y,
        'R_sersic_source': R_sersic_source,
        'n_sersic_source': n_sersic_source,
        'e1_source': e1_source,
        'e2_source': e2_source,
        'R_sersic_lens_light': R_sersic_lens_light,
        'amp_lens_light': amp_lens_light,
    }

    return {
        'image': image,
        'label': int(is_lens),
        'metadata': metadata
    }

#Generate dataset: 4000 images, HDF5 archive files

def generate_dataset(n_lens, n_nonlens, output_dir, numPix=100, deltaPix=0.05, seed_start=0):
    """
    Generate a full dataset of simulated lens and non-lens images, saving
    all images into a single HDF5 file and all per-image metadata into
    a separate Parquet file.

    :param n_lens: number of lens images to generate.
    :param n_nonlens: number of non-lens images to generate.
    :param output_dir: directory to save the dataset files into.
    :param numPix: image size in pixels.
    :param deltaPix: pixel scale in arcsec/pixel.
    :param seed_start: starting seed value; each image gets a unique
                        incrementing seed from this point, so the whole
                        dataset is reproducible from a single number.
    :return: path to the saved HDF5 file and Parquet file.
    """
    n_total = n_lens + n_nonlens
    images = np.zeros((n_total, numPix, numPix), dtype=np.float32)
    metadata_rows = []

    idx = 0
    for i in tqdm(range(n_lens), desc="Generating lens images"):
        result = generate_lens(is_lens=True, numPix=numPix, deltaPix=deltaPix, seed=seed_start + idx)
        images[idx] = result['image']
        row = {'index': idx, 'label': result['label'], 'seed': seed_start + idx}
        row.update(result['metadata'])
        metadata_rows.append(row)
        idx += 1

    for i in tqdm(range(n_nonlens), desc="Generating non-lens images"):
        result = generate_lens(is_lens=False, numPix=numPix, deltaPix=deltaPix, seed=seed_start + idx)
        images[idx] = result['image']
        row = {'index': idx, 'label': result['label'], 'seed': seed_start + idx}
        row.update(result['metadata'])
        metadata_rows.append(row)
        idx += 1

    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, 'lens_dataset.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip', compression_opts=4)

    metadata_df = pd.DataFrame(metadata_rows)
    parquet_path = os.path.join(output_dir, 'lens_dataset_metadata.parquet')
    metadata_df.to_parquet(parquet_path, index=False)

    print(f"Saved {n_total} images to {h5_path}")
    print(f"Saved metadata ({len(metadata_df)} rows) to {parquet_path}")
    print(f"Label distribution:\n{metadata_df['label'].value_counts()}")

    return h5_path, parquet_path