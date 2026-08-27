import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI


def generate_lens_signal(numPix, pixel_scale, is_lens=True, seed=None):
    """
    Generate a clean (near-zero-noise) simulated lens signal, sized and
    scaled to match a real background image, for additive injection.

    :param numPix: image size in pixels, matching the target background crop.
    :param pixel_scale: arcsec/pixel, matching the target background's real WCS scale.
    :param is_lens: whether to apply gravitational lensing (True) or not (False).
    :param seed: optional random seed for reproducibility.
    :return: dict with the clean signal image and metadata (same physical
             parameter fields as generate_lens(), for consistency).
    """
    if seed is not None:
        np.random.seed(seed)

    theta_E = np.random.uniform(0.8, 2.0)
    e1_lens = np.random.uniform(-0.2, 0.2)
    e2_lens = np.random.uniform(-0.2, 0.2)

    source_offset_r = np.random.uniform(0.3, 0.8)
    source_offset_angle = np.random.uniform(0, 2 * np.pi)
    source_x = source_offset_r * np.cos(source_offset_angle)
    source_y = source_offset_r * np.sin(source_offset_angle)

    source_amp = np.random.uniform(50, 150)
    R_sersic_source = np.random.uniform(0.1, 0.4)
    n_sersic_source = np.random.uniform(1.0, 4.0)
    e1_source = np.random.uniform(-0.3, 0.3)
    e2_source = np.random.uniform(-0.3, 0.3)

    R_sersic_lens_light = np.random.uniform(0.5, 1.2)
    amp_lens_light = np.random.uniform(200, 500)

    if is_lens:
        lens_model_list = ['SIE']
        lens_kwargs_for_render = [{
            'theta_E': theta_E, 'e1': e1_lens, 'e2': e2_lens,
            'center_x': 0.0, 'center_y': 0.0
        }]
    else:
        lens_model_list = []
        lens_kwargs_for_render = []

    kwargs_source = [{
        'amp': source_amp, 'R_sersic': R_sersic_source, 'n_sersic': n_sersic_source,
        'e1': e1_source, 'e2': e2_source, 'center_x': source_x, 'center_y': source_y
    }]
    kwargs_lens_light = [{
        'amp': amp_lens_light, 'R_sersic': R_sersic_lens_light, 'n_sersic': 4,
        'e1': e1_lens, 'e2': e2_lens, 'center_x': 0.0, 'center_y': 0.0
    }]

    # Deliberately near-zero background noise: the REAL background image
    # supplies genuine noise later, at injection time. Adding our own
    # simulated noise here would double-count it.
    kwargs_data = {
        'pixel_scale': pixel_scale,
        'exposure_time': 100,
        'magnitude_zero_point': 25.0,
        'sky_brightness': 30.0,
        'read_noise': 0.001,
        'ccd_gain': 2.5,
        'seeing': 0.1,
        'psf_type': 'GAUSSIAN'
    }

    sim_api = SimAPI(numpix=numPix, kwargs_single_band=kwargs_data, kwargs_model={
        'lens_model_list': lens_model_list,
        'source_light_model_list': ['SERSIC_ELLIPSE'],
        'lens_light_model_list': ['SERSIC_ELLIPSE']
    })

    image_model = sim_api.image_model_class()
    signal = image_model.image(
        kwargs_lens=lens_kwargs_for_render,
        kwargs_source=kwargs_source,
        kwargs_lens_light=kwargs_lens_light
    )

    metadata = {
        'theta_E': theta_E, 'e1_lens': e1_lens, 'e2_lens': e2_lens,
        'source_x': source_x, 'source_y': source_y,
        'R_sersic_source': R_sersic_source, 'n_sersic_source': n_sersic_source,
        'e1_source': e1_source, 'e2_source': e2_source,
        'R_sersic_lens_light': R_sersic_lens_light, 'amp_lens_light': amp_lens_light,
    }

    return {'signal': signal, 'label': int(is_lens), 'metadata': metadata}

def inject_lens_into_background(background_image, pixel_scale, is_lens=True, seed=None):
    """
    Inject a simulated lens signal into a real background image.

    :param background_image: 2D numpy array of real sky background data.
                              Will be cropped to a square if not already one.
    :param pixel_scale: arcsec/pixel of the background image (from its real WCS).
    :param is_lens: whether to inject a lensed or unlensed signal.
    :param seed: optional random seed.
    :return: dict with the combined image, label, and metadata.
    """
    numPix = min(background_image.shape)
    background_square = background_image[:numPix, :numPix]

    result = generate_lens_signal(numPix=numPix, pixel_scale=pixel_scale,
                                    is_lens=is_lens, seed=seed)

    combined_image = background_square + result['signal']

    return {
        'image': combined_image,
        'label': result['label'],
        'metadata': result['metadata']
    }