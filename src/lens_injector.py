import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from astropy.convolution import convolve_fft


def generate_lens_signal_raw(numPix, pixel_scale, is_lens=True, seed=None):
    """
    Generate a clean, UNCONVOLVED simulated lens signal (no PSF blur
    applied). Convolve the result with a real empirical PSF separately,
    via convolve_with_psf(), for correct real-data matching.
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

    kwargs_data = {
        'pixel_scale': pixel_scale, 'exposure_time': 100, 'magnitude_zero_point': 25.0,
        'sky_brightness': 30.0, 'read_noise': 0.001, 'ccd_gain': 2.5, 'psf_type': 'NONE'
    }

    sim_api = SimAPI(numpix=numPix, kwargs_single_band=kwargs_data, kwargs_model={
        'lens_model_list': lens_model_list,
        'source_light_model_list': ['SERSIC_ELLIPSE'],
        'lens_light_model_list': ['SERSIC_ELLIPSE']
    })

    image_model = sim_api.image_model_class()
    full_signal = image_model.image(
        kwargs_lens=lens_kwargs_for_render, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light
    )

    # Also render the lens-galaxy-light-ONLY component separately, using
    # the same model setup - this is what makes exact subtraction possible
    # later, since we know precisely what we added.
    galaxy_only_signal = image_model.image(
        kwargs_lens=lens_kwargs_for_render,
        kwargs_source=[{**kwargs_source[0], 'amp': 0}],
        kwargs_lens_light=kwargs_lens_light
    )

    metadata = {
        'theta_E': theta_E, 'e1_lens': e1_lens, 'e2_lens': e2_lens,
        'source_x': source_x, 'source_y': source_y,
        'R_sersic_source': R_sersic_source, 'n_sersic_source': n_sersic_source,
        'e1_source': e1_source, 'e2_source': e2_source,
        'R_sersic_lens_light': R_sersic_lens_light, 'amp_lens_light': amp_lens_light,
    }

    return {'signal': full_signal, 'galaxy_only_signal': galaxy_only_signal,
            'label': int(is_lens), 'metadata': metadata}

def convolve_with_psf(signal, psf_kernel):
    """Convolve a raw simulated signal with a real empirical PSF kernel."""
    return convolve_fft(signal, psf_kernel, normalize_kernel=False)


def check_injection_site_clean(background_crop, sigma_threshold=5.0):
    """
    Check whether a background crop's central region is free of a
    pre-existing bright real source, before injecting a simulated lens
    there. A real source already sitting where we plan to inject would
    produce physically nonsensical overlapping-galaxy training data.

    :return: True if the center region looks clean, False if a bright
             source already dominates it.
    """
    from astropy.stats import sigma_clipped_stats
    mean, median, std = sigma_clipped_stats(background_crop, sigma=3.0)

    numPix = background_crop.shape[0]
    center = numPix // 2
    check_radius = numPix // 8
    center_region = background_crop[center - check_radius:center + check_radius,
                                      center - check_radius:center + check_radius]

    return center_region.max() < (median + sigma_threshold * std)


def normalize_image(image, low_percentile=1, high_percentile=99.5):
    """
    Normalize an image to roughly [0, 1] using percentile clipping,
    which is robust to extreme outlier pixels (bright stars, cosmic
    rays) - unlike plain min-max scaling, a single hot pixel can't
    single-handedly skew the whole normalization.
    """
    lo = np.percentile(image, low_percentile)
    hi = np.percentile(image, high_percentile)
    clipped = np.clip(image, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-10)

def inject_lens_into_background(large_background, pixel_scale, crop_size,
                                    psf_kernel, is_lens=True, seed=None,
                                    max_placement_attempts=20):
    """
    Find a clean injection site within a larger real background field,
    render a PSF-matched simulated lens signal, inject it additively,
    and provide a galaxy-subtracted residual version too.

    :param large_background: 2D array, larger than crop_size, to search
                              within for a clean injection site.
    :param pixel_scale: real background's arcsec/pixel (from its WCS).
    :param crop_size: size of the square output image.
    :param psf_kernel: real empirical PSF (e.g. from build_stacked_psf).
    :param is_lens: whether to inject a lensed or unlensed signal.
    :param seed: optional random seed.
    :param max_placement_attempts: how many random locations to try
                                     before giving up on finding a clean site.
    :return: dict with the combined image, galaxy-subtracted residual,
             normalized versions of both, label, and metadata. None if
             no clean site was found.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = large_background.shape
    background_crop = None
    for attempt in range(max_placement_attempts):
        y0 = np.random.randint(0, h - crop_size)
        x0 = np.random.randint(0, w - crop_size)
        candidate = large_background[y0:y0 + crop_size, x0:x0 + crop_size]
        if check_injection_site_clean(candidate):
            background_crop = candidate
            break

    if background_crop is None:
        return None

    result_raw = generate_lens_signal_raw(numPix=crop_size, pixel_scale=pixel_scale,
                                             is_lens=is_lens, seed=seed)
    convolved_signal = convolve_with_psf(result_raw['signal'], psf_kernel)
    convolved_galaxy_only = convolve_with_psf(result_raw['galaxy_only_signal'], psf_kernel)

    combined_image = background_crop + convolved_signal
    residual_image = combined_image - convolved_galaxy_only

    return {
        'image': combined_image,
        'image_normalized': normalize_image(combined_image),
        'residual': residual_image,
        'residual_normalized': normalize_image(residual_image),
        'label': result_raw['label'],
        'metadata': result_raw['metadata']
    }