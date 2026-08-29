import numpy as np
from scipy.ndimage import label

def einstein_ring_score(theta_E, source_offset_r):
    """
    How ring-like (vs separated-arc-like) a lensing configuration is,
    based on the geometric relationship between source offset and
    Einstein radius. 1.0 = perfectly centered source, full ring.
    0.0 = source offset at or beyond the Einstein radius, fully
    separated images, no ring structure.
    """
    return max(0.0, 1.0 - source_offset_r / theta_E)


def brightness_asymmetry_score(isolated_signal):
    """
    Conselice-style asymmetry index: rotate the image 180 degrees,
    compare to the original. A real, standard technique from galaxy
    morphology literature (the 'A' in the CAS system), not an
    invented metric.

    :param isolated_signal: 2D array containing ONLY the feature being
                              measured (e.g. galaxy-subtracted arc
                              signal), with no unrelated background.
    :return: asymmetry score, roughly 0 (symmetric) to ~1+ (highly asymmetric).
    """
    rotated = np.rot90(isolated_signal, k=2)
    numerator = np.sum(np.abs(isolated_signal - rotated))
    denominator = np.sum(np.abs(isolated_signal)) + 1e-10
    return numerator / denominator

def render_two_band_signal(kwargs_lens, kwargs_lens_light, kwargs_source, lens_model_list,
                              numPix, pixel_scale, seed=None):
    """
    Render the same known lens light + source light model in two
    synthetic bands ('blue' and 'red'), with the lens galaxy assigned
    a realistically redder color and the source assigned a realistically
    bluer color - reflecting the real astronomical tendency for lens
    galaxies to be old ellipticals and lensed sources to be young,
    star-forming galaxies. The exact color offset is randomized per
    image for genuine dataset variation.

    :return: dict with 'blue_galaxy', 'red_galaxy', 'blue_source', 'red_source'
             - the four separately-rendered components needed to compute
             an isolated arc-vs-galaxy color difference.
    """
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    if seed is not None:
        np.random.seed(seed)

    lens_color_offset = np.random.uniform(0.3, 0.7)
    source_color_offset = np.random.uniform(-0.7, -0.3)

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

    kwargs_lens_light_blue = [{**kwargs_lens_light[0], 'amp': kwargs_lens_light[0]['amp'] * (1 - lens_color_offset)}]
    kwargs_lens_light_red = [{**kwargs_lens_light[0], 'amp': kwargs_lens_light[0]['amp'] * (1 + lens_color_offset)}]
    kwargs_source_blue = [{**kwargs_source[0], 'amp': kwargs_source[0]['amp'] * (1 - source_color_offset)}]
    kwargs_source_red = [{**kwargs_source[0], 'amp': kwargs_source[0]['amp'] * (1 + source_color_offset)}]

    zero_source = [{**kwargs_source[0], 'amp': 0}]
    zero_lens_light = [{**kwargs_lens_light[0], 'amp': 0}]

    blue_galaxy = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=zero_source, kwargs_lens_light=kwargs_lens_light_blue)
    red_galaxy = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=zero_source, kwargs_lens_light=kwargs_lens_light_red)
    blue_source = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source_blue, kwargs_lens_light=zero_lens_light)
    red_source = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source_red, kwargs_lens_light=zero_lens_light)

    return {'blue_galaxy': blue_galaxy, 'red_galaxy': red_galaxy,
            'blue_source': blue_source, 'red_source': red_source}


def colour_gradient_score(two_band_signals):
    """
    Measures the color difference between the arc (source) and the
    galaxy, using a log color-index style ratio - genuinely reflecting
    whether the arc's color differs meaningfully from the foreground
    galaxy's color, which is a real technique for confirming a lensing
    arc isn't just an image artifact.
    """
    galaxy_blue_flux = np.sum(two_band_signals['blue_galaxy'])
    galaxy_red_flux = np.sum(two_band_signals['red_galaxy'])
    source_blue_flux = np.sum(two_band_signals['blue_source'])
    source_red_flux = np.sum(two_band_signals['red_source'])

    galaxy_color = np.log10((galaxy_red_flux + 1e-10) / (galaxy_blue_flux + 1e-10))
    source_color = np.log10((source_red_flux + 1e-10) / (source_blue_flux + 1e-10))

    return abs(galaxy_color - source_color)


def arc_geometry_score(isolated_arc_signal, threshold_fraction=0.05):
    """
    Measures arc-like elongation, but only for genuinely connected,
    continuous shapes. Multiple disconnected blobs (e.g. separated
    multiple images, or an unrelated companion source) are flagged
    as such rather than scored as if they were a single curved arc.

    :return: (elongation_score, n_connected_components) - a fragmented
             shape (n_components > 1) should be interpreted differently
             from a genuine single continuous arc.
    """
    total_flux = np.sum(isolated_arc_signal)
    if total_flux <= 0:
        return 0.0, 0

    binary_mask = isolated_arc_signal > (threshold_fraction * isolated_arc_signal.max())
    labeled_array, n_components = label(binary_mask)

    y_coords, x_coords = np.indices(isolated_arc_signal.shape)
    x_mean = np.sum(x_coords * isolated_arc_signal) / total_flux
    y_mean = np.sum(y_coords * isolated_arc_signal) / total_flux

    dx = x_coords - x_mean
    dy = y_coords - y_mean

    mu_xx = np.sum(isolated_arc_signal * dx**2) / total_flux
    mu_yy = np.sum(isolated_arc_signal * dy**2) / total_flux
    mu_xy = np.sum(isolated_arc_signal * dx * dy) / total_flux

    cov_matrix = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])
    eigenvalues = np.clip(np.linalg.eigvalsh(cov_matrix), 1e-10, None)
    elongation = 1.0 - (eigenvalues.min() / eigenvalues.max())

    return elongation, n_components