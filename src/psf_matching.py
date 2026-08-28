import numpy as np
from scipy.ndimage import shift
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats


def find_star_candidates(image, fwhm=5.0, threshold_sigma=10.0):
    """
    Detect likely point-source (star) candidates in a real background image.

    :param image: 2D numpy array of the background image.
    :param fwhm: rough initial guess for point-source width, in pixels.
    :param threshold_sigma: detection threshold, in multiples of background std.
    :return: astropy Table of detected sources, sorted brightest first.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(image - median)
    if sources is not None:
        sources.sort('flux', reverse=True)
    return sources, median


def _extract_and_align_stamp(image, x_center, y_center, half_size=15):
    """
    Extract a square stamp centered on (x_center, y_center), correcting
    for the sub-pixel fractional offset via interpolated shifting.
    """
    x_int, y_int = int(round(x_center)), int(round(y_center))
    stamp = image[y_int - half_size - 2:y_int + half_size + 3,
                   x_int - half_size - 2:x_int + half_size + 3].astype(np.float64)

    frac_x = x_center - x_int
    frac_y = y_center - y_int

    aligned = shift(stamp, shift=(-frac_y, -frac_x), order=3, mode='nearest')
    return aligned[2:-2, 2:-2]


def build_stacked_psf(image, star_coords, half_size=15):
    """
    Build a smooth empirical PSF by stacking multiple real, isolated stars.
    Averaging cancels individual stars' photon noise while reinforcing
    their shared true PSF shape - more reliable than fitting or using a
    single star, which are both prone to noise-dominated outer wings.

    :param image: 2D numpy array of the real background image the stars
                  were detected in.
    :param star_coords: list of (x, y) pixel coordinate tuples for
                         confirmed clean, isolated stars. Must be visually
                         verified (e.g. via log-stretch plot) beforehand -
                         this function does not check for contamination.
    :param half_size: half-width of the stamp in pixels; final kernel
                       size is (2*half_size + 1) square.
    :return: normalized 2D numpy array (sums to 1.0), usable directly as
             a convolution kernel.
    """
    bg_subtracted_stamps = []
    for x_center, y_center in star_coords:
        stamp = _extract_and_align_stamp(image, x_center, y_center, half_size=half_size)

        local_bg = np.median(
            stamp[:3, :].flatten().tolist() + stamp[-3:, :].flatten().tolist() +
            stamp[:, :3].flatten().tolist() + stamp[:, -3:].flatten().tolist()
        )
        bg_subtracted_stamps.append(stamp - local_bg)

    stacked = np.mean(bg_subtracted_stamps, axis=0)
    stacked[stacked < 0] = 0
    stacked_normalized = stacked / stacked.sum()

    return stacked_normalized