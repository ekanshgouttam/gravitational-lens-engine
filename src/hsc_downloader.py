import os
import requests
from astropy.io import fits
from io import BytesIO


def download_hsc_cutout(ra_center, dec_center, size_arcsec, filter_band='HSC-I',
                          rerun='pdr2_dud', output_path=None):
    """
    Download a single square cutout image from the HSC-SSP public data
    release, centered on a given sky position.

    :param ra_center: right ascension of the cutout center, in degrees.
    :param dec_center: declination of the cutout center, in degrees.
    :param size_arcsec: width/height of the square cutout, in arcseconds.
    :param filter_band: HSC filter name, e.g. 'HSC-I', 'HSC-R', 'HSC-G'.
    :param rerun: HSC data release/rerun identifier.
    :param output_path: if given, saves the raw FITS bytes to this path.
    :return: the downloaded FITS data as an astropy HDUList.
    """
    username = os.environ.get('SSP_PDR_USR')
    password = os.environ.get('SSP_PDR_PWD')
    if username is None or password is None:
        raise ValueError("HSC credentials not found in environment variables "
                          "SSP_PDR_USR / SSP_PDR_PWD.")

    half_size_deg = (size_arcsec / 2.0) / 3600.0
    ra1 = ra_center - half_size_deg
    ra2 = ra_center + half_size_deg
    dec1 = dec_center - half_size_deg
    dec2 = dec_center + half_size_deg

    url = f"https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr2/cgi-bin/cutout"
    params = {
        'ra1': ra1, 'dec1': dec1,
        'ra2': ra2, 'dec2': dec2,
        'type': 'coadd', 'image': 'on', 'mask': 'off', 'variance': 'off',
        'filter': filter_band, 'rerun': rerun
    }

    response = requests.get(url, params=params, auth=(username, password), timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"HSC cutout request failed with status "
                            f"{response.status_code}: {response.text[:200]}")

    if 'fits' not in response.headers.get('Content-Type', ''):
        raise RuntimeError(f"Expected FITS data but got Content-Type "
                            f"'{response.headers.get('Content-Type')}'. "
                            f"Response preview: {response.text[:200]}")

    hdul = fits.open(BytesIO(response.content))

    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(response.content)

    return hdul