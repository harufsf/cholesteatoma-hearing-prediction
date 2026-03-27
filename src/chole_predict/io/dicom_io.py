from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


DEFAULT_ISO_MM = 0.5

def load_dicom_series_to_hu_zyx(dicom_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM dir not found: {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()
    sx, sy, sz = img.GetSpacing()  # SITK: (x,y,z)
    vol_zyx = sitk.GetArrayFromImage(img).astype(np.int16, copy=False)  # numpy: ZYX
    spacing_zyx = (float(sz), float(sy), float(sx))
    return vol_zyx, spacing_zyx

def load_dicom_volume_normalized(dicom_dir: str, iso_spacing_mm=None):
    """
    Backward-compatible wrapper for sphere ROI generation.

    Parameters
    ----------
    dicom_dir : str
        DICOM directory.
    iso_spacing_mm : float or None
        Optional isotropic spacing in mm.

    Returns
    -------
    vol_zyx : np.ndarray
        CT volume in ZYX order.
    spacing_zyx : tuple[float, float, float]
        Voxel spacing in ZYX order.
    """
    vol_zyx, spacing_zyx = load_dicom_series_to_hu_zyx(dicom_dir)

    if iso_spacing_mm is not None:
        vol_zyx = resample_to_iso(vol_zyx, spacing_zyx, float(iso_spacing_mm))
        spacing_zyx = (float(iso_spacing_mm), float(iso_spacing_mm), float(iso_spacing_mm))

    return vol_zyx, spacing_zyx

def resample_to_iso(vol_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float], iso_mm: float) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    zf = (sz / iso_mm, sy / iso_mm, sx / iso_mm)
    if all(abs(a - 1.0) < 1e-6 for a in zf):
        return vol_zyx.astype(np.float32)
    return zoom(vol_zyx.astype(np.float32), zf, order=1).astype(np.float32)
