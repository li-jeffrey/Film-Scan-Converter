import os
import pickle

import cv2
import numpy as np
import pytest
import tifffile

from conversion import icc
from conversion.params import SPEC_DEFAULTS
from RawProcessing import RawProcessing
from tests.test_pipeline_smoke import synth_negative


@pytest.fixture
def photo(tmp_path):
    lin, _ = synth_negative(300, 450)
    # a 16-bit sRGB-encoded "scan" of the synthetic negative, with a black scanner border
    from conversion.output import linear_to_srgb
    enc = np.rint(linear_to_srgb(lin) * 65535).astype(np.uint16)
    full = np.zeros((enc.shape[0] + 80, enc.shape[1] + 80, 3), np.uint16)
    full[40:-40, 40:-40] = enc
    path = str(tmp_path / 'scan.tif')
    tifffile.imwrite(path, full, photometric='rgb')
    defaults = dict(film_type=1, dark_threshold=5, light_threshold=100, border_crop=1, flip=False, rotation=0,
                    reject=False, base_detect=0, base_rgb=None, remove_dust=False, **SPEC_DEFAULTS)
    cfg = tmp_path / 'cfg'
    cfg.mkdir()
    RawProcessing.class_parameters = RawProcessing.default_parameters.copy()
    return RawProcessing(path, defaults, defaults, str(cfg)), tmp_path


@pytest.mark.parametrize('ft', ['JPG', 'TIFF', 'PNG'])
def test_export_formats_embed_icc(photo, ft):
    p, tmp = photo
    RawProcessing.class_parameters['filetype'] = ft
    p.load(True)
    p.process(True, for_export=True)
    out = str(tmp / 'out')
    p.export(out)
    f = f'{out}.{ft}'
    assert os.path.exists(f)
    if ft == 'TIFF':
        assert icc.tiff_icc(f) is not None
    elif ft == 'PNG':
        assert icc.extract_png_iccp(open(f, 'rb').read()) is not None
    else:
        from PIL import Image
        assert Image.open(f).info.get('icc_profile')


def test_adobe_output_and_bw_gray(photo):
    p, tmp = photo
    RawProcessing.class_parameters['filetype'] = 'TIFF'
    RawProcessing.class_parameters['output_colour_space'] = 'Adobe RGB'
    p.load(True); p.process(True, for_export=True); p.export(str(tmp / 'a'))
    assert icc.tiff_icc(str(tmp / 'a.TIFF')) == icc.bundled_icc('Adobe RGB')
    p.film_type = 0
    p.process(True, for_export=True); p.export(str(tmp / 'b'))
    assert tifffile.imread(str(tmp / 'b.TIFF')).ndim == 2


def test_preview_then_export_rerenders_space(photo):
    p, tmp = photo
    RawProcessing.class_parameters['filetype'] = 'TIFF'
    RawProcessing.class_parameters['output_colour_space'] = 'Adobe RGB'
    p.process()  # sRGB preview
    p.export(str(tmp / 'c'))
    assert icc.tiff_icc(str(tmp / 'c.TIFF')) == icc.bundled_icc('Adobe RGB')


def test_pickle_small_and_roundtrip(photo):
    p, _ = photo
    p.process()
    blob = pickle.dumps(p)
    assert len(blob) < 1_000_000
    q = pickle.loads(blob)
    assert q._analysis is not None and q._lock is not None
    q.load(True); q.process(True)
    assert q.IMG.ndim == 3


def test_wb_picker_and_base_and_crop_only(photo):
    p, _ = photo
    p.process()
    p.set_wb_from_picker(0.5, 0.7)
    assert p.wb_magenta != 0.0 or p.wb_yellow != 0.0
    p.get_base_colour(0.02, 0.02)
    assert len(p.base_rgb) == 3 and all(0 <= v <= 1 for v in p.base_rgb)
    p.base_detect = 1
    p.process(skip_crop=True)
    p.film_type = 3
    p.process(skip_crop=True)
    assert p.IMG.dtype == np.uint16


def test_migration_v1(tmp_path):
    from conversion.params import migrate_v1
    d = migrate_v1(dict(film_type=1, rotation=2, sat=130, gamma=5, base_rgb=(10, 20, 30), reject=True))
    assert d['rotation'] == 2 and d['reject'] is True and d['dye_separation'] == 1.3 and d['base_rgb'] is None
    assert d['settings_version'] == 2 and 'gamma' not in d
