import numpy as np

from cardiff_converter_opt.parameters import DesignParameters
from cardiff_converter_opt.physics import total_loss, total_volume_m3


def test_volume_reference_point():
    p = DesignParameters()
    a = p.area_swref
    f = 150e3
    ir = 1.0
    v = total_volume_m3(p, a, f, ir)
    assert 180e-6 < v < 230e-6  # ~200 cm³ ballpark


def test_loss_positive():
    p = DesignParameters()
    pl = total_loss(p, p.area_swref, 150e3, 1.0, 50.0)
    assert 0 < pl < 500
