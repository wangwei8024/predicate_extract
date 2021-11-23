from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def setup(opt):
    from .motifs import MotifPredictor
    model = MotifPredictor(opt)

    return model
