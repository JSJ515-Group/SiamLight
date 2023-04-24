from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamlight.core.config import cfg
from siamlight.tracker.siam_tracker import SiamTracker

TRACKS = {
          'SiamTracker': SiamTracker
         } 

def build_tracker(model): 
    return TRACKS[cfg.TRACK.TYPE](model) 
