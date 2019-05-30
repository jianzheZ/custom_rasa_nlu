from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import warnings
import numpy as np

import typing
from typing import Any, Dict, List, Optional, Text

from rasa_nlu_gao import utils
from rasa_nlu_gao.config import RasaNLUModelConfig
from rasa_nlu_gao.featurizers import Featurizer
from rasa_nlu_gao.training_data import Message
from rasa_nlu_gao.training_data import TrainingData

import numpy as np

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu_gao.model import Metadata


PSEQ_FEATURIZER_FILE_NAME = "pseg_featurizer.json"


class PseqFeaturizer(Featurizer):
    name = "hanlp_pseq_featurizer"

    provides = ["text_features"]

    requires = ["tokens"]

    def __init__(self, component_config=None):
        self.pseq_list = ["a","ad","ag","al","an","b","bg","bl","c","cc","d","dg","dl","e","f","g","gb","gbc","gc","gg","gi","gm","gp","h","i","j","k","l","m","mg","Mg","mq","n","nb.nba","nbc","nbp","nf","ng","nh","nhd","nhm","ni","nic","nis","nit","nl","nm","nmc","nn","nnd","nnt","nr","nr1","nr2","nrf","nrj","ns","nsf","nt","ntc","ntcb","ntcf","ntch","nth","nto","nts","ntu","nx","nz","o","p","pba","pbei","q","qg","qt","qv","r","rg","Rg","rr","ry","rys","ryt","ryv","rz","rzs","rzt","rzv","s","t","tg","u","ud","ude1","ude2","ude3","udeng","udh","ug","uguo","uj","ul","ule","ulian","uls","usuo","uv","uyy","uz","uzhe","uzhi","v","vd","vf","vg","vi","vl","vn","vshi","vx","vyou","w","wb","wd","wf","wh","wj","wky","wkz","wm","wn","wp","ws","wt","ww","wyy","wyz","x","xu","xx","y","yg","z","zg"]
        super(PseqFeaturizer, self).__init__(component_config)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            updated = self._text_features_with_pseq(example)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_pseq(message)
        message.set("text_features", updated)

    def _text_features_with_pseq(self, message):
        extras = self.features_for_pseq(message)
        return self._combine_with_existing_text_features(message, extras)

    def features_for_pseq(self, message):
        """Checks which hanlp pseq match the message.

        Given a sentence, returns a vector of {1,0} values indicating which
        pseqes did match. Furthermore, if the
        message is tokenized, the function will mark all tokens with a dict
        relating the name of the pseq to whether it was matched."""
        match = []
        for token_index, t in enumerate(self.pseq_list):
            if t in message.get("pseqs", []):
                match.append(1)
            else:
                match.append(0)
        found = [1.0 if m is not None else 0.0 for m in match]
        print(len(found))
        return np.array(found)
