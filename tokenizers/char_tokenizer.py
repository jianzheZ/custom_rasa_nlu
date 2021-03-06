from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import os
import glob


class CharTokenizer(Tokenizer, Component):
    
    
    name = "tokenizer_char"

    provides = ["tokens"]

    language_list = ["zh"]

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 tokenizer=None
                 ):
        # type: (...) -> None
        
        super(CharTokenizer, self).__init__(component_config)

        self.tokenizer = tokenizer


    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> CharTokenizer
        
        component_conf = cfg.for_component(cls.name, cls.defaults)
                        
        return CharTokenizer(component_conf)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CharTokenizer
                

        component_meta = model_metadata.for_component(cls.name)

        return CharTokenizer(component_meta)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return []


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
            
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        
        message.set("tokens", self.tokenize(message.text))


    def tokenize(self, text):
        """
        Tokenize a sentence and yields tuples of (word, start, end)
        type: (Text) -> List[Token]
        Parameter:
            - text: the str(unicode) to be segmented.
        """
        tokens = []
        start = 0
        for char in list(text):
            print(char)
            #yield (w, start, start + width)
            tokens.append(Token(char, start))
            start += 1
        return tokens