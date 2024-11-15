# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import json
from modelloader.base_model_loader import BaseModelLoader


class CocoObjectDetection_De(BaseModelLoader):
    def __init__(self, config, model_name):
        """
        Initialize the Detecto model loader.

        Args:
            config (dict): The configuration dictionary.
            model_name (str): The name of the modelloader.

        """
        from detectoinference import Detecto
        model_path = None if len(config[model_name]['model_path']) == 0\
            else config[model_name]['model_path']
        name = None if config[model_name]['name'] == "" else config[model_name]['name']
        classes=None if len(config[model_name]['classes']) == 0 else config[model_name]['classes']
        logger=config[model_name].get('logger')
        self.model_obj = Detecto(model_weight_file=model_path if model_path is None else model_path[0],
                                 model_name=name,
                                 classes=classes,
                                 logger=logger)