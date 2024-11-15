# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from modelloader.base_model_loader import BaseModelLoader


class FacialExpressionRecognition(BaseModelLoader):
    """
            Initialize the Facial Expression Recognition model loader.

            Args:
                config (dict): The configuration dictionary.
                model_name (str): The name of the modelloader.

            """
    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        from ferinference import FacialExpressionRecognition
        self.model_obj = FacialExpressionRecognition()