# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import io
from modelloader.base_model_loader import BaseModelLoader


class Yolo_Ultralytics(BaseModelLoader):
    def __init__(self, config, model_name):
        """Initialize the Yolo Ultralytics modelloader loader.

        Args:
            config (dict): Configuration dictionary.
            model_name (str): Name of the modelloader.

        """
        super().__init__(config, model_name)
        from ultralytics import YOLO
        from PIL import Image
        self.PilImage = Image
        self.config = config
        self.model_path = None if config[model_name]['model_path'][0] == '' else config[model_name]['model_path'][0]
        self.device = config[model_name].get("device")
        self.model_obj = YOLO(self.model_path)
        self.model_obj.conf = config[model_name].get('conf_thres', 0.25)
        self.model_obj.iou = config[model_name].get('iou_thres', 0.7)
        self.classes = None if len(config[model_name].get("classes")) == 0 else config[model_name].get("classes")

    def predict(self, Base_64, Cs):
        """Predict the result using the modelloader.

        Args:
            Base_64 (str): Base64 encoded image.
            Cs (float): Confidence threshold.

        Returns:
            list: List of dictionaries containing the predicted results.

        """
        image = self.PilImage.open(io.BytesIO(base64.b64decode(Base_64)))
        result = self.model_obj.predict(image, classes=self.classes, conf=Cs)
        # boundingBox, cs, cl = yolo_extract(result[0])
        boundingBox = result[0].boxes.xywhn.numpy().tolist()
        cs = result[0].boxes.conf.numpy().tolist()
        cl = result[0].boxes.cls.numpy().tolist()
        output = []
        for i in range(len(result[0])):
            if cs[i] >= Cs:
                output.append({
                    "Lb": self.model_obj.names[int(cl[i])],
                    "Cs": cs[i],
                    "Dm": {
                        "X": boundingBox[i][0],
                        "Y": boundingBox[i][1],
                        "H": boundingBox[i][2] + boundingBox[i][0],
                        "W": boundingBox[i][3] + boundingBox[i][1]}})
        return output