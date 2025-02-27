# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import io
from modelloader.base_model_loader import BaseModelLoader


def yolo_extract(result):
    """
    Function that extracts the bounding boxes, confidence scores, and class labels
    from YOLO v5 and YOLO v8 Ultralytics models.

    Parameters:
        result (object): The result object containing the predictions.

    Returns:
        tuple: A tuple containing the bounding boxes, confidence scores, and class labels.
            - boundingBox (list): A list of bounding boxes in the format [x1, y1, x2, y2].
            - cs (list): A list of confidence scores.
            - cl (list): A list of class labels.
    """
    boundingBox = result.xyxyn[0].numpy().tolist()
    cs = result.xyxy[0][:, 4].numpy().tolist()
    cl = result.xyxy[0][:, 5].numpy().tolist()
    return boundingBox, cs, cl


class CocoObjectDetection_Y5(BaseModelLoader):
    def __init__(self, config, model_name):
        """Initialize the Yolov8 modelloader loader.

        Args:
            config (dict): Configuration dictionary.
            model_name (str): Name of the modelloader.

        """
        super().__init__(config, model_name)
        import torch
        from PIL import Image
        self.PilImage = Image
        self.model_path = None if config[model_name]['model_path'][0] == '' else config[model_name]['model_path'][0]
        self.device = config[model_name].get("device")
        self.model_obj = torch.hub.load("ultralytics/yolov5",
                                        "custom",
                                        self.model_path,
                                        device=self.device)
        self.model_obj.conf = config[model_name].get('conf_thres', 0.25)
        self.model_obj.iou = config[model_name].get('iou_thres', 0.7)
        self.classes = [i for i in range(len(self.model_obj.names))] if len(
            config[model_name].get("classes")) == 0 else config[model_name].get("classes")


    def predict(self, Base_64, Cs):
        """Predict the result using the modelloader.

        Args:
            Base_64 (str): Base64 encoded image.
            Cs (float): Confidence threshold.

        Returns:
            list: List of dictionaries containing the predicted results.

        """
        image = self.PilImage.open(io.BytesIO(base64.b64decode(Base_64)))
        result = self.model_obj(image, size=640)
        boundingBox, cs, cl = yolo_extract(result)
        output = []
        for i in range(len(boundingBox)):
            if cs[i] >= float(Cs) and int(cl[i]) in self.classes:
                output.append({
                    "Lb": self.model_obj.names[int(cl[i])],
                    "Cs": cs[i],
                    "Dm": {
                        "X": boundingBox[i][0],
                        "Y": boundingBox[i][1],
                        "W": boundingBox[i][2] - boundingBox[i][0],
                        "H": boundingBox[i][3] - boundingBox[i][1]}})
        return output