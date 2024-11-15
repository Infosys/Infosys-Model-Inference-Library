# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import io
import time

from milutils.dataset_class_names import imagenet_class_names
from modelloader.base_model_loader import BaseModelLoader


class PyTorch(BaseModelLoader):
    def __init__(self, config, model_name):
        """Initialize the PyTorch model loader.

        Args:
            config (dict): Configuration dictionary containing modelloader settings.
            model_name (str): Name of the modelloader.

        """
        import torch
        from torchvision.transforms import transforms
        from torchvision import models
        from PIL import Image

        self.pytorch_models_dict = {
            "fasterrcnn_resnet50_fpn_v2": models.detection.fasterrcnn_resnet50_fpn_v2,
            "fasterrcnn_resnet50_fpn_v2_weights": models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn,
            "fasterrcnn_resnet50_fpn_weights": models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            "fasterrcnn_mobilenet_v3_large_fpn": models.detection.fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_mobilenet_v3_large_fpn_weights": models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            "fasterrcnn_mobilenet_v3_large_320_fpn": models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            "fasterrcnn_mobilenet_v3_large_320_fpn_weights": models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            "retinanet_resnet50_fpn": models.detection.retinanet_resnet50_fpn,
            "retinanet_resnet50_fpn_weights": models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT,
            "retinanet_resnet50_fpn_v2": models.detection.retinanet_resnet50_fpn_v2,
            "retinanet_resnet50_fpn_v2_weights": models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            "fcos_resnet50_fpn": models.detection.fcos_resnet50_fpn,
            "fcos_resnet50_fpn_weights": models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT,
            "ssd300_vgg16": models.detection.ssd300_vgg16,
            "ssd300_vgg16_weights": models.detection.SSD300_VGG16_Weights.DEFAULT,
            "ssdlite320_mobilenet_v3_large": models.detection.ssdlite320_mobilenet_v3_large,
            "ssdlite320_mobilenet_v3_large_weights": models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        }

        self.torch = torch
        self.transforms = transforms
        self.PilImage = Image
        self.models = models
        self.model_path = None if config[model_name]['model_path'][0] == '' else config[model_name]['model_path'][0]
        self.name = config[model_name]['name']
        if self.name in self.pytorch_models_dict:
            self.pytorch_model_loader = self.pytorch_models_dict[self.name]
        classes = config[model_name].get("classes")
        category_fuction_map = {
            "image_classification": self.get_image_classification_fs,
            "object_detection": self.get_object_detection_fs
        }
        self.predict_fs = category_fuction_map[config[model_name].get("category", "image_classification")]
        if self.model_path is None or self.model_path == "" or self.model_path.lower() == "default":
            class_names = self.pytorch_models_dict[self.name + "_weights"].meta["categories"]
        else:
            class_names = config[model_name].get("class_names")
        if class_names == ["imagenet"]:
            class_names = list(imagenet_class_names.values())

        logger = config[model_name].get("logger")
        self.device = config[model_name].get("device")
        input_image_size = (config[model_name]['input_image_size'][0], config[model_name]['input_image_size'][1])
        self.model_obj = self.load_model(self.model_path, self.device)
        self.classes = classes
        self.class_names = class_names
        if self.classes is None or self.classes == "" or self.classes == [""] or self.classes == []:
            self.classes = self.class_names
        self.input_image_size = (input_image_size[0], input_image_size[1])
        # print("self.classes : ", self.classes)
        # print("self.class_names : ", self.class_names)

    def load_model(self, model_path, device):
        """Load the pytorch model.

        Returns:
            torch.nn.Module: The loaded PyTorch model.

        """
        if self.model_path is None or self.model_path == "" or self.model_path.lower() == "default":
            weights = self.pytorch_models_dict[self.name + "_weights"]
            model = self.pytorch_model_loader(weights=weights)
            model.eval()
        else:
            try:
                model = self.torch.load(model_path, map_location=device)
                model.eval()
            except Exception as e:
                print("Error: while loading pytorch model with torch.load().")
                print("Error message: ", e)
                print("Attempting to load model with state_dict.")
                state_dict = self.torch.load(self.model_path, self.device)
                model = self.pytorch_model_loader()  # (weights=state_dict)
                model.load_state_dict(state_dict)
                model.eval()

        return model

    def preprocessing(self, image):
        """Preprocess the image before passing it to the model.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            torch.Tensor: The preprocessed image tensor.

        """
        prediction_transform = self.transforms.Compose([self.transforms.Resize(size=self.input_image_size),
                                                        self.transforms.ToTensor(),
                                                        self.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])])

        return prediction_transform(image)[:3, :, :].unsqueeze(0)

    def get_image_classification_fs(self, predictions, confidence_threshold):
        """
        Get the image classification results.
        """
        fs = []
        for i in range(len(predictions[0])):
            if predictions[0][i] >= float(confidence_threshold) and self.class_names[i] in self.classes:
                j = {'Cs': round(float(predictions[0][i]), 2), 'Lb': self.class_names[i],
                     'Dm': {}, 'Kp': {},
                     'Nobj': "", 'Info': "", 'Uid': ""}
                fs.append(j)
        return fs

    def get_object_detection_fs(self, predictions, confidence_threshold):
        """
        Get the object detection results
        """
        fs = []
        pred_boxes = predictions[0]['boxes'].cpu().detach().numpy().tolist()
        pred_labels = predictions[0]['labels'].cpu().detach().numpy().tolist()
        pred_scores = predictions[0]['scores'].cpu().detach().numpy().tolist()

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            print("score : ", score)
            if score >= confidence_threshold:
                print(f'Label: {label}, Score: {score}, Box: {box}')
                j = {'Cs': round(score, 2), 'Lb': self.class_names[label],
                     'Dm': {
                         'X': box[0] / self.original_image_width,
                         'Y': box[1] / self.original_image_height,
                         'W': (box[2] - box[0]) / self.original_image_width,
                         'H': (box[3] - box[1]) // self.original_image_height
                     }, 'Kp': {},
                     'Nobj': "", 'Info': "", 'Uid': ""}
                fs.append(j)
        return fs

    def predict(self, base64_image, confidence_threshold):
        """Predict the result using the model.

        Args:
            base64_image (str): The base64 encoded image.
            confidence_threshold (float): The confidence threshold for predictions.

        Returns:
            list: A list of dictionaries containing the predicted results. Refer to the structure of "Fs" in README.md

        """
        image = self.PilImage.open(io.BytesIO(base64.b64decode(base64_image)))
        self.original_image_width, self.original_image_height = image.size

        image = self.preprocessing(image)
        st1 = time.time()
        predictions = self.model_obj(image)
        print("Time taken for PyTorch model prediction : ", time.time() - st1)

        output = self.predict_fs(predictions, confidence_threshold)
        return output
