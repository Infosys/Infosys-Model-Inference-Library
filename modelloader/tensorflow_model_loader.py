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


class Tensorflow(BaseModelLoader):
    def __init__(self, config, model_name):
        """Initialize the Tensorflow model loader.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            model_name (str): Name of the model.

        """
        super().__init__(config, model_name)
        import tensorflow as tf
        from keras.src.utils import load_img, img_to_array
        import numpy as np
        self.tf = tf
        self.load_img = load_img
        self.img_to_array = img_to_array
        self.np = np

        self.model_path = None if config[model_name]['model_path'][0] == '' else config[model_name]['model_path'][0]
        self.device = config[model_name].get("device")
        try:
            self.model_obj = self.load_model()
        except Exception as e:
            print("Error: while loading tensorflow modelloader with tensorflow.keras.models.load_model().")
            print(e)
            try:
                self.model_obj = self.load()
            except Exception as e:
                print("Error: while loading tensorflow modelloader with tensorflow.saved_model.load(self.model_path).")
                print(e)
                raise
        name = config[model_name]['name']
        self.classes = config[model_name].get("classes")
        self.class_names = config[model_name].get("class_names")
        if self.class_names == ["imagenet"]:
            self.class_names = list(imagenet_class_names.values())
        if self.classes is None or self.classes == "" or self.classes == [""] or self.classes == []:
            self.classes = self.class_names
        logger = config[model_name].get("logger")

        self.input_image_size = (config[model_name]['input_image_size'][0], config[model_name]['input_image_size'][1])

    def load_model(self):
        """Load the model from the model path."""
        return self.tf.keras.models.load_model(self.model_path)

    def load(self):
        """Load the model from the model path."""
        return self.tf.saved_model.load(self.model_path)

    def predict(self, base64_image, confidence_threshold):
        """Predict the result using the tensorflow model.

        Args:
            base64_image (str): Base64 encoded image.
            confidence_threshold (float): Confidence threshold for predictions.

        Returns:
            list: List of dictionaries containing prediction results. Refer to the structure of "Fs" in README.md

        """
        img = self.load_img(io.BytesIO(base64.b64decode(base64_image)), target_size=self.input_image_size)
        img = self.img_to_array(img)
        img = img / 255
        img = self.np.expand_dims(img, axis=0)
        st1 = time.time()
        predictions = self.model_obj.predict(img)
        print("Time taken for Tensorflow model prediction : ", time.time() - st1)

        output = []
        for pred_j in range(len(predictions)):
            for i in range(len(predictions[pred_j])):
                if predictions[pred_j][i] >= float(confidence_threshold) and self.class_names[i] in self.classes:
                    j = {'Cs': round(float(predictions[0][i]), 2), 'Lb': self.class_names[i],
                         'Dm': {}, 'Kp': {},
                         'Nobj': "", 'Info': "", 'Uid': ""}
                    output.append(j)
        return output
