# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import numpy as np
import cv2


def image_to_base64(image_path):
    """Converts an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def base64_to_image(base64_string, image_path):
    """Converts a base64 string to an image and saves it to the specified path."""
    with open(image_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))


def base64_to_numpy(base64_string):
    """Converts a base64 string to a numpy array."""
    image = base64.b64decode(base64_string)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
