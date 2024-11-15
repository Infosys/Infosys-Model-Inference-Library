# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import tensorflow as tf

# Load a pretrained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Save the model
model.save('mobilenetv2_model.h5')  # model.save('mobilenetv2_model.keras')
