# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

model_to_class_mapper = {
    "FireSmokeClassifier_R50": ["Pytorch", "Pytorch"],
    "CocoObjectDetection_Mp": ["Mediapipe", "Mediapipe"],
    "FaceDetection_Mp": ["Mediapipe", "Mediapipe"],
    "PoseEstimation_Y7": ["YoloV7", "PoseEstimation_Y7"],
    "CocoObjectDetection_Y5": ["Yolo_Ultralytics", "Yolo_Ultralytics"],
    "CocoObjectDetection_Y7": ["YoloV7", "CocoObjectDetection_Y7"],
    "CocoObjectDetection_Y8": ["Yolo_Ultralytics", "Yolo_Ultralytics"]
}