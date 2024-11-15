# Infosys Model Inference Library - Supported Models

## Dependencies Installation
#### Installing the dependencies for IMIL library
For the Infosys Model Inference Library to work we need the following packages to be installed,
##### Base
- opencv-python
- numpy
- pillow
- pyfiglet

Apart from them if you want the Library to host Fast API server, you need to install sharedfastapi wheel file using the following command,
##### Fast API
- bcrypt
- cryptography
- fastapi
- pydantic
- PyJWT
- python-jose
- python-multipart
- starlette
- uvicorn
- passlib

It can also be installed using the requirements.txt file.

```bash
pip install -r requirements.txt
```
#### Other Dependencies
Apart from the packages mentioned in the `Installing the dependencies for IMIL library` session, there are other dependency which are related to the model. All the packages required for the model to run should be installed. 

## Supported Models
Below are the list of model currently supported by the Infosys Model Inference Library. `Model Name` is the name that needs to be passed in the request JSON to run the model.

| S.No. | Usecase/Framwork Name            | Model Name                  | 
|-------|----------------------------------|-----------------------------|
| 1     | ObjectDetection - COCO - Detecto | CocoObjectDetection_De      | 
| 2     | ObjectDetection - COCO - Yolov5  | CocoObjectDetection_Y5      |
| 3     | ObjectDetection - COCO - Yolov7  | CocoObjectDetection_Y7      | 
| 4     | ObjectDetection - COCO - Yolov8  | CocoObjectDetection_Y8      |
| 5     | FacialExpresionRecognition       | FacialExpressionRecognition |
| 6     | PyTorch                          | Pytorch                     |
| 7     | TensorFlow                       | TensorFlow                  |


## The Infosys Model Inference Library supports the following models

### 1. ObjectDetection - COCO Dataset - Detecto
Detecto is an object detection model that was trained of COCO dataset. It is a PyTorch based model.

#### Installation
For the Detecto model to work with Infosys Model Inference Library we need to install the `detecto` library and the `DetectoInference` wheel file using the following command,
```bash
# While installing torch refer to https://pytorch.org/get-started/locally/ for installing the cpu or cuda version of torch if you have a GPU hardware.
pip install torch torchvision 
pip install detecto==1.2.2
# This file is available in the repository inside the `ModelInference_wheel_files.zip`. Extract the zip file and install the wheel file.
pip install DetectoInference-0.0.1-py3-none-any.whl 
```

#### Configuration
The Detecto model needs certain parameters to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
  "CocoObjectDetection_De":
  {
    "name" : "fasterrcnn_resnet50_fpn",
    "model_path": [],
    "classes": []
  }
}
```

| Config Name	 | Explanation                                                                                                                                                                                                                                      | Sample value               |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| name         | The name of the model to be used.                                                                                                                                                                                                                | fasterrcnn_resnet50_fpn    |
| model_path   | The path to the model file. Passing an empty string will download the model from internet and place it in a cache location.                                                                                                                      | []                         |
| classes      | The classes that you want the model to predict on. Passing an empty string would make the model to consider all the COCO dataset classes. **Note**: The index of the classes begin from 1 for this model(i.e, for "person" class,the index is 1) | ["person", "car"] or [1,2] |

**Reference** : [coco-labels-2014_2017](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt)

#### Verification
For verifying that the Detecto model is installed correctly, you can run the following command,
```bash
python test.py --model CocoObjectDetection_De --confidence 0.5 --iteration 3 --image references/people.jpg
```

### 2. & 3. ObjectDetection - COCO - Yolov5 and Yolov8 Models
Yolo is  Pytorch framework based object detection model. The pretrained model weight files available in the ultralytics repo were trained on COCO dataset.

Using Yolov5 and Yolov8 are made very easy by `ultralytics` package.

#### Installation
For the Yolov5 and Yolov8 model to work with Infosys Model Inference Library we need to install the `ultralytics` library.
You can install ultralytics using pip:
```bash
pip install ultralytics
```
#### Downloading weight files
Users can refer to the below links to download the Yolov5 and Yolov8 weight files.
 - Yolo V5 - [Download YoloV5 weights](https://docs.ultralytics.com/models/yolov5/#performance-metrics)
 - Yolo V8 - [Download YoloV8 weights](https://docs.ultralytics.com/models/yolov8/#performance-metrics)

#### Configuration
The YoloV5 and YoloV8 model needs certain parameters to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
  "CocoObjectDetection_Y5":
  {
    "model_path": ["path to the weight file"],
    "device" : "cpu",
    "conf_thres" : 0.25,
    "iou_thres" : 0.7,
    "classes": [0, 15, 16]
  },
  "CocoObjectDetection_Y8":
  {
    "model_path": ["path to the weight file"],
    "device" : "cpu",
    "conf_thres" : 0.25,
    "iou_thres" : 0.7,
    "classes": []
  }
}
```

In the above configuration sample, the `CocoObjectDetection_Y5` and `CocoObjectDetection_Y8` are the model names. As their name suggests, CocoObjectDetection_Y5 is for Yolo V5 and CocoObjectDetection_Y8 is for Yolo V8



| Config Name	 | Explanation                                                                                                                                                                                                      | Sample value                |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| model_path   | The path to the model file.                                                                                                                                                                                      | ["path/to/the/weight/file"] |
| device       | The device on which the model needs to run.                                                                                                                                                                      | "cpu"                       |
| conf_thres   | The confidence threshold for filtering the model predictions.                                                                                                                                                    | 0.25                        |
| iou_thres    | The IOU threshold for the model.                                                                                                                                                                                 | 0.7                         |
| classes      | List of indexes of the classes, the model needs to predict. **Note**: The index of the classes begin from 0(i.e, for "person" class,the index is 0). If [], then all the classes in COCO dataset are considered. | [0,1] or []                 |

**Reference** : [coco-labels-2014_2017](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt)

#### Verification
For verifying that the Yolov5 and Yolov8 models are installed correctly, you can run the following command,
```bash
python test.py --model CocoObjectDetection_Y5 --confidence 0.5 --iteration 3 --image references/people.jpg
python test.py --model CocoObjectDetection_Y8 --confidence 0.5 --iteration 3 --image references/people.jpg
```

### 4. ObjectDetection - COCO - Yolov7 Model
Yolo is Pytorch framework based object detection model. The model weight files available in the YoloV7 repo were trained on COCO dataset.

#### Installation
For the Yolov7 model to work with Infosys Model Inference Library we need to download the YoloV7 GitHub repo from https://github.com/WongKinYiu/yolov7. 

- Follow the instructions in the repo to install the required dependencies for running YoloV7 model.
- Download the repo as a zipfile and extract it.
- In the configuration file, set the `yolo_path` parameter to the location of the downloaded YoloV7 repo and provide the other configuration parameters as required.

#### Downloading weight files
Users can download Yolov7 weight files from this [weight files link](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#performance).

#### Configuration
The YoloV7 model needs certain parameters to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
  "CocoObjectDetection_Y7":
  {
    "yolo_path" : "place/the/yolo7/repo/directory/here",
    "model_path": ["path/to/Yolov7/model/weight/yolov7.pt"],
    "device" : "cpu",
    "img_size" : 640,
    "conf_thres" : 0.25,
    "iou_thres" : 0.7,
    "classes": [0]
  }
}
```

| Config Name	 | Explanation                                                                                                                                                                                                      | Sample value                      |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| yolo_path    | The path to the directory where the yolov7 repository is present.                                                                                                                                                | ["C:/external_repos/yolov7-main"] |
| model_path   | The path to the model file.                                                                                                                                                                                      | ["path/to/yolov7.pt"]             |
| device       | The device on which the model needs to run.                                                                                                                                                                      | "cpu"                             |
| img_size     | The input size of the image with which the model was trained on.                                                                                                                                                 | 640                               |
| conf_thres   | The confidence threshold for filtering the model predictions.                                                                                                                                                    | 0.25                              |
| iou_thres    | The IOU threshold for the model.                                                                                                                                                                                 | 0.7                               |
| classes      | List of indexes of the classes, the model needs to predict. **Note**: The index of the classes begin from 0(i.e, for "person" class,the index is 0). If [], then all the classes in COCO dataset are considered. | [0,1] or []                       |

**Reference** : [coco-labels-2014_2017](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt)


#### Verification
For verifying that the Yolov7 model is installed correctly, you can run the following command,
```bash
python test.py --model CocoObjectDetection_Y7 --confidence 0.5 --iteration 3 --image references/people.jpg
```

### 5. FacialExpressionRecognition
The Model had the ability to identify face of a person and classify the emotion in their face.

The emotions are listed below,
- Happiness
- Sadness
- Fear
- Anger
- Surprise
- Disgust


#### Installation
For the FacialExpressionRecognition model to work with Infosys Model Inference Library we need to install the `fer` library and the `FERInference` wheel file using the following command,
```bash
# While installing torch refer to https://pytorch.org/get-started/locally/ for installing the cpu or cuda version of torch if you have a GPU hardware.
pip install tensorflow
pip install fer==22.4.0
# This file is available in the repository inside the `ModelInference_wheel_files.zip`. Extract the zip file and install the wheel file.
pip install FERInference-0.0.1-py3-none-any.whl
```

#### Configuration
The FacialExpressionRecognition model needs certain parameters to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
  "FacialExpressionRecognition": {
    "model_path": []
  }
}
```

| Config Name	 | Explanation                                                                                                                             | Sample value               |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| model_path   | The path to the model file. Passing an empty list will automatically download the model from internet and place it in a cache location. | []                         |

#### Verification
For verifying that the FacialExpressionRecognition model is installed correctly, you can run the following command,
```bash
python test.py --model FacialExpressionRecognition --confidence 0.5 --iteration 3 --image references/people.jpg
```

### 6. PyTorch Models

#### Installation
For the PyTorch model to work with Infosys Model Inference Library we need to install the `torch` library.

**Note:** As of now, the Infosys Model Inference Library only supports Image Classification models for PyTorch. Other category models support will be added in future releases.

You can install torch using pip:
```bash
# while installing refer to https://pytorch.org/get-started/locally/ for installing the cpu or cuda version of torch if you have a GPU hardware.
pip install torch torchvision torchaudio
```
#### Configuration
For running the Image Classification pytorch model needs certain parameters to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
"PyTorch":
  {
    "name" : "fire_smoke_classifier",
    "category" : "image_classification",
    "model_path" : ["weight_files\\FireSmokeClassificationResnet.pth"],
    "class_names": ["fire", "neutral", "smoke"],
    "classes" : ["fire"],
    "input_image_size": [224,224],
    "device" : "cpu"
  }
}
```
#### Downloading weight files
Users can download and save the pretrained model weight files trained on Imagenet dataset by referring to [milutils/download_torch_models.py](milutils/download_torch_models.py)

**Fire Smoke Classifier weight file download:** [Fire Smoke Classifier - Resnet50 model](https://github.com/imsaksham-c/Fire-Smoke-Detection/blob/master/trained-models/model_final.pth)

Download the model weight file, place it in your preferred location and mentioned that path in the `model_path` parameter.

| Config Name	     | Explanation                                                                                                                                                                                       | Sample value                                 |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| name             | Name of the pytorch model.                                                                                                                                                                        | resnet50                                     |
| category         | The category of the model. (i.e, image_classification or object_detection). Currently supporting `image_classification` only                                                                      | image_classification                         |
| model_path       | The location of the weight file.                                                                                                                                                                  | ["path/to/weight/file"]                      |
| class_names      | The class names with which the model was trained on. Give ["imagenet"] if the model was trained on imagenet dataset                                                                               | ["fire", "neutral", "smoke"] or ["imagenet"] |
| classes          | The classes the user is interested in filtering. **For Example** specifying ["fire", "smoke"] gives the predictions of those classes alone and does not return any prediction for "neutral" image | ["fire"]                                     |
| input_image_size | Input dimensions of the input image. It is a list of image width and height.                                                                                                                      | [640,640]                                    |
| device           | The device on which the model needs to run.                                                                                                                                                       | "cpu"                                        |

**Note:** Refer to [milutils/dataset_class_names.py](milutils/dataset_class_names.py) for the list of classes in the ImageNet dataset.


#### Verification
For verifying that the pytorch models are installed correctly, you can run the following command,
```bash
python test.py --model PyTorch --confidence 0.5 --iteration 3 --image references/people.jpg
```

### 7. TensorFlow Models

#### Installation
For the TensorFlow model to work with Infosys Model Inference Library we need to install the `tensorflow` library.

**Note:** As of now, the Infosys Model Inference Library only supports Image Classification models for Tensorflow. Other category models support will be added in future releases.

You can install Tensorflow using pip:
```bash
# Please refer to https://www.tensorflow.org/install for installing the CPU or GPU version of TensorFlow.
pip install tensorflow
```
#### Downloading weight files
Users can download and save the pretrained model weight files available in Tensorflow library by referring to [milutils/download_tennsorflow_models.py](milutils/download_tennsorflow_models.py)

#### Configuration
For running the Image Classification Tensorflow model, certain parameters are required to be set in the configuration file [mil_config.json](mil_config.json). The parameters that need to be set are as follows,
```json
{
  "Tensorflow":
  {
    "name" : "mobilenetv2_model",
    "model_path" : ["C:/Demo/mobilenetv2_model.h5"],
    "class_names": ["imagenet"],
    "classes" : [],
    "input_image_size": [224,224],
    "device" : "cpu"
  }
}
```

| Config Name	     | Explanation                                                                                                                                                                                           | Sample value                                           |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| name             | Name of the pytorch model. Will be used to download the Pytorch model from torchvision if the model_path is an empty list                                                                             | mobilenetv2_model                                      |
| model_path       | The location of the weight file.                                                                                                                                                                      | ["path/to/weight/file"]                                |
| class_names      | The class names with which the model was trained on. Give ["imagenet"] if the model was trained on imagenet dataset                                                                                   | ["model", "trained", "class", "names"] or ["imagenet"] |
| classes          | The classes the user is interested in filtering. **For Example** specifying ["Irish wolfhound", "Italian greyhound"] gives the predictions of those classes alone and does not return any predictions | ["A class from ImageNet dataset"]                      |
| input_image_size | Input dimensions of the input image. It is a list of image width and height.                                                                                                                          | [640,640]                                              |
| device           | The device on which the model needs to run.                                                                                                                                                           | "cpu"                                                  |

**Note:** Refer to [milutils/dataset_class_names.py](milutils/dataset_class_names.py) for the list of classes in the ImageNet dataset.

#### Verification
For verifying that the pytorch models are installed correctly, you can run the following command,
```bash
python test.py --model PyTorch --confidence 0.05 --iteration 3 --image references/people.jpg
```
