# Advanced Usage
This document is for advanced user who wants to customize the Infosys Model Inference Library to run their own custom models.

## How to add a Custom Python Model Loader:
This session is about creating a custom class to integrate your code with IMIL.

### Example 1:
We'll be seeing this with an example. Let's try to add a custom class to load OpenAI's Clip model.

Reference :[OpenAI Clip Github](https://github.com/openai/CLIP)

Refer to the `CustomClipModelLoader` class in [custom_model_loader.py](custom_model_loader.py)

If you are a developer, and you want to add your own custom code to work with Imfosys model Inference Library, it is simple.
- Create a class in a python file [custom_model_loader.py](custom_model_loader.py). In this case, the class name is `CustomClipModelLoader`
- The class name should be appended with ModelLoader at the end. This is a convention to follow.
- The functional design of the Class module is at the Developer's ease. Inherit the `BaseModelLoader` class and overwrite the necessary methods. 
- In this case, we are overwriting :__init __(), predict(), and predict_request() methods.
- While overwriting a method, keep in mind that the `predict_request` method is the main driver method that accepts the input dictionary and outputs a dictionary. All the other methods needs to be written accordingly.
- If preprocessing and post-processing are required for your process, then add it accordingly in the predict method.

### Example 2:
Let's look at another simpler example. Refer to the `CustomModelLoader` class in [custom_model_loader.py](custom_model_loader.py) where we load Detecto - an object detection model trained on COCO Dataset.

In CustomModelLoader class,
- We are overloading only the constructor, i.e. the __init __() to load and initialize the model.
- The `predict()` and `predict_request()` are not overwritten and are inherited from the `BaseModelLoader` class.
- The preprocessing() and post-processing() methods are not utilized in this sample.
- 