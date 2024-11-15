# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import importlib
import json
import pyfiglet
from milutils.model_to_class_mapper import model_to_class_mapper

model_obj = None
with open("mil_config.json") as conf_file:
    config = json.loads(conf_file.read())


def print_figlet():
    global figlet_ran
    figlet = pyfiglet.Figlet(font="ansi_regular", width=150)
    stylized_text = figlet.renderText('Welcome to')
    print(stylized_text)
    stylized_text_IMIL = figlet.renderText('Infosys Model Inference Library')
    print(stylized_text_IMIL)
    print("Starting IMIL process...")

def executeModel(request):
    req_data = json.loads(request)
    result = predict_method(req_data)
    return json.dumps(result)


def predict_method(req_data):
    global model_obj
    model_name = req_data["Model"]
    class_name = model_name
    module_name = config["ModelExecutor"]["model_loader_file"]
    if model_obj is None:
        if module_name == "default":
            if model_name in model_to_class_mapper:
                module_name = f"modelloader.{model_to_class_mapper[model_name][0].lower()}_model_loader"
                module = importlib.import_module(module_name)
                model_loader_class = getattr(module, model_to_class_mapper[model_name][1])
            else:
                module_name = f"modelloader.{model_name.lower()}_model_loader"
                module = importlib.import_module(module_name)
                model_loader_class = getattr(module, model_name)

        else:
            module = importlib.import_module(module_name)
            model_loader_class = getattr(module, class_name)
        model_obj = model_loader_class(config, model_name)
    final_result = model_obj.predict_request(req_data)
    return final_result


if __name__ == '__main__':
    from milapi.fastapi_caller import fastapi_runner

    print_figlet()
    print("Configuring FastAPI...")
    model_name = config["ModelExecutor"]["model_name"]
    fastapi_runner(config)
