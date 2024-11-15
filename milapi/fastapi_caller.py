# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

from sharedfastapi.api_utils import ModelAppBuilder
import uvicorn
from milutils.model_to_class_mapper import model_to_class_mapper


def fastapi_runner(config_parser):
    """
    This function runs the FastAPI server for the model
    """
    model_loader_file = config_parser["ModelExecutor"]["model_loader_file"]
    if model_loader_file == "default":
        model_name = config_parser["ModelExecutor"]["model_name"]
        # module1 = "PythonModelExecutor"
        if model_name in model_to_class_mapper:
            module1 = f"modelloader.{model_to_class_mapper[model_name][0].lower()}_model_loader"
        else:
            module1 = f"modelloader.{model_name.lower()}_model_loader"
        module2 = "predict_request"
        endpoint = config_parser["FastAPI"]["endpoint"]
        file_location = "."
        class_module = model_name
        class_attributes = {"config": config_parser, "model_name": model_name}
        class_type = "yes"
    else:
        model_name = config_parser["ModelExecutor"]["model_name"]
        module1 = model_loader_file
        module2 = config_parser["ModelLoader"]["first_method_to_call_in_class"]
        endpoint = config_parser["FastAPI"]["endpoint"]
        file_location = config_parser["ModelLoader"]["file_location"]
        class_module = config_parser["ModelLoader"]["class_name"]
        # class_module = model_name
        class_attributes = {"config": config_parser, "model_name": model_name}
        class_type = "yes"

    app_builder = ModelAppBuilder(
        module1=module1,
        module2=module2,
        endpoint=endpoint,
        file_location=file_location,
        class_module=class_module,
        class_attributes=class_attributes,
        class_type=class_type)
    app_builder.build()

    host = config_parser["FastAPI"]["host"]
    port = int(config_parser["FastAPI"]["port"])
    endpoint = config_parser["FastAPI"]["endpoint"]

    # Print Swagger UI URL
    host_name = "localhost" if host=='0.0.0.0' else host
    print(f"Swagger UI: http://{host_name}:{port}/docs")

    # Print endpoint URL
    print(f"Endpoint URL: http://{host_name}:{port}{endpoint}")

    uvicorn.run(app_builder.app, host=host, port=port)
