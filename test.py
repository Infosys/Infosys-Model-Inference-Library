# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import argparse
import ast
import json
from PythonModelExecutor import executeModel
from milutils.generalutils import image_to_base64

sample_input_json = {
    "Tid": "1",
    "Did": "DeviceId_11",
    "Fid": "1160",
    "C_threshold": 0.3,
    "Per": [],
    "Mtp": [
        {
            "Etime": "08-02-2023,02:11:33.513 PM",
            "Src": "grabber",
            "Stime": "08-02-2023,02:11:22.744 PM"
        },
        {
            " Etime": "08-02-2023,02:11:33.513 PM",
            "Src": "predictor",
            " Stime": "08-02-2023,02:11:22.744 PM"
        }
    ],
    "Ts": "",
    "Ts_ntp": "",
    "Inf_ver": "",
    "Msg_ver": "",
    "Model": "CocoObjectDetection_De",
    "Ad": "",
    "Ffp": "ffp",
    "Ltsize": "ltsize",
    "Lfp": "lfp",
    "Base_64": "<replace with base64 format of image>",
    "Prompt": [["fire", "smoke", "neutral"]],
    "I_fn": "",
    "Msk_img": [],
    "Rep_img": [],
    "Img_url": []
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some command line arguments.")
    parser.add_argument('--model', type=str, help='Name of the model.')
    parser.add_argument('--image', type=str, help='location of the image.')
    parser.add_argument('--iteration', type=int, help='Number of times the model executes.')
    parser.add_argument('--confidence', type=float, help='confidence threshold for the model.')
    parser.add_argument('--prompt', help='Prompt for the model.')

    args = parser.parse_args()

    model_to_test = args.model if args.model else "CocoObjectDetection_De"
    iteration = args.iteration if args.iteration else 1
    confidence = args.confidence if args.confidence else 0.5
    image = args.image if args.image else "references/people.jpg"
    prompt = ast.literal_eval(args.prompt) if args.prompt else ["fire", "smoke", "neutral"]

    print(f"Testing Model: {model_to_test}")
    print(f"The model will execute for : {iteration} times")

    sample_input_json["Model"] = model_to_test
    sample_input_json["C_threshold"] = confidence
    sample_input_json["Base_64"] = image_to_base64(image)
    sample_input_json["Prompt"] = [prompt]
    for i in range(iteration):
        print(f"Result {i+1} : ", executeModel(json.dumps(sample_input_json)))
        print(f"---------------------------{model_to_test} {i+1} execution completed---------------------------")
