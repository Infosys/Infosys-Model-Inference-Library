# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import io
import json
import time
from milapi.utils import get_mtp, get_datetime_utc
from modelloader.base_model_loader import BaseModelLoader


class CustomDetectoModelLoader(BaseModelLoader):
    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        from detectoinference import Detecto
        model_path = None if config[model_name]['model_path'][0] == '' \
            else json.loads(config[model_name]['model_path'])
        self.model_obj = Detecto(model_path if model_path is None else model_path[0])


class CustomClipModelLoader(BaseModelLoader):
    def __init__(self, config, model_name):
        """Initializing the VideoSearch CLip module.

        Args:
            clip_model (str, None): The file location of the model weight(.pth file). (defaults is None, If None the model weight is downloaded and placed in torch cache folder when the Class in initialised for the first time in an environment)
            prompt: Eg: ["fire", "smoke", "neutral"]
            logger (None, logger object): Used for logging. (defaults to None)
        """
        super().__init__(config, model_name)
        import torch
        import clip
        from PIL import Image
        self.torch = torch
        self.clip = clip
        self.Image = Image
        clip_model = config[model_name]['clip_model']
        device = config[model_name]['device']
        self.logger = config[model_name].get('logger')
        if device != None or device == "":
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model, self.preprocess = self.clip.load(clip_model, device=self.device)
        if self.logger:
            self.logger.debug("device : " + str(self.device))
            self.logger.debug("VideoSearch Clip model loaded successfully")

    def predict(self, base64_image, prompt, confidence_threshold=0.5):
        """Predict the objects present in an image.

        Args:
            base64_image (str): base64 encoding of an image
            confidence_threshold (float): confidence threshold for prediction. Ranges between 0.0 and 1.0 (defaults to 0.5)

        Returns:
            A List of dictionaries, each having 'Confidence Score', 'Label', 'Dimensions of bounding box'
        """
        st1 = time.time()
        image = self.Image.open(io.BytesIO(base64.b64decode(base64_image)))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        prompt = prompt[0]  # Eg: ["fire", "smoke", "neutral"]
        text = self.clip.tokenize(prompt).to(self.device)

        with self.torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if self.logger:
            self.logger.info("Time taken for prediction " + str(time.time() - st1) + " seconds")

        output = []
        predicted_scores_list = list(probs[0])
        for i in range(len(predicted_scores_list)):
            if predicted_scores_list[i] >= float(confidence_threshold):
                j = {}
                j['Cs'] = round(float(predicted_scores_list[i]), 4)
                j['Lb'] = prompt[i]
                j['Dm'] = {}
                j['Kp'] = {}
                j['Info'] = ""
                j['Nobj'] = ""
                j['Uid'] = ""
                output.append(j)
        # print("output: ", output)
        return output

    def predict_request(self, req_data):
        output = []

        # Extraction
        Tid = req_data["Tid"]
        DeviceId = req_data["Did"]
        Fid = req_data["Fid"]
        Cs = req_data["C_threshold"]
        Base_64 = req_data["Base_64"]
        Per = req_data["Per"]
        mtp = req_data["Mtp"]
        Ts_ntp = req_data["Ts_ntp"]
        Ts = req_data["Ts"]
        Inf_ver = req_data["Inf_ver"]
        Msg_ver = req_data["Msg_ver"]
        Model = req_data["Model"]
        Ad = req_data["Ad"]
        Lfp = req_data["Lfp"]
        Ltsize = req_data["Ltsize"]
        Ffp = req_data["Ffp"]
        prompt = req_data["Prompt"]

        start_time = get_datetime_utc()
        predicted_fs_list = self.predict(Base_64, prompt, Cs)
        end_time = get_datetime_utc()
        mtp = get_mtp(mtp, start_time, end_time, Model)

        output.append({"Tid": Tid, "Did": DeviceId, "Fid": Fid, "Fs": predicted_fs_list, "Mtp": mtp,
                       "Ts": Ts, "Ts_ntp": Ts_ntp, "Msg_ver": Msg_ver, "Inf_ver": Inf_ver,
                       "Obase_64": [], "Img_url": [],
                       "Rc": "200", "Rm": "Success", "Ad": Ad, "Lfp": Lfp, "Ffp": Ffp, "Ltsize": Ltsize})
        return output[0]
