# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import time
from abc import ABC, abstractmethod
from datetime import datetime
from milapi.utils import get_mtp, get_datetime_utc


class BaseModelLoader(ABC):
    @abstractmethod
    def __init__(self, config, model_name):
        """Initialize the modelloader loader."""
        self.model_obj = None

    def preprocessing(self, image):
        pass

    def postprocessing(self, data):
        pass

    def predict(self, Base_64, Cs):
        """Predict the result using the modelloader."""
        modelOutput = self.model_obj.executeModel(Base_64, Cs)
        return modelOutput

    def predict_request(self, req_data):
        """This function takes the request as input and returns the response after processing the request.
        The request is in dictionary format.
        The response is also in dictionary format."""
        start_time = get_datetime_utc()
        start_t = time.time()
        output = []
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

        predicted_fs_list = self.predict(Base_64, Cs)
        end_time = get_datetime_utc()
        print(f"Time taken for model prediction: {time.time() - start_t} seconds")

        mtp = get_mtp(mtp, start_time, end_time, Model)

        output.append({"Tid": Tid, "Did": DeviceId, "Fid": Fid, "Fs": predicted_fs_list, "Mtp": mtp,
                       "Ts": Ts, "Ts_ntp": Ts_ntp, "Msg_ver": Msg_ver, "Inf_ver": Inf_ver,
                       "Obase_64": [], "Img_url": [],
                       "Rc": "200", "Rm": "Success", "Ad": Ad, "Lfp": Lfp, "Ffp": Ffp, "Ltsize": Ltsize})
        return output[0]
