# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import importlib
import sys
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext
from typing import List, Optional, Dict
from typing import Union
from milapi.utils import get_mtp, get_datetime_utc


class ModelAppBuilder:
    '''
    args:

    module1: The python file name where our main function is present
    module2: The function name which needs to be invoked
    endpoint: The endpoint with which the model should be called
    file_location: The directory where module1 is present2
    class_module: If the module2 is present inside any class, then we can mention class name
    class_attributes: If the class requires any attributes, we can provide as a dictionary
    class_type: If code is structured a class module then "Yes" else "No". Default value is "No"

    '''

    def __init__(self, module1, module2, endpoint, file_location, class_module, class_attributes,
                 class_type="no", schema="IVA"):
        self.app = FastAPI()
        self.module1 = module1
        self.module2 = module2
        self.endpoint = endpoint
        self.file_location = file_location
        self.class_type = class_type
        self.class_module = class_module
        self.class_attributes = class_attributes
        self.schema = schema

    def set_schema(self, schema_name):
        self.schema = schema_name

    def build(self):
        sys.path.append(self.file_location)
        if self.class_type.lower() == "yes" or self.class_type is True:
            module = importlib.import_module(self.module1)
            my_class = getattr(module, self.class_module)
            instance = my_class(**self.class_attributes)
            target_func = getattr(instance, self.module2)

        else:
            module = importlib.import_module(self.module1)
            target_func = getattr(module, self.module2)

        # Security
        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        SECRET_KEY = "your-secret-key"  # Replace with your actual secret key
        ALGORITHM = "HS256"
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


        class Token(BaseModel):
            access_token: str
            token_type: str

        class TokenData(BaseModel):
            username: str or None = None

        class User(BaseModel):
            username: str

        class UserInDB(User):
            hashed_password: str

        class PredictionInput(BaseModel):

            # We need to provide the input attributes here
            Tid: str
            Did: str
            Fid: str
            Per: Optional[List[Optional[dict]]]
            Ts: str
            Ts_ntp: str
            Inf_ver: str
            Msg_ver: str
            Model: str
            Ad: Union[dict, str]
            Ffp: str
            Ltsize: str
            Lfp: str
            Mtp: list
            Base_64: str
            C_threshold: Union[float, str]
            I_fn: str
            Msk_img: list
            Rep_img: list
            Prompt: list

        class ModelResult(BaseModel):
            Fs: list
            Tid: str
            Did: str
            Fid: str
            Ts: str
            Ts_ntp: str
            Inf_ver: str
            Msg_ver: str
            Ad: Union[dict, str]
            Ffp: str
            Ltsize: str
            Lfp: str
            Mtp: list
            Rc: str
            Rm: str
            Obase_64: list
            Img_url: list

        # Function to verify the password
        def verify_password(plain_password, hashed_password):
            return pwd_context.verify(plain_password, hashed_password)

        def get_password_hash(password):
            return pwd_context.hash(password)

        # Function to get user info from the database
        def get_user(db, username: str):
            if username in db:
                user_dict = db[username]
                return UserInDB(**user_dict)

        # Function to authenticate the user
        def authenticate_user(db, username: str, password: str):
            user = get_user(db, username)
            if not user:
                return False
            if not verify_password(password, user.hashed_password):
                return False
            return user

        def create_access_token(data: dict):
            to_encode = data.copy()
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt

        async def get_current_user(db, token: str = Depends(oauth2_scheme)):
            credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                                 detail="Could not validate error",
                                                 headers={"WWW-Authenticate": "Bearer"})
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username: str = payload.get("sub")
                if username is None:
                    raise credential_exception
                token_data = TokenData(username=username)

            except JWTError:
                raise credential_exception

            user = get_user(db, username=token_data.username)
            if user is None:
                raise credential_exception

            return user

        async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
            if current_user.disabled:
                raise HTTPException(status_code=400, detail="Inactive User")
            return current_user

        @self.app.post(str(self.endpoint), response_model=ModelResult)
        async def predict(input_data: PredictionInput):
            # api_start_time = get_datetime_utc()
            # req_data = input_data.dict()
            req_data = input_data.model_dump()
            req_data["C_threshold"] = float(req_data["C_threshold"])
            output = target_func(req_data)
            # api_end_time = get_datetime_utc()
            # mtp = output["Mtp"]
            # mtp = get_mtp(mtp, api_start_time, api_end_time, "FastAPI")
            # output["Mtp"] = mtp
            return output
