# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#
import base64
from datetime import datetime
import io
import time
from milapi.utils import get_mtp
from modelloader.base_model_loader import BaseModelLoader


class CocoObjectDetection_Y7(BaseModelLoader):
    def __init__(self, config, model_name):
        """
        Initialize the Yolov7 modelloader loader.

        Args:
            config (dict): Configuration dictionary.
            model_name (str): Name of the modelloader.

        """
        super().__init__(config, model_name)
        import torch
        import sys
        self.yolo_path = config[model_name]['yolo_path']
        sys.path.insert(0, self.yolo_path)
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import select_device
        from argparse import Namespace
        from PIL import Image
        import numpy as np
        self.PilImage = Image
        self.np = np
        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.Namespace = Namespace

        self.model_path = config[model_name]['model_path']
        self.device = config[model_name].get("device") if config[model_name].get("device") else ""
        self.classes = config[model_name].get("classes") if len(config[model_name].get("classes")) > 0 else None
        self.conf_thres = config[model_name].get("conf_thres") if config[model_name].get("conf_thres") else 0.25
        self.iou_thres = config[model_name].get("iou_thres") if config[model_name].get("iou_thres") else 0.7
        self.img_size = config[model_name].get("img_size") if config[model_name].get("img_size") else 640

        self.opt = self.get_initial_argparse(agnostic_nms=False, augment=False,
                                             classes=self.classes, conf_thres=self.conf_thres,
                                             device=self.device, exist_ok=False,
                                             img_size=self.img_size, iou_thres=self.iou_thres,
                                             name="exp", nosave=False, project="runs/detect",
                                             save_conf=False, save_txt=False, source="inference/images", no_trace=False,
                                             update=False, view_img=False, weights=self.model_path)

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load modelloader
        self.model_obj = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 modelloader
        self.stride = int(self.model_obj.stride.max())  # modelloader stride
        self.imgsz = check_img_size(self.opt.img_size, s=self.stride)  # check img_size

        if self.half:
            self.model_obj.half()  # to FP16

        self.names = self.model_obj.module.names if hasattr(self.model_obj, 'module') else self.model_obj.names

        if self.device.type != 'cpu':
            self.model_obj(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model_obj.parameters())))

    def get_initial_argparse(self, agnostic_nms,
                             augment, classes, conf_thres, device, exist_ok,
                             img_size, iou_thres, name, nosave, project,
                             save_conf, save_txt, source, no_trace,
                             update, view_img, weights):
        """
        Method to convert the parameters into argparse.Namespace.

        Args:
            agnostic_nms (bool): Whether to use agnostic NMS.
            augment (bool): Whether to apply augmentation.
            classes (str): Classes to predict.
            conf_thres (float): Confidence threshold.
            device (str): Device to use.
            exist_ok (bool): Whether to allow existing files.
            img_size (int): Image size.
            iou_thres (float): IoU threshold.
            name (str): Name of the modelloader.
            nosave (bool): Whether to save the results.
            project (str): Project directory.
            save_conf (bool): Whether to save the confidence.
            save_txt (bool): Whether to save the results in text format.
            source (str): Source directory.
            no_trace (bool): Whether to disable traceback.
            update (bool): Whether to update the modelloader.
            view_img (bool): Whether to view the image.
            weights (str): Path to the weights file.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        args_dict = {
            "agnostic_nms": agnostic_nms,
            "augment": augment,
            "classes": classes,
            "conf_thres": conf_thres,
            "device": device,
            "exist_ok": exist_ok,

            "img_size": img_size,
            "iou_thres": iou_thres,

            "name": name,
            "nosave": nosave,
            "project": project,

            "save_conf": save_conf,
            "save_txt": save_txt,
            "source": source,

            "no_trace": no_trace,

            "update": update,

            "view_img": view_img,
            "weights": weights,
        }
        parsed = self.Namespace(**args_dict)
        return parsed

    def predict(self, base64_image, confidence_threshold):
        """
        Method to predict the objects in the given image.

        Args:
            base64_image (str): Base64 encoded image.
            confidence_threshold (float): Confidence threshold for object detection.

        Returns:
            list: List of dictionaries containing the predicted objects.

        """
        output = []
        im = self.PilImage.open(io.BytesIO(base64.b64decode(base64_image)))
        image_width, image_height = im.size
        current_frame = self.np.array(im)
        current_frame = current_frame[:, :, ::-1]
        with self.torch.no_grad():
            img0 = current_frame
            img = self.letterbox(img0, self.imgsz, stride=self.stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = self.np.ascontiguousarray(img)

            img = self.torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            start_time_yolo = time.time()
            pred = self.model_obj(img, augment=self.opt.augment)[0]
            print("Time taken for yolo : ", time.time() - start_time_yolo)

            # Apply NMS
            start_time_yolo_NMS = time.time()
            pred = self.non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                            classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            print("Time taken for yolo NMS : ", time.time() - start_time_yolo_NMS)

            boxes = []
            confidences = []
            classes = []

            for i, det in enumerate(pred):  # detections per image
                im0 = img0.copy()
                if len(det):
                    boxes = self.scale_coords(img.shape[2:], det[:, :4], im0.shape)
                    confidences = det[:, 4]
                    classes = det[:, 5]

                    if self.device != self.torch.device('cpu'):
                        boxes = boxes.cpu().numpy()
                        confidences = confidences.cpu().numpy()
                        classes = classes.cpu().numpy()
                    else:
                        boxes = boxes.detach().numpy()
                        confidences = confidences.cpu().numpy()
                        classes = classes.cpu().numpy()

            if len(boxes) > 0:
                for i in range(0, len(boxes)):
                    if confidences[i] > confidence_threshold:
                        x, y = int(boxes[i][0]), int(boxes[i][1])
                        w, h = int(boxes[i][2]) - x, int(boxes[i][3]) - y
                        j = {"Dm": {
                            "X": round(x / image_width, 2),
                            "Y": round(y / image_height, 2),
                            "W": round(w / image_width, 2),
                            "H": round(h / image_height, 2)
                        },
                            "Cs": round(float(confidences[i]), 3),
                            "Lb": self.names[int(classes[i])],

                            "Kp": dict(),
                             "Uid": "",
                             "Np": "", "Info": ""}
                        output.append(j)
        return output


class PoseEstimation_Y7(BaseModelLoader):
    def __init__(self, config, model_name):
        import sys
        import cv2
        from PIL import Image
        import torch
        import time

        self.yolo_path = config[model_name]['yolo_path']
        sys.path.insert(0, self.yolo_path)

        print("sys.path", sys.path)
        from utils.general import non_max_suppression_kpt, non_max_suppression
        from utils.plots import output_to_keypoint
        from utils.datasets import letterbox
        from torchvision import transforms
        import numpy as np

        super().__init__(config, model_name)

        self.torch = torch
        self.PilImage = Image
        self.non_max_suppression_kpt = non_max_suppression_kpt
        self.non_max_suppression = non_max_suppression
        self.output_to_keypoint = output_to_keypoint
        self.letterbox = letterbox
        self.transforms = transforms
        self.cv2 = cv2
        self.np = np

        self.model_path = config[model_name]['model_path'][0]
        self.model_name = model_name
        self.config = config

        self.device = config[model_name].get("device") if config[model_name].get("device") else "cpu"
        self.device = torch.device(self.device)

        # get the start time

        # Loading pose estimation model
        weigths = torch.load(self.model_path, map_location=self.device)
        self.pose_model = weigths["model"]
        _ = self.pose_model.float().eval()

        if torch.cuda.is_available():
            self.pose_model.half()
            self.pose_model.to(self.device)
        print("Pose Estimation model loaded successfully")

        self.POSE_IMAGE_SIZE = int(self.config[model_name]['img_size'])
        self.STRIDE = int(self.config[model_name]['stride'])
        self.CONFIDENCE_TRESHOLD = float(self.config[model_name]['conf_thres'])
        self.IOU_TRESHOLD = float(self.config[model_name]['iou_thres'])

    def pose_pre_process_frame(self, frame, device):

        # Resize and pad the frame to the desired size, while maintaining aspect ratio
        image = self.letterbox(frame, (self.POSE_IMAGE_SIZE, self.POSE_IMAGE_SIZE),
                               stride=self.STRIDE, auto=True)[0]
        # Convert the image to a PyTorch tensor
        image = self.transforms.ToTensor()(image)
        # Add an extra dimension to the tensor to represent the batch size
        image = self.torch.tensor(self.np.array([image.numpy()]))
        # If a GPU is available, convert the tensor to half precision and move it to the GPU
        if self.torch.cuda.is_available():
            image = image.half()
            image = image.to(self.device)

        return image

    def post_process_pose(self, pose, image_size, scaled_image_size):
        # Extract the original image height and width
        height, width = image_size
        # Extract the scaled image height and width
        scaled_height, scaled_width = scaled_image_size
        # Calculate the scaling factors for the vertical and horizontal dimensions
        vertical_factor = height / scaled_height
        horizontal_factor = width / scaled_width
        result = pose.copy()

        # Loop over each joint in the pose
        for i in range(17):
            # result[i * 3] = horizontal_factor * result[i * 3]
            result[i * 3] = result[i * 3] / scaled_width
            # result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
            result[i * 3 + 1] = result[i * 3 + 1] / scaled_height
        return result

    def pose_post_process_output(
            self,
            output,
            confidence_trashold: float,
            iou_trashold: float,
            image_size,
            scaled_image_size
    ):
        # Apply non-maximum suppression to the output to remove overlapping bounding boxes
        output = self.non_max_suppression_kpt(
            prediction=output,
            conf_thres=confidence_trashold,
            iou_thres=iou_trashold,
            nc=self.pose_model.yaml['nc'],
            nkpt=self.pose_model.yaml['nkpt'],
            kpt_label=True)

        # Disable gradient calculation for performance
        with self.torch.no_grad():
            output = self.output_to_keypoint(output)
            # print("output_to_keypoint output : ", output)

            # Loop over each item in the output
            for idx in range(output.shape[0]):
                # Post-process the pose for the current item
                output[idx, 7:] = self.post_process_pose(
                    output[idx, 7:],
                    image_size=image_size,
                    scaled_image_size=scaled_image_size
                )
        return output

    def predict_request(self, request_data):
        start_time = datetime.now()
        # Extracting request data
        Tid = request_data["Tid"]
        DeviceId = request_data["Did"]
        Fid = request_data["Fid"]
        Per = request_data["Per"]
        mtp = request_data["Mtp"]
        Ts_ntp = request_data["Ts_ntp"]
        Ts = request_data["Ts"]
        Inf_ver = request_data["Inf_ver"]
        Msg_ver = request_data["Msg_ver"]
        Model = request_data["Model"]
        Ad = request_data["Ad"]
        Ffp = request_data["Ffp"]
        Ltsize = request_data["Ltsize"]
        Lfp = request_data["Lfp"]
        # Decoding the base64 image and setting the confidence threshold
        base64_code = request_data.get('Base_64')
        confidence_threshold = float(request_data.get('C_threshold'))
        # Converting the base64 image to a PIL image
        image = self.PilImage.open(io.BytesIO(base64.b64decode(base64_code)))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        width, height = image.size

        # Converting the PIL image to an OpenCV image
        opencvImage = self.cv2.cvtColor(self.np.array(image), self.cv2.COLOR_RGB2BGR)

        image_size = opencvImage.shape[:2]

        try:
            pose_pre_processed_frame = self.pose_pre_process_frame(
                frame=opencvImage,
                device=self.device)
        except Exception as e:
            print("Error occurred in pose estimation yoloV7 model preprocessing " + str(e))
        pose_scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]
        pose_scaled_height, pose_scaled_width = pose_scaled_image_size

        # Executing the pose estimation model
        st4 = time.time()
        try:
            with self.torch.no_grad():
                pose_output = self.pose_model(pose_pre_processed_frame)[0].cpu()
        except Exception as e:
            print("Error occurred in pose estimation yoloV7 model execution " + str(e))

        # Postprocessing the output of the pose estimation model
        try:
            pose_output = self.pose_post_process_output(
                output=pose_output,
                confidence_trashold=self.CONFIDENCE_TRESHOLD,
                iou_trashold=self.IOU_TRESHOLD,
                image_size=image_size,
                scaled_image_size=pose_scaled_image_size
            )
        except Exception as e:
            print("Error occurred in pose estimation yoloV7 model postprocessing " + str(e))

        pose_output_list = pose_output.tolist()
        pose_length = len(pose_output_list)
        output = []
        Fs = []
        Kp_skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                       [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                       [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        for i in range(0, pose_length):
            pi = pose_output_list[i]
            if pi[6] > confidence_threshold:
                j = {}
                final_kp = pi[7:]
                final_kp_dict = dict()
                for kp_i in range(17):
                    extract_kp = final_kp[kp_i * 3:(kp_i * 3) + 3]
                    extract_kp.insert(-1, float(0))
                    final_kp_dict[str(kp_i + 1)] = extract_kp
                j['Cs'] = pi[6]
                j['Lb'] = "Person" if int(pi[1]) == 0 else str(pi[1])

                j["Dm"] = {}
                j["Kp"] = final_kp_dict
                j["Uid"] = " "
                j["Nobj"] = " "
                j['Info'] = ""
                Fs.append(j)

        end_time = datetime.now()
        mtp = get_mtp(mtp, start_time, end_time, Model)
        output.append({"Tid": Tid, "Did": DeviceId, "Fid": Fid, "Fs": Fs, "Mtp": mtp,
                       "Ts": Ts, "Ts_ntp": Ts_ntp, "Msg_ver": Msg_ver, "Inf_ver": Inf_ver,
                       "Rc": "200", "Rm": "Success", "Ad": str(Ad), "Ffp": Ffp,
                       "Lfp": Lfp, "Ltsize": Ltsize})
        self.torch.cuda.empty_cache()
        return output[0]