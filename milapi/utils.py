# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#
import datetime


def convert_output_fs_to_apiV1(fs_list):
    """
    This function converts the given Fs list to the API V1 format by adding four parameters to each item.

    Parameters:
        fs_list (list): The input Fs list to be converted.

    Returns:
        list: The converted Fs list with the added parameters 'Nobj', 'Uid', 'Info', and 'Kp'.
    """
    for i in fs_list:
        i["Nobj"] = ""
        i["Uid"] = ""
        i["Info"] = "{}"
        i["Kp"] = {}
    return fs_list


def get_mtp(mtp, start_time, end_time, model_name):
    """
    This function adds the start time, end time, and modelloader name to the Mtp list.

    Parameters:
        mtp (list): The Mtp list to which the data will be added.
        start_time (datetime): The start time of the modelloader inference.
        end_time (datetime): The end time of the modelloader inference.
        model_name (str): The name of the modelloader.

    Returns:
        list: The updated Mtp list with the added data.
    """
    start_1 = str(start_time.strftime("%Y-%m-%d,%I:%M:%S"))
    start_2 = str(start_time.strftime("%f"))[:3]
    start_3 = str(start_time.strftime("%p"))
    start_time = start_1 + "." + start_2 + " " + start_3

    end_1 = str(end_time.strftime("%Y-%m-%d,%I:%M:%S"))
    end_2 = str(end_time.strftime("%f"))[:3]
    end_3 = str(end_time.strftime("%p"))
    end_time = end_1 + "." + end_2 + " " + end_3

    mtp_dict = {
        "Etime": end_time,
        "Src": model_name,
        "Stime": start_time
    }
    mtp.append(mtp_dict)
    return mtp


def get_datetime_utc():
    try:
        start_time = datetime.datetime.now(datetime.UTC)
    except Exception as e:
        start_time = datetime.datetime.utcnow()
    return start_time

def load_imagenet_class_names():
    """
    This function loads the ImageNet class names from the 'imagenet_class_names.txt' file.

    Returns:
        list: The list of ImageNet class names.
    """
    class_names = []
    with open("imagenet_class_names.txt", "r") as f:
        for line in f:
            class_names.append(line.strip())
    return class_names
