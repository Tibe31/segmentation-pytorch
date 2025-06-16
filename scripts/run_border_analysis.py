import halcon as ha
from PIL import Image
import cv2 as cv
import json
import numpy as np
from copan_border.utils.extract_plate_info import extract_plate_info
from run_inference import extract_crop_and_run_inference
from utils import load_torch_model_and_halcon, find_inner_circle, extract_annulus
import time
import re
import math


def main(
    image_front_path,
    image_back_path,
    model_external,
    model_internal,
    procedure_handler,
    procedures,
    outputs_path,
):

    image_back = Image.open(image_back_path).convert("RGB")
    image_front = Image.open(image_front_path).convert("RGB")
    image_back = image_back.resize((4096, 4096))
    image_front = image_front.resize((4096, 4096))
    himage_back = ha.himage_from_numpy_array(np.array(image_back))
    himage_front = ha.himage_from_numpy_array(np.array(image_front))
    start = time.time()
    timestamp, camera_back, camera_id = extract_plate_info(image_back_path)

    if timestamp == "T0" and camera_back:
        find_inner_circle(himage_back, procedure_handler, image_back_path, procedures)
    with open(
        image_back_path.replace("png", "json").replace(timestamp, "T0"), "r"
    ) as plate_info:
        parameters = json.load(plate_info)

    image_polar, circumference, largest_num_fragments = extract_annulus(
        himage_back,
        procedure_handler,
        parameters["xc_filter"],
        parameters["yc_filter"],
        parameters["r_filter"] + 200,
        parameters["r_filter"],
        procedures,
    )
    filter_mask_external = extract_crop_and_run_inference(
        image_polar, parameters, model_external, camera_back
    )

    image_polar, circumference, largest_num_fragments = extract_annulus(
        himage_front,
        procedure_handler,
        parameters["xc_filter"],
        parameters["yc_filter"],
        parameters["r_filter"],
        parameters["r_filter"] - 200,
        procedures,
    )
    filter_mask_internal = extract_crop_and_run_inference(
        image_polar, parameters, model_internal, False
    )

    image_cartesian = []
    if filter_mask_external.max() >= 1:
        himage = ha.himage_from_numpy_array(filter_mask_external)
        himage_cartesian = ha.polar_trans_image_inv(
            himage,
            parameters["xc_filter"],
            parameters["yc_filter"],
            0,
            2 * math.pi,
            parameters["r_filter"] + 200,
            parameters["r_filter"],
            4096,
            4096,
            "nearest_neighbor",
        )
        image_cartesian.append(ha.himage_as_numpy_array(himage_cartesian))
    if filter_mask_internal.max() >= 1:
        himage = ha.himage_from_numpy_array(filter_mask_internal)
        himage_cartesian = ha.polar_trans_image_inv(
            himage,
            parameters["xc_filter"],
            parameters["yc_filter"],
            0,
            2 * math.pi,
            parameters["r_filter"],
            parameters["r_filter"] - 200,
            4096,
            4096,
            "nearest_neighbor",
        )
        image_cartesian.append(ha.himage_as_numpy_array(himage_cartesian))
    end = time.time()
    print(end - start)
    if len(image_cartesian) > 0:
        output_totale = np.sum(image_cartesian, axis=0)
        output_totale[output_totale > 1] = 1
        kernel = np.ones((9, 9), np.uint8)
        output_totale = cv.morphologyEx(
            output_totale.astype(np.uint8), cv.MORPH_CLOSE, kernel
        )

        contours, _ = cv.findContours(
            output_totale, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        overlay_image = np.array(image_front).copy()
        # Draw the contour border on the overlay image
        overlapped = cv.drawContours(
            overlay_image, contours, -1, (255, 0, 0), 2
        )  # Green border with thickness 2
        cv.imwrite(
            outputs_path.format(re.findall(r"\d+", timestamp)[0]),
            cv.cvtColor(overlapped, cv.COLOR_RGB2BGR),
        )
        pass
    else:
        cv.imwrite(
            outputs_path.format(re.findall(r"\d+", timestamp)[0]),
            cv.cvtColor(np.array(image_front), cv.COLOR_RGB2BGR),
        )
        print("No crescita su bordo")


if __name__ == "__main__":
    (
        procedures,
        procedure_handler,
        model_external,
        model_internal,
        images_path,
        outputs_path,
    ) = load_torch_model_and_halcon()
    for timestamp in range(0, 7):  # we run for 6 timestamp (totally random value)
        try:
            image_file_front = images_path.format(timestamp)
            image_file_back = image_file_front.replace(
                "F1400B0", "F0B200"
            )  # difference in front and back images filename is just in this part

            main(
                image_file_front,
                image_file_back,
                model_external,
                model_internal,
                procedure_handler,
                procedures,
                outputs_path,
            )
        except Exception as e:
            print(e)
