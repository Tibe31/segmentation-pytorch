import json
from copan_border.halcon_procedures.halcon_procedure_border import HalconProcedureBorder
from copan_border.model.segmentation import SegmentationModels


def load_torch_model_and_halcon():
    with open("models.json", "r") as model_info:
        models_path = json.load(model_info)
    procedure_handler = HalconProcedureBorder(
        procedure_path=models_path["halcon_procedures_folder_path"],
        procedures_name={
            "find_circle": "find_inner_circle",
            "rectify_border": "border_rectifier",
        },
    )
    procedures = procedure_handler.load_halcon_procedure()

    model_external = SegmentationModels.load_from_checkpoint(
        models_path["external_filter_model_path"],
        arch="fpn",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
    )
    model_external.eval()

    model_internal = SegmentationModels.load_from_checkpoint(
        models_path["internal_filter_model_path"],
        arch="fpn",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
    )
    model_internal.eval()
    images_path = models_path["images_path"]
    outputs_path = models_path["outputs_path"]
    return (
        procedures,
        procedure_handler,
        model_external,
        model_internal,
        images_path,
        outputs_path,
    )


def find_inner_circle(himage, procedure_handler, plate_path, procedures):
    (
        xc_filter,
        yc_filter,
        r_filter,
        xc_border,
        yc_border,
        r_border,
    ) = procedure_handler.find_inner_circle(himage, procedures["find_circle"])
    parameters = {
        "xc_filter": xc_filter,
        "yc_filter": yc_filter,
        "r_filter": r_filter,
        "xc_border": xc_border,
        "yc_border": yc_border,
        "r_border": r_border,
    }
    with open(plate_path.replace("png", "json"), "w") as plate_info:
        json.dump(parameters, plate_info)


def extract_annulus(
    himage, procedure_handler, xc_filter, yc_filter, r_ext, r_int, procedures
):
    (
        image_polar,
        circumference,
        largest_num_fragments,
    ) = procedure_handler.extract_annulus(
        himage, r_ext, r_int, xc_filter, yc_filter, 600, procedures["rectify_border"]
    )
    return image_polar, circumference, largest_num_fragments
