from util import gcio
from util.constants import (
    GC_CELL_FPATH, 
    GC_TISSUE_FPATH, 
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)
from user.inference import Model
from user.model.model_base.decoders.double_deeplab_inter.model import PytorchDeeplab_CellModel
import torch

def process():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)
    
    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    # Instantiate the inferring model
    # model = Model(meta_dataset)
    # model = PytorchUnetCellModel(meta_dataset)
    # model = PytorchDeeplab_CellModel()
    model = PytorchDeeplab_CellModel()
    # model = PytorchDeeplab_inter2_CellModel()

    # NOTE: Batch size is 1
    # for cell_patch, tissue_patch, pair_id in loader:
    with torch.no_grad():
        for batch in loader:
            cell_patch = batch["image"]
            tissue_patch = batch["tissue_img"]
            pair_id = batch["filename"] # int
            roi_loc = batch["roi_loc"]
            roi_loc_ratio = batch["roi_loc_ratio"]
            # print(f"Processing sample pair {pair_id}")

            # Cell-tissue patch pair inference
            # cell_classification = model(cell_patch, tissue_patch, pair_id)
            # print(cell_classification)
            cell_classification = model(cell_patch, tissue_patch, roi_loc, roi_loc_ratio, pair_id) # unet(3,3)时 (B,C,H,W)
            # print(cell_classification)

            # Updating predictions
            writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    process()