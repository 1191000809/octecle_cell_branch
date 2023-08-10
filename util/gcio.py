import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List
from torchvision import transforms
import torch
from util.constants import SAMPLE_SHAPE, GC_METADATA_FPATH


def read_json(fpath: Path) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data 
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

#  we decided to stack all cell and tissue patches across the first image dimension in order to reduce the number of files
#  tif文件中，在第一个维度是堆叠起来的比如(1024*n,1024,3)这样的格式。
class DataLoader:
    """This class is meant to load and iterate over the samples
    already uploaded to GC platform. All cell and tissue samples are
    concatenated/stacked together sequentially in a single file, one
    for cell and another for tissue.

    Parameters
    ----------
    cell_path: Path
        Path to where the cell patches can be found
    tissue_path: Path
        Path to where the tissue patches can be found
    """
    def __init__(self, cell_path, tissue_path):
        cell_fpath = [os.path.join(cell_path, f) for f in os.listdir(cell_path) if ".tif" in f]
        tissue_fpath = [os.path.join(tissue_path, f) for f in os.listdir(tissue_path) if ".tif" in f]
        assert len(cell_fpath) == len(tissue_fpath) == 1

        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # weight:imagenet
            # transforms.Normalize(mean=[0.8079029, 0.64413816, 0.7433438], std=[0.13459155, 0.1738798, 0.1296224])
        ])

        self.cell_patches = np.array(Image.open(cell_fpath[0]))
        self.tissue_patches = np.array(Image.open(tissue_fpath[0]))
        
        self.meta_dataset = read_json(GC_METADATA_FPATH)

        assert (self.cell_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
            "The same of the input cell patch is incorrect"
        assert (self.tissue_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
            "The same of the input tissue patch is incorrect"

        # Samples are concatenated across the first axis
        assert self.cell_patches.shape[0] % SAMPLE_SHAPE[0] == 0
        assert self.tissue_patches.shape[0] % SAMPLE_SHAPE[0] == 0

        self.num_images = self.cell_patches.shape[0] // SAMPLE_SHAPE[0]
        # self.num_images = 1
        assert self.num_images == self.tissue_patches.shape[0]//SAMPLE_SHAPE[0], \
            "Cell and tissue patches have different number of instances"

        self.cur_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx < self.num_images:
            # Read patch pair and the corresponding id 
            cell_patch = self.cell_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]
            cell_patch = self.data_transforms(cell_patch)
            # print('cell'+str(cell_patch.shape))   # [3, 1024, 1024]
            tissue_patch = self.tissue_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]
            tissue_patch = self.data_transforms(tissue_patch)
            # print('tissue'+str(tissue_patch.shape))   

            pair_id = self.cur_idx
            pair = self.meta_dataset[pair_id]

            offset_x = pair["patch_x_offset"]*SAMPLE_SHAPE[1]
            offset_y = pair["patch_y_offset"]*SAMPLE_SHAPE[0]
            loc = [offset_x,offset_y]
            loc = np.array(loc)
            loc = torch.IntTensor(loc)

            offset_x = pair["patch_x_offset"]
            offset_y = pair["patch_y_offset"]
            loc_ratio = [offset_x,offset_y]
            loc_ratio = np.array(loc_ratio)
            loc_ratio = torch.IntTensor(loc_ratio)

            # Increment the current image index
            self.cur_idx += 1

            # Return the image data
            # return cell_patch, tissue_patch, pair_id
            return {'image': cell_patch, 
                    'mask': cell_patch, # false
                    'tissue_img': tissue_patch,
                    'tissue_mask': tissue_patch,
                    'roi_loc_ratio': loc_ratio,
                    'roi_loc': loc, 
                    'filename': pair_id,    
                    }
        else:
            # Raise StopIteration when no more images are available
            raise StopIteration


# class DataLoader:
#     """This class is meant to load and iterate over the samples
#     already uploaded to GC platform. All cell and tissue samples are
#     concatenated/stacked together sequentially in a single file, one
#     for cell and another for tissue.

#     Parameters
#     ----------
#     cell_path: Path
#         Path to where the cell patches can be found
#     tissue_path: Path
#         Path to where the tissue patches can be found
#     """
#     def __init__(self, cell_path, tissue_path):
#         cell_fpath = [os.path.join(cell_path, f) for f in os.listdir(cell_path) if ".tif" in f]
#         tissue_fpath = [os.path.join(tissue_path, f) for f in os.listdir(tissue_path) if ".tif" in f]
#         assert len(cell_fpath) == len(tissue_fpath) == 1

#         data_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.8079029, 0.64413816, 0.7433438], std=[0.13459155, 0.1738798, 0.1296224])
#         ])

#         self.cell_patches = data_transforms(Image.open(cell_fpath[0]))
#         self.tissue_patches = np.array(Image.open(tissue_fpath[0]))

#         # assert (self.cell_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
#         #     "The same of the input cell patch is incorrect"
#         assert (self.tissue_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
#             "The same of the input tissue patch is incorrect"

#         # Samples are concatenated across the first axis
#         # assert self.cell_patches.shape[0] % SAMPLE_SHAPE[0] == 0
#         assert self.tissue_patches.shape[0] % SAMPLE_SHAPE[0] == 0

#         # self.num_images = self.cell_patches.shape[0] // SAMPLE_SHAPE[0]
#         self.num_images = 1
#         # assert self.num_images == self.tissue_patches.shape[0]//SAMPLE_SHAPE[0], \
#         #     "Cell and tissue patches have different number of instances"

#         self.cur_idx = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.cur_idx < self.num_images:
#             # Read patch pair and the corresponding id 
#             cell_patch = self.cell_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]
#             tissue_patch = self.tissue_patches[self.cur_idx*SAMPLE_SHAPE[0]:(self.cur_idx+1)*SAMPLE_SHAPE[0],:,:]

#             pair_id = self.cur_idx

#             # Increment the current image index
#             self.cur_idx += 1

#             # Return the image data
#             return cell_patch, tissue_patch, pair_id
#         else:
#             # Raise StopIteration when no more images are available
#             raise StopIteration

class DetectionWriter:
    """This class writes the cell predictions to the designated 
    json file path with the Multiple Point format required by 
    Grand Challenge

    Parameters
    ----------
    output: Path
        path to json output file to be generated
    """

    def __init__(self, output_path: Path):

        if output_path.suffix != '.json':
            output_path = output_path / '.json' 

        self._output_path = output_path
        self._data = {
            "type": "Multiple points",
            "points": [],
            "version": {"major": 1, "minor": 0},
        } 

    def add_point(
            self, 
            x: int, 
            y: int,
            class_id: int,
            prob: float, 
            sample_id: int
        ):
        """Recording a single point/cell

        Parameters
        ----------
        x: int
            Cell's x-coordinate in the cell patch
        y: int
            Cell's y-coordinate in the cell patch
        class_id: int
            Class identifier of the cell, either 1 (BC) or 2 (TC)
        prob: float
            Confidence score
        sample_id: str
            Identifier of the sample
        """
        point = {
            "name": "image_{}".format(str(sample_id)),
            "point": [int(x), int(y), int(class_id)],
            "probability": prob}
        self._data["points"].append(point)

    def add_points(self, points: List, sample_id: str):
        """Recording a list of points/cells

        Parameters
        ----------
        points: List
            List of points, each point consisting of (x, y, class_id, prob)
        sample_id: str
            Identifier of the sample
        """
        for x, y, c, prob in points:
            self.add_point(x, y, c, prob, sample_id)

    def save(self):
        """This method exports the predictions in Multiple Point json
        format at the designated path. 
        
        - NOTE: that this will fail if not cells are predicted
        """
        assert len(self._data["points"]) > 0, "No cells were predicted"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)
        print(f"Predictions were saved at `{self._output_path}`")

