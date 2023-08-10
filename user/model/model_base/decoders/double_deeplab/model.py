from torch import nn
from typing import Optional
from user.model.model_base.base.double_module import DoubleBranchModel

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn.functional as F
import numpy as np
from skimage import feature
import cv2
from util.constants import SAMPLE_SHAPE
from .decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder, pred_to_input


class DeepLabV3_double(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=8,
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class DeepLabV3Plus_double(DoubleBranchModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    def __init__(
        self,
        encoder_name: str = "resnet50", # resnet50
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        encoder_output_stride: int = 16,    # 下采样率 eval()阶段,encoder_output_stride=8 orign=16
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        cell_classes: int = 3,
        tissue_classes: int=2,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.cell_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels+1,  # concat the tissue
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.tissue_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.cell_decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.cell_encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.tissue_decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.tissue_encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.cell_segmentation_head = SegmentationHead(
            in_channels=self.cell_decoder.out_channels,
            out_channels=cell_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.tissue_segmentation_head = SegmentationHead(
            in_channels=self.tissue_decoder.out_channels,
            out_channels=tissue_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.tissue2cell = pred_to_input()

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

# class PytorchDeeplab_CellModel():
#     """
#     Deeplab model for cell detection implemented with the Pytorch library

#     NOTE: this model does not utilize the tissue patch but rather
#     only the cell patch.

#     Parameters
#     ----------
#     metadata: Dict
#         Dataset metadata in case you wish to compute statistics

#     """
#     def __init__(self, metadata=None):
#         self.device = torch.device('cpu')    # 'cuda:0' or 'cpu'
#         self.metadata = metadata
#         self.resize_to = None # The model is trained with 512 resolution
#         # RGB images and 2 class prediction
#         self.n_classes =  3 # Two cell classes and background

#         self.unet = DeepLabV3Plus_double()
#         self.unet2 = DeepLabV3Plus_double()
#         self.unet3 = DeepLabV3Plus_double()
#         self.load_checkpoint()

#         self.unet2 = self.unet2.to(self.device)
#         self.unet2.eval()
#         self.unet = self.unet.to(self.device)
#         self.unet.eval()
#         self.unet3 = self.unet3.to(self.device)
#         self.unet3.eval()
        

#     def load_checkpoint(self):
#         # """Loading the trained weights to be used for validation"""
#         # _curr_path = os.path.split(__file__)[0]
#         # _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/ocelot_unet.pth")
#         # state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))
#         # self.unet.load_state_dict(state_dict, strict=True)
#         # print("Weights were successfully loaded!")
 
#         # torch.save()包含了state_dict 加载自己的模型用下面的代码
#         _path_to_checkpoint = 'user/model/model_base/decoders/double_deeplab/checkpoints/epoch 4 double_deeplab iou0.4052753439890336 valid_loss 0.4709463355436213.pth'
#         # print('loading model')
#         checkpoint = torch.load(_path_to_checkpoint, map_location='cpu')
#         self.unet.load_state_dict(checkpoint['state_dict'])        
#         # print("Weights were successfully loaded!")

#         _path_to_checkpoint2 = 'user/model/model_base/decoders/double_deeplab/checkpoints/epoch 3 double_deeplab iou0.42800028771186216 valid_loss 0.43694123876941804.pth'
#         # print('loading model')
#         checkpoint2 = torch.load(_path_to_checkpoint2, map_location='cpu')
#         self.unet2.load_state_dict(checkpoint2['state_dict'])        
#         # print("Weights were successfully loaded!")

#         _path_to_checkpoint3 = 'user/model/model_base/decoders/double_deeplab/checkpoints/epoch 5 double_deeplab iou0.42246341799387466 valid_loss 0.4411735376857408.pth'
#         # print('loading model')
#         checkpoint3 = torch.load(_path_to_checkpoint3, map_location='cpu')
#         self.unet3.load_state_dict(checkpoint3['state_dict'])        
#         # print("Weights were successfully loaded!")

#     def prepare_input(self, cell_patch):
#         """This function prepares the cell patch array to be forwarded by
#         the model

#         Parameters
#         ----------
#         cell_patch: np.ndarray[uint8]
#             Cell patch with shape [1024, 1024, 3] with values from 0 - 255

#         Returns
#         -------
#             torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
#             dimension
#         """
#         cell_patch = torch.from_numpy(cell_patch).permute((2, 0, 1)).unsqueeze(0)
#         cell_patch = cell_patch.to(self.device).type(torch.FloatTensor) # torch.cuda.FloatTensor之前是
#         cell_patch = cell_patch / 255 # normalize [0-1]
#         if self.resize_to is not None:
#             cell_patch= F.interpolate(
#                     cell_patch, size=self.resize_to, mode="bilinear", align_corners=True
#             ).detach()
#         return cell_patch
        
#     def find_cells(self, heatmap):
#         """This function detects the cells in the output heatmap

#         Parameters
#         ----------
#         heatmap: torch.tensor
#             output heatmap of the model,  shape: [1, 3, 1024, 1024]

#         Returns
#         -------
#             List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
#         """
#         arr = heatmap[0,:,:,:].cpu().detach().numpy()
#         # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

#         bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
#         bg = np.squeeze(bg, axis=0)
#         obj = 1.0 - bg


#         arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
#         peaks = feature.peak_local_max(
#             arr, min_distance=3, exclude_border=0, threshold_abs=0.0
#         ) # List[y, x]

#         maxval = np.max(pred_wo_bg, axis=0)
#         maxcls_0 = np.argmax(pred_wo_bg, axis=0)

#         peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])# Filter out peaks if background score dominates
#         if len(peaks) == 0:
#             return []

#         scores = maxval[peaks[:, 0], peaks[:, 1]]# Get score and class of the peaks
#         peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

#         predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

#         # peaks = [[2, 3], [5, 10], [4, 6]]
#         # peak_class = [0, 1, 2]
#         # scores = [1,0, 0.5, 0.2]
#         # predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

#         return predicted_cells

#     def post_process(self, logits):
#         """This function applies some post processing to the
#         output logits
        
#         Parameters
#         ----------
#         logits: torch.tensor
#             Outputs of U-Net

#         Returns
#         -------
#             torch.tensor after post processing the logits
#         """
#         if self.resize_to is not None:
#             logits = F.interpolate(logits, size=SAMPLE_SHAPE[:2],
#                 mode='bilinear', align_corners=False
#             )
#         return torch.softmax(logits, dim=1)

#     def __call__(self, cell_patch, tissue_patch, roi_loc, roi_loc_ratio, pair_id):
#         """This function detects the cells in the cell patch using Pytorch U-Net.

#         Parameters
#         ----------
#         cell_patch: np.ndarray[uint8]
#             Cell patch with shape [1024, 1024, 3] with values from 0 - 255
#         tissue_patch: np.ndarray[uint8] 
#             Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
#         pair_id: str
#             Identification number of the patch pair

#         Returns
#         -------
#             List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
#         """
#         # cell_patch = self.prepare_input(cell_patch)
#         cell_patch = cell_patch.unsqueeze(0).to(self.device)
#         tissue_patch = tissue_patch.unsqueeze(0).to(self.device)
#         roi_loc = roi_loc.unsqueeze(0).to(self.device)
#         roi_loc_ratio = roi_loc_ratio.unsqueeze(0).to(self.device)
#         # print(cell_patch.shape)
#         cell_logits = self.unet(cell_patch, tissue_patch, roi_loc, roi_loc_ratio)

#         cell_logits2 = self.unet2(cell_patch, tissue_patch, roi_loc, roi_loc_ratio)
        
#         cell_logits3 = self.unet3(cell_patch, tissue_patch, roi_loc, roi_loc_ratio)

#         cell_logits = (cell_logits+cell_logits2+cell_logits3) / 2

#         # print(logits.shape)
#         heatmap = self.post_process(cell_logits)
#         return self.find_cells(heatmap)
