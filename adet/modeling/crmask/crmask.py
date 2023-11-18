# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.poolers import ROIPooler

from .global_mask_branch import build_mask_branch
from .head import build_head
from .visual import batch_visual

from adet.utils.comm import aligned_bilinear

__all__ = ["CRMask"]

logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class CRMask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.global_mask = cfg.MODEL.CRMASK.MASK_G.USE_GLOBAL

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.crmask_head = build_head(cfg)
        self.instance_loss_weight = cfg.MODEL.CRMASK.INSTANCE_LOSS_WEIGHT
        if self.global_mask:
            self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CRMASK.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CRMASK.MAX_PROPOSALS

        self.topk_proposals_per_im = cfg.MODEL.CRMASK.TOPK_PROPOSALS_PER_IM # -1

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.gc = cfg.MODEL.CRMASK.SVD.GLOBAL_CONTEXT
        if self.gc:
            self.gc_in_feature = 'p3'
            self.gc_out_channels = 32
            self.gc_layer = nn.Conv2d(
                    in_channels, self.gc_out_channels,
                    kernel_size=3, stride=1, padding=1
                )

    def forward(self, batched_inputs):
        original_images = [x["image"].to(self.device) for x in batched_inputs]
        #batch_visual(original_images[0].unsqueeze(0), 1)
        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm, self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)
        # Global context
        if self.gc:
            gc_feature = self.gc_layer(features[self.gc_in_feature])
            for k,v in features.items():
                _, _, H, W = v.size()
                gc_f = F.adaptive_max_pool2d(gc_feature, (H, W))
                features[k] = torch.cat((v, gc_f), dim=1)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            if self.global_mask:
                self.add_bitmasks(gt_instances, images_norm.tensor.size(-2), images_norm.tensor.size(-1))
        else:
            gt_instances = None

        if self.global_mask:
            mask_feats, sem_losses = self.mask_branch(features, gt_instances)
        else:
            mask_feats = None
            sem_losses = {'loss_sem':torch.tensor([0.]).to(self.device)}

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, None
        )
        detector_results, detector_losses = self.crmask_head(proposals, gt_instances)

        if self.training:
            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            return losses
        
        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images_norm.image_sizes)):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            detector_r = detector_postprocess(detector_result, height, width) #Resize the output instance
            #batch_visual(detector_r.pred_masks, 10)
            processed_result = {"instances": detector_r}
            processed_results.append(processed_result)

        return processed_results

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
