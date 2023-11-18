import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .fcos_outputs import FCOSOutputs
import pdb

__all__ = ["FCOS"]

INF = 100000000

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL

        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, mask_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            results, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, mask_pred, gt_instances, top_feats
            )
            
            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, mask_pred, images.image_sizes, top_feats
                    )
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, mask_pred, images.image_sizes, top_feats
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  False)}
        
        # SVD options
        self.svd_k = cfg.MODEL.CRMASK.SVD.TOPK
        self.m_size = cfg.MODEL.CRMASK.SVD.MASK_SIZE
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        svd_out_channel = [self.svd_k * num_bases * s for s in self.m_size]
        self.gc = cfg.MODEL.CRMASK.SVD.GLOBAL_CONTEXT
        self.gc_out_channels = 32
        self.mode = cfg.MODEL.CRMASK.SVD.MASK_MODE
        self.direct_mask = cfg.MODEL.CRMASK.SVD.DIRECT_MASK
        self.mask_branch = cfg.MODEL.CRMASK.SVD.MASK_BRANCH
        if self.mask_branch:
            head_configs["mask"] = (cfg.MODEL.CRMASK.NUM_MASK_CONVS, cfg.MODEL.FCOS.USE_DEFORMABLE)

        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        # Global context
        if self.gc:
            in_channels_gc = in_channels + self.gc_out_channels

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d

                if self.gc and i == 0:
                    tower.append(conv_func(
                        in_channels_gc, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                else:
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())

            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )
        # CRM branch
        self.U_pred = nn.Conv2d(
        in_channels, svd_out_channel[0], kernel_size=3,
        stride=1, padding=1
        )
        self.V_pred = nn.Conv2d(
        in_channels, svd_out_channel[1], kernel_size=3,
        stride=1, padding=1
        )
        self.S_pred = nn.Conv2d(
        in_channels, self.svd_k * num_bases, kernel_size=3,
        stride=1, padding=1
        )
        mask_modules = [self.U_pred, self.V_pred, self.S_pred]
        #self.Sigma_scales = Scale(init_value=1.0)
        '''else: # direct mask
            self.mask_pred = nn.Conv2d(
            in_channels, mask_channel, kernel_size=3,
            stride=1, padding=1
            )
            mask_modules = [self.mask_pred]
'''
        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        for module in mask_modules:
            for l in module.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        mask = {'U':[], 'V':[], 'S':[]}
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))

            if self.mask_branch:
                mask_tower = self.mask_tower(feature)
                U_pred = self.U_pred(mask_tower)
                V_pred = self.V_pred(mask_tower)
                S_pred = self.S_pred(mask_tower)
            else:
                U_pred = self.U_pred(bbox_tower)
                V_pred = self.V_pred(bbox_tower)
                S_pred = self.S_pred(bbox_tower)
            
            if self.mode == 'cls':
                S_pred = torch.exp(S_pred)
            elif self.mode == 'reg':
                S_pred = F.relu(S_pred)
            else:
                raise NotImplementedError
            #means = torch.tensor([18.6, 6.1, 3.8, 2.8, 2.2, 1.8]).to(device=S_pred.device)

            for k,v in mask.items():
                mask[k].append(eval(k+'_pred'))
            '''else:
                mask_pred = self.mask_pred(mask_tower) #[N, mask_len, Hi, Wi]
                mask.append(F.sigmoid(mask_pred)) 
'''
        return logits, bbox_reg, ctrness, mask, top_feats, bbox_towers