import torch
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.modeling.poolers import ROIPooler

def build_blender(cfg):
    return Blender(cfg)


class Blender(object):
    def __init__(self, cfg):

        # fmt: off
        self.pooler_resolution = cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION
        sampling_ratio         = cfg.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.BLENDMASK.POOLER_TYPE
        pooler_scales          = cfg.MODEL.BLENDMASK.POOLER_SCALES
        self.attn_size         = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.top_interp        = cfg.MODEL.BLENDMASK.TOP_INTERP
        self.num_bases         = cfg.MODEL.BASIS_MODULE.NUM_BASES
        # fmt: on
        self.m_size = cfg.MODEL.CRMASK.SVD.MASK_SIZE
        self.svd_k = cfg.MODEL.CRMASK.SVD.TOPK
        #self.attn_len = num_bases * self.attn_size * self.attn_size

        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

    def __call__(self, bases, proposals, gt_instances):
        if gt_instances is not None:
            # training
            # reshape attns
            dense_info = proposals["instances"]
            #attns = dense_info.top_feats
            attns = self.svd_combinate(
                dense_info.U_pred,
                dense_info.S_pred,
                dense_info.V_pred,
                combinate= True
            )
            pos_inds = dense_info.pos_inds
            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([x.sum() * 0 for x in attns]) + bases[0].sum() * 0}

            gt_inds = dense_info.gt_inds

            rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])
            rois = rois[gt_inds]
            pred_mask_logits = self.merge_bases(rois, attns)

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=pred_mask_logits.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            gt_masks = gt_masks[gt_inds]
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)

            gt_ctr = dense_info.gt_ctrs
            loss_denorm = proposals["loss_denorm"]
            mask_losses = F.binary_cross_entropy_with_logits(
                pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="none")
            mask_loss = ((mask_losses.mean(dim=-1) * gt_ctr).sum()
                         / loss_denorm)
            return None, {"loss_mask": mask_loss}
        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.pooler_resolution, self.pooler_resolution)
                return proposals, {}
            rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            #attns = cat([x.top_feat for x in proposals], dim=0)
            attns = self.svd_combinate(
                cat([x.U_pred for x in proposals], dim=0),
                cat([x.S_pred for x in proposals], dim=0),
                cat([x.V_pred for x in proposals], dim=0),
                combinate= True
            )
            pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)
            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                start_ind = end_ind
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()
        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)

    def svd_combinate(self, U, Sigma, V, combinate = True):
        N = U.size(0)
        u_pred = U.reshape(N, self.num_bases, self.m_size[0], self.svd_k)
        v_pred = V.reshape(N, self.num_bases, self.m_size[1], self.svd_k)
        if combinate:
            e = torch.eye(self.svd_k, self.svd_k).to(dtype=Sigma.dtype, device=Sigma.device)
            s_pred = Sigma.reshape(N, self.num_bases, -1).unsqueeze(3) * e[None, None,:,:] #[n,K,k,k]

        u_pred = u_pred/(torch.sqrt(torch.pow(u_pred, 2).sum(2)).unsqueeze(2))
        v_pred = v_pred/(torch.sqrt(torch.pow(v_pred, 2).sum(2)).unsqueeze(2))
        if combinate:
            mask = torch.matmul(torch.matmul(u_pred, s_pred), v_pred.permute(0, 1, 3, 2)) #[n, m_size, m_size]
            return mask
        else:
            return [u_pred, Sigma, v_pred]
