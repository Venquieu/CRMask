import logging
import torch
import torch.nn
import torch.nn.functional as F

from detectron2.layers import cat
from .visual import batch_visual

logger = logging.getLogger(__name__)

def build_head(cfg):
    return CRMaskHead(cfg)

class CRMaskHead(object):
    def __init__(self, cfg) -> None:
        self.m_size = cfg.MODEL.CRMASK.SVD.MASK_SIZE
        self.svd_k = cfg.MODEL.CRMASK.SVD.TOPK
        self.mode = cfg.MODEL.CRMASK.SVD.MASK_MODE
        self.mask_lmd = 2.0
        self.sigma_lmd = 2.0
        self.activate = 'mse' if self.mode == 'reg' else 'bce'
        self.p = 2

    def __call__(self, proposals, gt_instances):
        if gt_instances is not None:
            # training
            p_instances = proposals['instances']
            pos_inds = p_instances.pos_inds
            gt_inds = p_instances.gt_inds

            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([p_instances.U_pred.sum() * 0 + p_instances.S_pred.sum() * 0, \
                    p_instances.V_pred.sum() * 0])}

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                if self.m_size[0] != self.m_size[1]:
                    raise NotImplementedError
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.m_size[0]
                ).to(device=p_instances.U_pred.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            gt_masks = gt_masks[gt_inds].to(dtype = torch.float32)

            if self.mode == 'reg':
                gt_u, gt_s, gt_v = torch.svd(gt_masks)
                gt_masks = self.svd_combinate(
                    gt_u[:, :, :self.svd_k],
                    gt_s[:, :self.svd_k],
                    gt_v[:, :, :self.svd_k]
                )
            pred_masks = self.svd_combinate(
                p_instances.U_pred,
                gt_s[:, :self.svd_k] if self.mode == 'reg' else p_instances.S_pred,
                p_instances.V_pred,
                combinate= True
            )
            '''visual_masks = []
            for i in range(4):
                visual_masks.append(gt_masks[i].unsqueeze(0))
                visual_masks.append(pred_masks[i].unsqueeze(0))
            visual_masks = cat(visual_masks,dim=0)
            batch_visual(visual_masks, 2)'''
            loss = {}
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)
            pred_masks = pred_masks.view(N, -1)
            gt_ctr = p_instances.gt_ctrs
            loss_denorm = proposals["loss_denorm"]

            logger.info('Sigma pred[0]: {}'.format(p_instances.S_pred[0]))
            if self.mode == 'reg':
                rescaled_gt_s = torch.pow(gt_s[:, :self.svd_k], 1/self.p)
                s_loss = F.mse_loss(p_instances.S_pred, rescaled_gt_s, reduction='none')
                s_loss = ((s_loss.mean(dim=-1) * gt_ctr).sum()
                            / loss_denorm)
                loss['s_loss'] = s_loss * self.sigma_lmd

            if self.activate == 'bce':
                mask_losses = F.binary_cross_entropy_with_logits(
                    pred_masks, gt_masks, reduction="none")
            elif self.activate == 'mse':
                mask_losses = F.mse_loss(
                    pred_masks, gt_masks, reduction="none")
            else:
                raise NotImplementedError

            mask_loss = ((mask_losses.mean(dim=-1) * gt_ctr).sum()
                         / loss_denorm)
            loss['loss_mask'] = mask_loss * self.mask_lmd

            return None, loss
        else:
            # testing, `proposals` is a list with len=N
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.m_size[0], self.m_size[1])
                return proposals, {}
            U_pred = cat([x.U_pred for x in proposals], dim=0)
            V_pred = cat([x.V_pred for x in proposals], dim=0)
            S_pred = cat([x.S_pred for x in proposals], dim=0)
            if self.mode == 'reg':
                S_pred = torch.pow(S_pred, self.p)
            pred_masks = self.svd_combinate(U_pred, S_pred, V_pred)
            if self.activate == 'bce':
                pred_masks = pred_masks.sigmoid()
            #batch_visual(pred_masks, 10)
            pred_masks = pred_masks.view(-1, 1, self.m_size[0], self.m_size[1])

            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_masks[start_ind:end_ind]
                box.remove('U_pred')
                box.remove('V_pred')
                box.remove('S_pred')
                start_ind = end_ind
            return proposals, {}

    def svd_combinate(self, U, Sigma, V, combinate = True, separate = False):
        N = U.size(0)
        u_pred = U.reshape(N, self.m_size[0], self.svd_k)
        v_pred = V.reshape(N, self.m_size[1], self.svd_k)
        if combinate:
            e = torch.eye(self.svd_k, self.svd_k).to(dtype=Sigma.dtype, device=Sigma.device)
            s_pred = Sigma.unsqueeze(2) * e.unsqueeze(0) #[n,k,k]

        u_pred = u_pred/(torch.sqrt(torch.pow(u_pred, 2).sum(1)).unsqueeze(1))
        v_pred = v_pred/(torch.sqrt(torch.pow(v_pred, 2).sum(1)).unsqueeze(1))
        if combinate:
            if separate:
                masks = []
                for l in range(self.svd_k):
                    ul = u_pred[:,:,l].view(N,self.m_size[0], 1)
                    sl = s_pred[:,l,l].view(N, 1, 1)
                    vl = v_pred[:,:,l].view(N,self.m_size[1], 1)
                    mask = torch.matmul(torch.matmul(ul, sl), vl.permute(0, 2, 1))
                    masks.append(mask)
                return masks
            mask = torch.matmul(torch.matmul(u_pred, s_pred), v_pred.permute(0, 2, 1)) #[n, m_size, m_size]
            return mask
        else:
            return [u_pred, Sigma, v_pred]