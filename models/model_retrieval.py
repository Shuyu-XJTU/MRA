import torch
from torch.nn import init

from models.mra import MRA


class Retrieval(MRA):
    def __init__(self, config):
        super().__init__(config)

        self.eda = config.get('eda', False)
        self.region = config.get('region', 0)


    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                idx=None, label=None,
                text_ids_eda=None, text_atts_eda=None,
                text_ids_box=None, text_atts_box=None, box_atts=None,):

        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        bs = image_embeds.size(0)


        if self.region > 0:
            text_embeds_box = self.get_text_embeds(text_ids_box, text_atts_box)
            x_bs = image_embeds[:, 1:]
            region_atts = - box_atts + 1
            weights = region_atts[:, 1:].unsqueeze(2)  # B L 1
            x_bs_cls = torch.sum((weights * x_bs).transpose(1, 2), dim=-1, keepdim=True)  # B C 1
            x_bs_cls = x_bs_cls / torch.sum(weights.transpose(1, 2), dim=-1, keepdim=True)  # avgpool
            box_embeds = torch.cat([x_bs_cls.transpose(1, 2), x_bs], dim=1)

            image_feat_box, text_feat_box = self.get_image_feat(box_embeds), self.get_text_feat(text_embeds_box)
            loss_itc_box = self.get_contrastive_loss(image_feat_box, text_feat_box, label)
            loss_itm_box = self.get_matching_loss(box_embeds, region_atts, image_feat_box,
                                                  text_embeds_box, text_atts_box, text_feat_box, label)
            loss_region = 0.8 * (loss_itc_box + loss_itm_box)
        else:
            loss_region = 0


        image_feat, text_feat = self.get_image_feat(image_embeds), self.get_text_feat(text_embeds)
        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx)


        # eda
        if self.eda:
            text_embeds_eda = self.get_text_embeds(text_ids_eda, text_atts_eda)
            text_feat_eda = self.get_text_feat(text_embeds_eda)
            loss_itc_eda = self.get_contrastive_loss(image_feat, text_feat_eda, idx)
            loss_itm_eda = self.get_matching_loss(image_embeds, image_atts, image_feat,
                                                  text_embeds_eda, text_atts_eda, text_feat_eda, idx)
            loss_itc = loss_itc + 0.8 * loss_itc_eda
            loss_itm = loss_itm + 0.8 * loss_itm_eda


        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts[:bs], image_embeds[:bs], image_atts[:bs], masked_pos, masked_ids)


        return loss_itc, loss_itm, loss_mlm, loss_region
