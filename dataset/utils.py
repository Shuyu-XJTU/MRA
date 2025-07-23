import re

from dataset.eda import *


def pre_caption(caption, max_words, icfg_rstp=False, is_eda=False):
    if icfg_rstp:
        try:
            caption_new = re.sub(
                r'[^0-9a-z]+',
                ' ',
                caption.lower(),
            )
        except:
            print('Wrong: when deal with', caption)
    else:
        caption_new = caption

    # eda
    if is_eda:
        caption_new = eda(caption_new, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]

    # truncate caption
    caption_words = caption_new.split()
    if len(caption_words) > max_words:
        caption_new = ' '.join(caption_words[:max_words])

    if not len(caption_new):
        raise ValueError("pre_caption yields invalid text")

    return caption_new
