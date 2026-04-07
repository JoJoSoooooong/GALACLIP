from models.segmentor.galaclip import GalaCLIP

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg
from models.decode_heads.gala_fusion_head import GalaFusionHead

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.context59 import ZeroPascalVOCDataset59
