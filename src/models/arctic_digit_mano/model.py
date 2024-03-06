import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR

import torch
import torchvision.models as models
import src.nets.digit.hrnet as hrnet
import src.nets.digit.layer as layer
from src.nets.digit.unet import MiniUNet

class SegmHead(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, class_dim):
        super().__init__()

        # upsample features
        self.upsampler = layer.UpSampler(in_dim, hidden_dim1, hidden_dim2)

        segm_net = layer.DoubleConv(hidden_dim2, class_dim)
        segm_net.double_conv = segm_net.double_conv[:4]
        self.segm_net = segm_net

    def forward(self, img_feat):
        # feature up sample to 256
        hr_img_feat = self.upsampler(img_feat)
        segm_logits = self.segm_net(hr_img_feat)
        return {'segm_logits': segm_logits}


class SegmNet(nn.Module):
    def __init__(self):
        super(SegmNet, self).__init__()
        self.segm_head = SegmHead(32, 128, 64, 33)

    def map2labels(self, segm_hand):
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand

    def forward(self, img_feat):
        segm_dict = self.segm_head(img_feat)
        segm_logits = segm_dict['segm_logits']

        segm_mask = self.map2labels(segm_logits)

        segm_dict['segm_mask'] = segm_mask
        segm_dict['segm_logits'] = segm_logits
        return segm_dict


class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-4])
        self.conv1 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = self.unpool(x)
        return x

 
class ArcticDigitMano(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(ArcticDigitMano, self).__init__()
        self.args = args
        self.use_resnet50 = False
        if backbone == "hrnet":
            self.backbone = hrnet.get_pose_net()
        elif backbone == "resnet50":
            self.use_resnet50 = True
            from src.nets.backbone.resnet import resnet50 as resnet50
            self.backbone = resnet50(pretrained=True)
            self.x_digit = nn.Sequential(
                nn.Conv2d(2048, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='nearest'),
                nn.Conv2d(256, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
            )
        else:
            self.backbone = ModifiedResNet50()
        
        self.hand_segm = SegmNet()
        self.conv_segm_feat = nn.Sequential(
            nn.Conv2d(33, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        in_channel = 32 + 512
        self.merge_layer = MiniUNet(in_channel, in_channel)
        self.conv_x = nn.Sequential(
            nn.Conv2d(544, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 2048, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        feat_dim = 2048
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)

        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        # features = self.backbone(images)              # [8, 2048, 7, 7]
        
        img = self.backbone(images)                     # [8, 32, 56, 56]
        if self.use_resnet50:
            img = self.x_digit(img)
        
        segm_dict = self.hand_segm(img)
        segm_logits = segm_dict["segm_logits"]          # [8, 33, 112, 112]
        segm_feat = self.conv_segm_feat(segm_logits)    # [8, 512, 56, 56]
        img = torch.cat((img, segm_feat), dim=1)        # [8, 544, 56, 56]
        img = self.merge_layer(img)                     # [8, 544, 56, 56]
        features = self.conv_x(img)                     # [8, 2048,16, 16]

        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        hmr_output_r = self.head_r(features)
        hmr_output_l = self.head_l(features)
        hmr_output_o = self.head_o(features)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_o["rot"],
            angle=hmr_output_o["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]
        root_o_init = hmr_output_o["cam_t.wp.init"]
        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init
        arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        output["segm_logits"] = segm_logits
        return output
