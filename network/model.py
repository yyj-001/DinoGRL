import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP,GeneralizedMeanPooling
import torch

from torch.nn import functional as F
from .DINOv2 import vit_small
from einops import rearrange
from kornia import morphology as morph

import random
from .opengait_modules import SetBlockWrapper,SeparateFCs,SeparateBNNecks,PackSequenceWrapper,HorizontalPoolingPyramid,AttentionFusion,Pre_ResNet9,Post_ResNet9,setpooling

from .graph_convolution import Graph, MultiScale_GraphConv

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.in_planes = 2048
        self.num_classes = self.config.pid_num
        self.image_encoder1 = RGB_Model()
        self.image_encoder2 = IR_Model()
        self.share_encoder = Shared_Model()

        self.TP = PackSequenceWrapper(torch.max)
        #self.HPP = HorizontalPoolingPyramid(bin_num = [16])
        self.HPP = HorizontalPoolingPyramid(bin_num = [1,2,4,8,16])
        self.FCs = SeparateFCs(parts_num=31, in_channels=2048, out_channels=256)
        self.BNNecks = SeparateBNNecks(parts_num=35, in_channels=256, class_num=500)

    def forward(self, x1=None, x2=None, stage=None):
        if x1 is not None and x2 is not None:

            if stage == 1:
                image_features_map1 = self.image_encoder1(x1)
                image_features_map2 = self.image_encoder2(x2)
                image_features_maps = torch.cat([image_features_map1, image_features_map2], dim=0)
                image_features_maps = self.share_encoder(image_features_maps, stage)
                return image_features_maps
            
            if stage == 2:
                image_features_maps = self.share_encoder(x1, stage)

                c,h,w = image_features_maps.size()[-3:]
                image_features_maps = image_features_maps.view(image_features_maps.size(0)//6, 6, c, h, w).transpose(1, 2)
                image_features_maps = self.TP(image_features_maps, seqL=None, options={"dim": 2})[0]

                feature_1 , feature_2, feature_4, feature_8, feature_16 = self.HPP(image_features_maps)
                return feature_1 , feature_2, feature_4, feature_8, feature_16

            if stage == 3:
                embed = self.FCs(x1)

                return embed
            
            if stage == 4:

                _, logits = self.BNNecks(x1)
                return logits

        elif x1 is not None and x2 is None:

            if stage == 1:
                image_features_maps = self.image_encoder1(x1)
                image_features_maps = self.share_encoder(image_features_maps, stage)
                return image_features_maps
            
            if stage == 2:
                image_features_maps = self.share_encoder(x1, stage)

                c,h,w = image_features_maps.size()[-3:]
                image_features_maps = image_features_maps.view(image_features_maps.size(0)//6, 6, c, h, w).transpose(1, 2)
                image_features_maps = self.TP(image_features_maps, seqL=None, options={"dim": 2})[0]

                feature_1 , feature_2, feature_4, feature_8, feature_16 = self.HPP(image_features_maps)
                return feature_1 , feature_2, feature_4, feature_8, feature_16

            if stage == 3:
                embed = self.FCs(x1)

                return embed


        elif x1 is None and x2 is not None:

            if stage == 1:
                image_features_maps = self.image_encoder2(x2)
                image_features_maps = self.share_encoder(image_features_maps, stage)
                return image_features_maps
            
            if stage == 2:
                image_features_maps = self.share_encoder(x2, stage)

                c,h,w = image_features_maps.size()[-3:]
                image_features_maps = image_features_maps.view(image_features_maps.size(0)//6, 6, c, h, w).transpose(1, 2)
                image_features_maps = self.TP(image_features_maps, seqL=None, options={"dim": 2})[0]

                feature_1 , feature_2, feature_4, feature_8, feature_16 = self.HPP(image_features_maps)
                return feature_1 , feature_2, feature_4, feature_8, feature_16

            if stage == 3:
                embed = self.FCs(x2)

                return embed

    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        # features = features.squeeze()
        # c,h,w = features.size()[-3:]
        # features = features.view(features.size(0)//t, t, c, h, w).permute(1, 0, 2, 3, 4)
        # features = features.mean(0)
        return features





class Shape_Model(nn.Module):
    def __init__(self, config):
        super(Shape_Model, self).__init__()

        self.config = config
        self.dinov2 = Dinov2(self.config)

    def forward(self, x1=None, x2=None, stage=None):
        if x1 is not None and x2 is not None:
            if stage == 1:
                x = torch.cat((x1,x2),dim=0)
                embed_1, loss = self.dinov2(x, stage)
                return embed_1,loss
            if stage == 2:
                embed_1, embed_2, embed_4, embed_8, embed_16 = self.dinov2(x1, stage)
                return embed_1, embed_2, embed_4, embed_8, embed_16
            if stage == 3:
                embed_1 = self.dinov2(x1, stage)
                return embed_1
            if stage == 4:
                logits = self.dinov2(x1, stage)
                return logits
        
        elif x1 is not None and x2 is None:
            if stage == 1:
                embed_1, loss = self.dinov2(x1, stage)
                return embed_1
            if stage == 2:
                embed_1, embed_2, embed_4, embed_8, embed_16 = self.dinov2(x1, stage)
                return embed_1, embed_2, embed_4, embed_8, embed_16
            if stage == 3:
                embed_1 = self.dinov2(x1, stage)
                return embed_1
        
        elif x1 is None and x2 is not None:
            if stage == 1:
                embed_1, loss = self.dinov2(x2, stage)
                return embed_1
            if stage == 2:
                embed_1, embed_2, embed_4, embed_8, embed_16 = self.dinov2(x2, stage)
                return embed_1, embed_2, embed_4, embed_8, embed_16
            if stage == 3:
                embed_1 = self.dinov2(x2, stage)
                return embed_1


    def tem_pool(self,features,t=6):
        features = features.squeeze()
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features

    def tem_reshape(self,features,t=6):
        features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
        features = features.mean(0)
        return features


class RGB_Model(nn.Module):
    def __init__(self):
        super(RGB_Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, rgb):
        rgb_features_map = self.resnet_conv(rgb)
        return rgb_features_map

class IR_Model(nn.Module):
    def __init__(self):
        super(IR_Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, ir):
        ir_features_map = self.resnet_conv(ir)
        return ir_features_map

class Shared_Model(nn.Module):

    def __init__(self):
        super(Shared_Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet_conv = nn.Sequential(resnet.layer1,
                                         resnet.layer2, resnet.layer3, resnet.layer4)
        self.GAP = GeneralizedMeanPoolingP()



        # self.conv0 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=256,kernel_size=1, stride=1, padding=0),  nn.BatchNorm3d(256),  nn.ReLU(inplace=True))
        # self.conv1 = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=512,kernel_size=3, stride=(1,2,2), padding=1),  nn.BatchNorm3d(512),  nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(nn.Conv3d(in_channels=512, out_channels=1024,kernel_size=3, stride=(1,2,2), padding=1),nn.BatchNorm3d(1024),  nn.ReLU(inplace=True)) 
        # self.conv3 = nn.Sequential(nn.Conv3d(in_channels=1024, out_channels=2048,kernel_size=1, stride=1, padding=0),nn.BatchNorm3d(2048),  nn.ReLU(inplace=True))
        

    def forward(self, x, stage):
        if stage == 1:
            x1 = self.resnet_conv[0](x)
            # x1 = x1 + self.conv3d(x)

            x2 = self.resnet_conv[1](x1)
            # x2 = x2 + self.conv3d(x1)

            x3 = self.resnet_conv[2](x2)
            # x3 = x3 + self.conv3d(x2)
            return x3
        if stage == 2:
            x4 = self.resnet_conv[3](x)
            # x4 = x4 + self.conv3d(x)
            return x4
        
    # def conv3d(self,x):
    #     x = x.view(x.size(0)//6, x.size(1), 6, x.size(2), x.size(3))
    #     if x.size(1) == 64:
    #         x = self.conv0(x)
    #     elif x.size(1) == 256:
    #         x = self.conv1(x)
    #     elif x.size(1) == 512:
    #         x = self.conv2(x)
    #     elif x.size(1) == 1024:
    #         x = self.conv3(x)
    #     x = x.view(x.size(0)*6, x.size(1), x.size(3), x.size(4))
    #     return x



class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.GEM = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

        self.Gem = GeneralizedMeanPoolingP()

    def forward(self, features):

        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return features, cls_score
        else:
            return self.l2_norm(bn_features)
        # return features, cls_score, self.l2_norm(bn_features)
    












class gcn(nn.Module):
    def __init__(self):
        super(gcn, self).__init__()

        self.linked_edges_p2p = [(0,1),(0,2),
                                (1,3),(1,4),
                                (2,5),(2,6),
                                (3,7),(3,8),
                                (4,9),(4,10),
                                (5,11),(5,12),
                                (6,13),(6,14),
                                (7,15),(7,16),
                                (8,17),(8,18),
                                (9,19),(9,20),
                                (10,21),(10,22),
                                (11,23),(11,24),
                                (12,25),(12,26),
                                (13,27),(13,28),
                                (14,29),(14,30)]
        self.node_nums_part = 31
        self.part_num_scales = 5
        self.graph_p = Graph(self.node_nums_part, self.linked_edges_p2p)
        self.stgc_p2p = MultiScale_GraphConv(self.part_num_scales, 2048, 2048, self.graph_p.A_binary)


    def forward(self, feature):

        # input = torch.cat([
        #         feature_1,  # 节点 0
        #         embed_1,    # 节点 1
        #         feature_2,  # 节点 2, 3
        #         embed_2,    # 节点 4, 5
        #         feature_4,  # 节点 6, 7, 8, 9
        #         embed_4,    # 节点 10, 11, 12, 13
        #         feature_8,  # 节点 14, ..., 21
        #         embed_8,    # 节点 22, ..., 29
        #         feature_16, # 节点 30, ..., 45
        #         embed_16    # 节点 46, ..., 61
        #                     ], dim=2)
        # output = self.stgc_p2p(input)

        output = self.stgc_p2p(feature)
        return output






##################################################################################################################






class attn(nn.Module):
    def __init__(self):
        super(attn, self).__init__()

        # self.enhance_app_net_2 = Cross_Non_local_a(2048)
        # self.enhance_shape_net_2 = Cross_Non_local_s(2048)

        self.enhance_app_net_3 = Cross_Non_local_a(256)
        self.enhance_shape_net_3 = Cross_Non_local_s(256)

    def forward(self, shape_feature_map,app_feature_map=None, stage=None):


        if stage == 3:
            shape_2 = shape_feature_map[:,:,0]
            app_2 = app_feature_map[:,:,0]
            for i in range(1, 3):
                s = shape_feature_map[:,:,i]
                a = app_feature_map[:,:,i]

                shape_2 = self.enhance_shape_net_3(shape_2, a)
                app_2 = self.enhance_app_net_3(s, app_2)

            shape_4 = shape_feature_map[:,:,0]
            app_4 = app_feature_map[:,:,0]
            for i in range(3, 7):
                s = shape_feature_map[:,:,i]
                a = app_feature_map[:,:,i]

                shape_4 = self.enhance_shape_net_3(shape_4, a)
                app_4 = self.enhance_app_net_3(s, app_4)

            shape_8 = shape_feature_map[:,:,0]
            app_8 = app_feature_map[:,:,0]
            for i in range(7, 15):
                s = shape_feature_map[:,:,i]
                a = app_feature_map[:,:,i]

                shape_8 = self.enhance_shape_net_3(shape_8, a)
                app_8 = self.enhance_app_net_3(s, app_8)

            shape_16 = shape_feature_map[:,:,0]
            app_16 = app_feature_map[:,:,0]
            for i in range(15, 31):
                s = shape_feature_map[:,:,i]
                a = app_feature_map[:,:,i]

                shape_16 = self.enhance_shape_net_3(shape_16, a)
                app_16 = self.enhance_app_net_3(s, app_16)

            shape_feature_map = torch.cat((shape_feature_map, shape_2.unsqueeze(2), shape_4.unsqueeze(2), shape_8.unsqueeze(2), shape_16.unsqueeze(2)),dim=2)
            app_feature_map = torch.cat((app_feature_map, app_2.unsqueeze(2), app_4.unsqueeze(2), app_8.unsqueeze(2), app_16.unsqueeze(2)), dim=2)

            return shape_feature_map,app_feature_map



class Cross_Non_local_a(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Cross_Non_local_a, self).__init__()

        # self.in_channels = in_channels
        # self.inter_channels = reduc_ratio//reduc_ratio
        self.in_channels = 1
        self.inter_channels = in_channels

        self.g = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )
        #####
        self.W1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels),
        )
        nn.init.constant_(self.W1[1].weight, 0.0)
        nn.init.constant_(self.W1[1].bias, 0.0)

        self.W2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels),
        )
        nn.init.constant_(self.W2[1].weight, 0.0)
        nn.init.constant_(self.W2[1].bias, 0.0)
        #####
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, s,a):
        s = s.unsqueeze(1)
        a = a.unsqueeze(1)

        batch_size = a.size(0)
        g_x = self.g(s).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(a).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        #################################
        ReLU = nn.ReLU()
        f_pos = ReLU(f_div_C)
        f_neg = ReLU(-f_div_C)

        y_share = torch.matmul(f_pos, g_x)
        y_share = y_share.permute(0, 2, 1).contiguous()
        y_spec  = torch.matmul(f_neg, g_x)
        y_spec  = y_spec.permute(0, 2, 1).contiguous()

        y_share = self.W1(y_share).squeeze()
        y_spec  = self.W2(y_spec).squeeze()

        y_fusion = a.squeeze() + y_share - y_spec
        return y_fusion
        #################################

        # y = torch.matmul(f_div_C, g_x)
        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *a.size()[2:])
        # W_y = self.W(y).squeeze()
        # z = W_y + a.squeeze()

        # return z



class Cross_Non_local_s(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Cross_Non_local_s, self).__init__()

        # self.in_channels = in_channels
        # self.inter_channels = reduc_ratio//reduc_ratio
        self.in_channels = 1
        self.inter_channels = in_channels

        self.g = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )
        ######
        self.W1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels),
        )
        nn.init.constant_(self.W1[1].weight, 0.0)
        nn.init.constant_(self.W1[1].bias, 0.0)

        self.W2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels),
        )
        nn.init.constant_(self.W2[1].weight, 0.0)
        nn.init.constant_(self.W2[1].bias, 0.0)
        ######

        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, s,a):
        s = s.unsqueeze(1)
        a = a.unsqueeze(1)

        batch_size = a.size(0)
        g_x = self.g(a)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(s).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(a).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        #################################
        ReLU = nn.ReLU()
        f_pos = ReLU(f_div_C)
        f_neg = ReLU(-f_div_C)

        y_share = torch.matmul(f_pos, g_x)
        y_share = y_share.permute(0, 2, 1).contiguous()
        y_spec  = torch.matmul(f_neg, g_x)
        y_spec  = y_spec.permute(0, 2, 1).contiguous()

        y_share = self.W1(y_share).squeeze()
        y_spec  = self.W2(y_spec).squeeze()

        y_fusion = s.squeeze() + y_share - y_spec
        return y_fusion
        #################################
        # y = torch.matmul(f_div_C, g_x)
        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *s.size()[2:])
        # W_y = self.W(y).squeeze()
        # z = W_y + s.squeeze()

        # return z







class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op



class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None


def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(need_padding_mask, (w - need_w) // 2, torch.tensor(0).to(x.device))
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.tensor(0).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()
    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x


class Dinov2(nn.Module):
    def __init__(self,config):
        super(Dinov2, self,).__init__()
        self.config = config



        self.image_size = 224
        self.sils_size = 32
        self.f4_dim = 384
        self.mask_dim = 2
        self.fc_dim = self.f4_dim*4
        self.denoising_dim = 16
        self.app_dim = 16


        self.dinov2 = vit_small(logger = NoOp())
        pretrain_dict = torch.load(self.config.dinov2_model)
        self.dinov2.load_state_dict(pretrain_dict, strict=True)

        self.Mask_Branch = infoDistillation(source_dim=384, target_dim=2, p=0.5, softmax=True, Relu=False, Up=True)
        load_dict = torch.load(self.config.mask_model, map_location=torch.device("cpu"))['model']
        self.Mask_Branch.load_state_dict(load_dict, strict=True)

        self.Denoising_Branch = infoDistillation(source_dim=1536, target_dim=16, p=0, softmax=True, Relu=True, Up=False)
        self.Appearance_Branch = infoDistillation(source_dim=1536, target_dim=16, p=0, softmax=False, Relu=False, Up=False)

        self.gait_net = Baseline(self.config)

    # resize image
    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size*2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    # cal foreground
    def get_body(self, mask):
        # value: [0,1]  shape: [nxs, h, w, c]
        def judge_edge(image, edge=1):
            # [nxs,h,w]
            edge_pixel_count = image[:, :edge, :].sum(dim=(1,2)) + image[:, -edge:, :].sum(dim=(1,2))
            return edge_pixel_count > (image.size(2)) * edge
        condition_mask = torch.round(mask[...,0]) - mask[...,0].detach() + mask[...,0]
        condition_mask = judge_edge(condition_mask, 5)
        mask[condition_mask, :, :, 0] = mask[condition_mask, :, :, 1]
        return mask[...,0]
    
    def connect_loss(self, images, n, s, c):
        images = images.view(n*s,c,self.sils_size*2,self.sils_size)
        gradient_x = F.conv2d(images, torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        gradient_y = F.conv2d(images, torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None,None,...].repeat(1,c,1,1).to(images.dtype).to(images.device), padding=1)
        loss_connectivity = (torch.sum(torch.abs(gradient_x)) + torch.sum(torch.abs(gradient_y))) / (n*s*c*self.sils_size*2*self.sils_size)
        return loss_connectivity
    
    # Binarization and Closing operations to enhance foreground
    def get_edge(self, sils, threshold=1):
        mask_sils = torch.round(sils * threshold)
        kernel = torch.ones((3,3))
        dilated_mask = morph.dilation(mask_sils, kernel.to(sils.device)).detach()  # Dilation
        kernel = torch.ones((5,5))
        eroded_mask = morph.erosion(dilated_mask, kernel.to(sils.device)).detach()  # Erosion
        edge_mask = (dilated_mask > 0.5) ^ (eroded_mask > 0.5)
        sils = edge_mask * sils + (eroded_mask > 0.5) * torch.ones_like(sils, dtype=sils.dtype, device=sils.device)
        return sils

    def diversity_loss(self, images, max_p):
        # [ns, hw, c]
        p = torch.sum(images, dim=1) / (torch.sum(images, dim=(1,2)) + 1e-6).view(-1,1).repeat(1,max_p)
        entropies = -torch.sum(p * torch.log2(p + 1e-6), dim=1)
        max_p = torch.Tensor([1/max_p]).repeat(max_p).to(images.dtype).to(images.device)
        max_entropies = -torch.sum(max_p * torch.log2(max_p), dim=0)
        return torch.mean(max_entropies - entropies)

    def forward(self, inputs, stage=None):

        if stage == 1:
            sils = inputs
            ns,c,h,w = sils.size()
            s=6
            n=ns//6
            ratios = torch.full((n, s, 1), 0.5).to('cuda')
            with torch.no_grad():

                if h == 2*w:
                    outs = self.preprocess(sils, self.image_size)                                           # [ns,c,448,224]    if have used pad_resize for input images
                else:
                    outs = self.preprocess(padding_resize(sils, ratios, 256, 128), self.image_size)         # [ns,c,448,224]    if have not used pad_resize for input images
                outs = self.dinov2(outs, is_training=True) # [ns,h*w,c]
                outs_last1 = outs["x_norm_patchtokens"].contiguous()
                outs_last4 = outs["x_norm_patchtokens_mid4"].contiguous()

                outs_last1 = rearrange(outs_last1.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                outs_last4 = rearrange(outs_last4.view(n, s, self.image_size//7, self.image_size//14, -1), 'n s h w c -> (n s) c h w').contiguous()
                outs_last1 = self.preprocess(outs_last1, self.sils_size) # [ns,c,64,32]
                outs_last4 = self.preprocess(outs_last4, self.sils_size) # [ns,c,64,32]
                outs_last1 = rearrange(outs_last1.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()
                outs_last4 = rearrange(outs_last4.view(n, s, -1, self.sils_size*2, self.sils_size), 'n s c h w -> (n s) (h w) c').contiguous()

            # get foreground
            mask = torch.ones_like(outs_last1[...,0], device=outs_last1.device, dtype=outs_last1.dtype).view(n*s,1,self.sils_size*2,self.sils_size)
            mask = padding_resize(mask, ratios, self.sils_size*2, self.sils_size)
            foreground = outs_last1.view(-1, self.f4_dim)[mask.view(-1) != 0]
            fore_feat, loss_mse1 = self.Mask_Branch(foreground)
            foreground = torch.zeros_like(mask, dtype=fore_feat.dtype, device=fore_feat.device).view(-1,1).repeat(1,self.mask_dim)
            foreground[mask.view(-1) != 0] = fore_feat
            loss_connectivity_shape = self.connect_loss(foreground, n, s, self.mask_dim)
            foreground = foreground.detach().clone()
            foreground = self.get_body(foreground.view(n*s,self.sils_size*2,self.sils_size,self.mask_dim)).view(n*s,-1) # [n*s,h*w]
            foreground = self.get_edge(foreground.view(n*s,1,self.sils_size*2,self.sils_size)).view(n*s,-1) # [n*s,h*w]
            del fore_feat, mask

            # get denosing
            denosing = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
            den_feat, _ = self.Denoising_Branch(denosing)
            denosing = torch.zeros_like(foreground, dtype=den_feat.dtype, device=den_feat.device).view(-1,1).repeat(1,self.denoising_dim)
            denosing[foreground.view(-1) != 0] = den_feat
            loss_connectivity_part = self.connect_loss(denosing.view(n*s,-1,self.denoising_dim)[...,:-1].permute(0,2,1), n, s, (self.denoising_dim-1))
            loss_diversity_part = self.diversity_loss(denosing.view(n*s,-1,self.denoising_dim), self.denoising_dim)
            del den_feat

            # get appearance
            appearance = outs_last4.view(-1, self.fc_dim)[foreground.view(-1) != 0]
            app_feat, _ = self.Appearance_Branch(appearance)
            appearance = torch.zeros_like(foreground, dtype=app_feat.dtype, device=app_feat.device).view(-1,1).repeat(1,self.app_dim)
            appearance[foreground.view(-1) != 0] = app_feat
            appearance = appearance.view(n*s,-1,self.app_dim)
            del app_feat

            # vis_num = n*s
            # vis_mask = foreground.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()
            # vis_denosing = pca_image(data={'embeddings':denosing.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True) # n s c h w
            # vis_appearance = pca_image(data={'embeddings':appearance.view(n*s, self.sils_size*2*self.sils_size, -1)[:vis_num].detach().cpu().numpy()}, mask=vis_mask, root=None, model_name=None, dataset=None, n_components=3, is_return=True) # n s c h w

            # image = torch.from_numpy(vis_denosing)
            # image = F.interpolate(image.squeeze(), size=(288, 144), mode='bilinear', align_corners=False)
            # image = image.float().to(denosing.device).requires_grad_(True)
            # denosing = denosing.view(n*s,16,64,32)
            # denosing = F.interpolate(denosing, size=(288, 144), mode='bilinear', align_corners=False)

            if self.training:
                mask_idx = random.sample(list(range(n)), int(round(n*0.2)))
                feat_list = [denosing.view(n,s,-1), appearance.view(n,s,-1)]
                for i in mask_idx:
                    idx = random.sample(list(range(2)), 1)
                    for j in idx:
                        feat_list[j][i] = torch.zeros_like(feat_list[j][i], device=feat_list[j].device, dtype=feat_list[j].dtype)

            embed_1 = self.gait_net(
                                            denosing.view(n,s,self.sils_size*2,self.sils_size,self.denoising_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                            appearance.view(n,s,self.sils_size*2,self.sils_size,self.app_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                            seqL=None,
                                            stage=stage
                                            )
            
            return embed_1,  (loss_connectivity_shape*0.02)+loss_mse1+(loss_connectivity_part*0.01)+(loss_diversity_part*5)
        
        if stage == 2:
            embed_1, embed_2, embed_4, embed_8, embed_16 = self.gait_net(
                                            inputs,
                                            None,
                                            seqL=None,
                                            stage=stage
                                            )
            
            return embed_1, embed_2, embed_4, embed_8, embed_16
        
        if stage == 3:
            embed_1 = self.gait_net(
                                            inputs,
                                            None,
                                            seqL=None,
                                            stage=stage
                                            )
            
            return embed_1
    
        if stage == 4:
            logits = self.gait_net(
                                            inputs,
                                            None,
                                            seqL=None,
                                            stage=stage
                                            )
            
            return logits





class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init__()
        self.config = config

        self.pre_part = SetBlockWrapper(Pre_ResNet9())
        self.pre_rgb = SetBlockWrapper(Pre_ResNet9())
        # self.pre_part = RGB_Model()
        # self.pre_rgb = IR_Model()

        #self.post_backbone = SetBlockWrapper(Post_ResNet9())
        self.post_backbone = Shared_Model()

        self.FCs = SeparateFCs(parts_num=31, in_channels=2048, out_channels=256)
        self.BNNecks = SeparateBNNecks(parts_num=35, in_channels=256, class_num=self.config.pid_num)
        self.TP = PackSequenceWrapper(torch.max)
        #self.TP = setpooling()
        #self.HPP = HorizontalPoolingPyramid(bin_num = [16])
        self.HPP = HorizontalPoolingPyramid(bin_num = [1,2,4,8,16])

        self.fusion = AttentionFusion(in_channels=64, squeeze_ratio=16, feat_len=2)

        self.size = None

        self.gcn_s = gcn()
        self.gcn_a = gcn()


    def forward(self, denosing, appearance, seqL, stage=None):
        if stage == 1:
            denosing = self.pre_part(denosing)  # [n, c, s, h, w]
            appearance = self.pre_rgb(appearance)  # [n, c, s, h, w]
            outs = self.fusion([denosing, appearance])
            # heat_mapt = rearrange(outs, 'n c s h w -> n s h w c')
            del denosing, appearance
            self.size = outs.size()
            outs = outs.transpose(1, 2).reshape(-1, self.size[1], self.size[3], self.size[4])
            outs = self.post_backbone(outs, stage)
            return outs

        if stage == 2:
            outs = self.post_backbone(denosing, stage)

            output_size = outs.size()
            outs = outs.reshape(self.size[0], self.size[2], *output_size[1:]).transpose(1, 2).contiguous()

            # Temporal Pooling, TP
            outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

            # Horizontal Pooling Matching, HPM
            embed_1, embed_2, embed_4, embed_8, embed_16 = self.HPP(outs)  # [n, c, p]

            return embed_1, embed_2, embed_4, embed_8, embed_16

        if stage == 3:
            outs = denosing

            # shape = outs[:,:,:31]
            # app = outs[:,:,31:]

            # shape = self.gcn_s(shape)
            # app = self.gcn_a(app)

            # outs = torch.cat((shape,app),dim=2)

            embed_1 = self.FCs(outs)  # [n, c, p]

            return embed_1
        
        if stage == 4:
            _, logits = self.BNNecks(denosing)  # [n, c, p]
            return logits






