import torch
import torch.nn.functional as F
import copy
from torch.cuda import amp


import cv2

def B32B2(x):
    B = x.size(0)//3
    x = torch.cat([x[:B],x[2*B:]],dim=0)
    return x
def B22B3(x):
    B = x.size(0)//2
    x_3b = torch.cat([x[:B],x[:B],x[B:]],dim=0)
    return x_3b

def shape_enhance_app(base,shape_branch_feature,features_map,features):
    shape_branch_feature = B22B3(shape_branch_feature)
    x = base.shape_enhance_net(shape_branch_feature,features_map,features)
    return x

def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def tem_pool(features,t=6):
    features = features.squeeze()
    features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
    features = features.mean(0)
    return features

def tem_reshape(features):
    features = features.view(features.size(0)*features.size(1), -1)
    return features

def copy_rgb(features):
    B = features.size(0)//2
    new_features = torch.cat([features[:B],features[:B],features[B:]])
    return new_features









def foward_video(iter,base,meter,scaler):

    for _ in range(base.steps):
        input1, input2, label1, label2 = iter.next_one()
        rgb_imgs, rgb_pids = input1, label1
        ir_imgs, ir_pids = input2, label2
        rgb_imgs,  rgb_pids = rgb_imgs.to(base.device),rgb_pids.to(base.device).long()
        ir_imgs, ir_pids = ir_imgs.to(base.device), ir_pids.to(base.device).long()


        rgb_imgs = process_vedio(rgb_imgs)
        ir_imgs = process_vedio(ir_imgs)

        with amp.autocast(enabled=True):

            pids = torch.cat([rgb_pids, ir_pids], dim=0)
            embed_1, dino_loss = base.shape_model(x1=rgb_imgs, x2=ir_imgs, stage=1)
            feature = base.model(x1=rgb_imgs, x2=ir_imgs, stage=1)

            # embed_1, feature = base.model2(embed_1, feature, stage=1)

            embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=embed_1, x2=embed_1, stage=2)
            feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feature, x2=feature, stage=2)

            # embed_1, feature = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
            embed_1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
            feature = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)

            # embed_1, feature = base.model2(embed_1, feature, stage=2)

            embed_1 = base.shape_model(x1=embed_1, x2=embed_1, stage=3)
            feature = base.model(x1=feature, x2=feature, stage=3)

            embed_1, feature = base.model2(embed_1, feature, stage=3)

            logits = base.shape_model(x1=embed_1, x2=embed_1, stage=4)
            cls_score = base.model(x1=feature, x2=feature, stage=4)

            ide_loss, accu = base.ce_loss(cls_score, pids)
            triplet_loss = base.tr_loss(feature, pids)
            triplet_loss = triplet_loss.mean()

            sid_loss, s_accu = base.ce_loss(logits, pids)
            stri_loss = base.tr_loss(embed_1, pids)
            stri_loss = stri_loss.mean()

            total_loss = dino_loss + sid_loss + stri_loss + ide_loss + triplet_loss

        for optimizer in base.optimizers:
            optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        for optimizer in base.optimizers:
            scaler.step(optimizer)
        scaler.update()

        meter.update({
                      'dino_loss':dino_loss,
                      'id_loss': ide_loss.data,
                      'tri_loss': triplet_loss.data,
                      'accu':accu,

                      'sid_loss':sid_loss,
                      'stri_loss':stri_loss,
                      's_accu':s_accu
                      })
    return meter





