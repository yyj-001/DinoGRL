
import numpy as np
import torch
from torch.autograd import Variable
from tools import eval_regdb, eval_sysu,evaluate_vcm,MODALITY_,evaluate_bupt
import os

from einops import rearrange
import torch.nn.functional as F
from kornia import morphology as morph

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


image_size = 224
sils_size = 32
f4_dim = 384
mask_dim = 2
fc_dim = f4_dim*4
denoising_dim = 16
app_dim = 16



def test(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat = np.zeros((loader.n_query, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(loader.query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = base.ir_model(input)
            feat = base.shared_model(feat)
            feat = base.classifier(feat)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc = 0
        all_mAP = 0
        all_mINP = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            gall_feat = np.zeros((loader.n_gallery, 2048))
            with torch.no_grad():
                for batch_idx, (input, label) in enumerate(gall_loader):
                    batch_num = input.size(0)
                    input = Variable(input.cuda())
                    feat = base.rgb_model(input)
                    feat = base.shared_model(feat)
                    feat = base.classifier(feat)
                    gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

                    ptr = ptr + batch_num
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_sysu(-distmat, loader.query_label, loader.gall_label, loader.query_cam,
                                       loader.gall_cam)
            all_cmc += cmc
            all_mAP += mAP
            all_mINP += mINP
        all_cmc /= 10.0
        all_mAP /= 10.0
        all_mINP /= 10.0

    elif loader.dataset == 'regdb':
        gall_loader = loader.gallery_loaders
        gall_feat = np.zeros((loader.n_gallery, 2048))
        ptr = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat = base.rgb_model(input)
                feat = base.shared_model(feat)
                feat = base.classifier(feat)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

                ptr = ptr + batch_num
        if config.regdb_test_mode == 't-v':
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.query_label, loader.gall_label)
        else:
            distmat = np.matmul(gall_feat, np.transpose(query_feat))
            cmc, mAP, mINP = eval_regdb(-distmat, loader.gall_label, loader.query_label)

        all_cmc, all_mAP, all_mINP = cmc, mAP, mINP


    return all_cmc, all_mAP, all_mINP


def process_vedio(x1,seq_len=6):
    b, c, h, w = x1.size()
    x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
    return x1

def tem_pool(features,t=6):
    features = features.squeeze()
    features = features.view(features.size(0)//t, t, -1).permute(1, 0, 2)
    features = features.mean(0)
    return features





def test_vcm(base, loader, t2v=True):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []

    # if config.vcm_test_mode == 't2v':
    #     t2v = True
    # else:
    #     t2v = False

    if t2v:
        query_loader = loader.query_loader
        query_feat = np.zeros((loader.samples.num_query_tracklets, 2048))
    else:
        query_loader = loader.query_loader_1
        query_feat = np.zeros((loader.samples.num_query_tracklets_1, 2048))

    with torch.no_grad():
        for batch_idx, (input, label,c_label) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)

            if t2v:
                feat = base.model(x2=input)
                feat = feat[:,:,0]
            else:
                feat = base.model(x1=input)
                feat = feat[:,:,0]

            query = feat

            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'vcm':
        g_pids = []
        g_camids = []

        ptr = 0
        if t2v:
            gall_loader = loader.gallery_loader
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets, 2048))
        else:
            gall_loader = loader.gallery_loader_1
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets_1, 2048))

        with torch.no_grad():
            for batch_idx, (input, label,c_label) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                input = Variable(input.cuda())
                input = process_vedio(input)

                if t2v:
                    feat = base.model(x1=input)
                    feat = feat[:,:,0]
                else:
                    feat = base.model(x2=input)
                    feat = feat[:,:,0]

                gall = feat

                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc2, mAP2 = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
                                  g_camids)

        all_cmc  = cmc2
        all_mAP  = mAP2

        if t2v:
            str = print_metrics(
                all_cmc, all_mAP,
                prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
            )
        else:
            str = print_metrics(
                all_cmc, all_mAP,
                prefix='{:<3}->{:<3}:  '.format('RGB', 'IR')
            )
    return all_cmc, all_mAP, str


def print_metrics(cmc, ap, prefix=''):
    str = '{}mAP: {:.2%} | Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}.'.format(prefix, ap, cmc[0], cmc[4], cmc[9], cmc[19])
    return str

def test_bupt(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []
    q_modalitys = []
    query_feats = []
    if config.vcm_test_mode == 't2v':
        t2v = True
    else:
        t2v = False


    query_loader = loader.query_loader
    query_feat = np.zeros((1048, 2048))


    with torch.no_grad():
        for batch_idx, (input, label,c_label,modals) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            q_modalitys.extend(modals)
            input = Variable(input.cuda())
            input = process_vedio(input)
            if modals[0]==0:
                feat = base.rgb_model(input)
            elif modals[0]==1:
                feat = base.ir_model(input)

            features_map,features = base.shared_model(feat)
            features = tem_pool(features)

            feat = base.classifier(features)
            query = feat

            query_feat[ptr:ptr + batch_num, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_modalitys = np.asarray(q_modalitys)
        # query_feat = torch.cat(query_feats, dim=0)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'bupt':
        g_pids = []
        g_camids = []
        g_modalitys = []
        # gall_feats = []
        ptr = 0

        gall_loader = loader.gallery_loader
        gall_feat = np.zeros((4790, 2048))


        with torch.no_grad():
            for batch_idx, (input, label,c_label,modals) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                g_modalitys.extend(modals)
                input = Variable(input.cuda())
                input = process_vedio(input)
                if modals[0]==0:
                    feat = base.rgb_model(input)
                elif modals[1]==1:
                    feat = base.ir_model(input)

                features_map,features = base.shared_model(feat)
                features = tem_pool(features)
                feat = base.classifier(features)
                gall = feat
                gall_feat[ptr:ptr + batch_num, :] = gall.detach().cpu().numpy()
                # gall_feats.append(gall.detach().cpu().numpy())
                ptr = ptr + batch_num
        # gall_feat = torch.cat(gall_feats, dim=0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_modalitys = np.asarray(g_modalitys)
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        # evaluate (intra/inter-modality)
        CMC, MAP = [], []
        eval_str = []
        for q_modal in (0, 1):
            for g_modal in (0, 1):
                if q_modal!=g_modal:
                    q_mask = q_modalitys == q_modal
                    g_mask = g_modalitys == g_modal
                    tmp_distance = distmat[q_mask, :][:, g_mask]
                    tmp_qid = q_pids[q_mask]
                    tmp_gid = g_pids[g_mask]
                    # tem_qcid = q_camids[q_mask]
                    # tem_gcid = g_camids[g_mask]
                    tmp_cmc, tmp_ap = evaluate_bupt(-tmp_distance, tmp_qid, tmp_gid, config)
                    CMC.append(tmp_cmc * 100)
                    MAP.append(tmp_ap * 100)

                    str = print_metrics(
                        tmp_cmc, tmp_ap,
                        prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                    )
                    eval_str.append(str)




    #     cmc2, mAP2 = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
    #                               g_camids)
    #
    #     all_cmc  = cmc2
    #     all_mAP  = mAP2
    #     all_mINP = mAP2
    #
    return CMC, MAP,eval_str






def test_bupt_gait(base, loader, config):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []
    q_modalitys = []
    query_feats = []
    if config.vcm_test_mode == 't2v':
        t2v = True
    else:
        t2v = False


    query_loader = loader.query_loader
    query_feat = np.zeros((1048, 256, 62))


    with torch.no_grad():
        for batch_idx, (input, label,c_label,modals) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            q_modalitys.extend(modals)
            input = Variable(input.cuda())
            input = process_vedio(input)
            if modals[0]==0:
                feat1 = base.shape_model(x1=input, stage=1)
                feat2 = base.model(x1=input, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)

                feat = base.shape_model(x1=feat1, stage=3)

            elif modals[0]==1:
                feat1 = base.shape_model(x2=input, stage=1)
                feat2 = base.model(x2=input, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)

                feat = base.shape_model(x2=feat1, stage=3)

            query = feat

            query_feat[ptr:ptr + batch_num, :, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_modalitys = np.asarray(q_modalitys)
        # query_feat = torch.cat(query_feats, dim=0)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'bupt':
        g_pids = []
        g_camids = []
        g_modalitys = []
        # gall_feats = []
        ptr = 0

        gall_loader = loader.gallery_loader
        gall_feat = np.zeros((4790, 256, 62))


        with torch.no_grad():
            for batch_idx, (input, label,c_label,modals) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                g_modalitys.extend(modals)
                input = Variable(input.cuda())
                input = process_vedio(input)
                if modals[0]==0:
                    feat1 = base.shape_model(x1=input, stage=1)
                    feat2 = base.model(x1=input, stage=1)

                    embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                    feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                    feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                    feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                    feat1 = torch.cat((feat1, feat2),dim=2)

                    feat = base.shape_model(x1=feat1, stage=3)
                elif modals[1]==1:
                    feat1 = base.shape_model(x2=input, stage=1)
                    feat2 = base.model(x2=input, stage=1)

                    embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                    feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                    feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                    feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                    feat1 = torch.cat((feat1, feat2),dim=2)

                    feat = base.shape_model(x2=feat1, stage=3)

                gall = feat
                gall_feat[ptr:ptr + batch_num, :, :] = gall.detach().cpu().numpy()
                # gall_feats.append(gall.detach().cpu().numpy())
                ptr = ptr + batch_num
        # gall_feat = torch.cat(gall_feats, dim=0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_modalitys = np.asarray(g_modalitys)

        #distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat = -gait_dist(query_feat, gall_feat).cpu().numpy()

        # evaluate (intra/inter-modality)
        CMC, MAP = [], []
        eval_str = []
        for q_modal in (0, 1):
            for g_modal in (0, 1):
                if q_modal!=g_modal:
                    q_mask = q_modalitys == q_modal
                    g_mask = g_modalitys == g_modal
                    tmp_distance = distmat[q_mask, :][:, g_mask]
                    tmp_qid = q_pids[q_mask]
                    tmp_gid = g_pids[g_mask]
                    # tem_qcid = q_camids[q_mask]
                    # tem_gcid = g_camids[g_mask]
                    tmp_cmc, tmp_ap = evaluate_bupt(-tmp_distance, tmp_qid, tmp_gid, config)
                    CMC.append(tmp_cmc * 100)
                    MAP.append(tmp_ap * 100)

                    str = print_metrics(
                        tmp_cmc, tmp_ap,
                        prefix='{:<3}->{:<3}:  '.format(MODALITY_[q_modal], MODALITY_[g_modal])
                    )
                    eval_str.append(str)




    #     cmc2, mAP2 = evaluate_vcm(-distmat, q_pids, g_pids, q_camids,
    #                               g_camids)
    #
    #     all_cmc  = cmc2
    #     all_mAP  = mAP2
    #     all_mINP = mAP2
    #
    return CMC, MAP,eval_str






def test_bu(base, loader):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []

    query_loader = loader.query_loader
    query_feat = np.zeros((1048, 256,62))

    with torch.no_grad():
        for batch_idx, (input, label,c_label,modality) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)

            if modality[0] == 1:
                feat1 = base.shape_model(x2=input, stage=1)
                feat2 = base.model(x2=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)

                feat = base.shape_model(x2=feat1, stage=3)
                # feat2 = base.model(x2=feat2, stage=3)
                # feat1, feat2 = base.model2(feat1, feat2, stage=3)
                # feat = torch.cat((feat1, feat2),dim=2)
            elif modality[0] == 0:
                feat1 = base.shape_model(x1=input, stage=1)
                feat2 = base.model(x1=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)

                feat = base.shape_model(x1=feat1, stage=3)
                # feat2 = base.model(x1=feat2, stage=3)
                # #feat1, feat2 = base.model2(feat1, feat2, stage=3)
                # feat = torch.cat((feat1, feat2),dim=2)

            query = feat

            query_feat[ptr:ptr + batch_num, :, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print('Extracting Gallery Feature...')

    g_pids = []
    g_camids = []
    ptr = 0

    gall_loader = loader.gallery_loader
    gall_feat = np.zeros((4790, 256,62))

    with torch.no_grad():
        for batch_idx, (input, label,c_label,modality) in enumerate(gall_loader):
            batch_num = input.size(0)
            g_pids.extend(label)
            g_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)

            if modality[0] == 0:
                feat1 = base.shape_model(x1=input, stage=1)
                feat2 = base.model(x1=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)

                feat = base.shape_model(x1=feat1, stage=3)
                # feat2 = base.model(x1=feat2, stage=3)
                # #feat1, feat2 = base.model2(feat1, feat2, stage=3)
                # feat = torch.cat((feat1, feat2),dim=2)

            elif modality[0] == 1:
                feat1 = base.shape_model(x2=input, stage=1)
                feat2 = base.model(x2=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                feat1 = torch.cat((feat1, feat2),dim=2)
                
                feat = base.shape_model(x2=feat1, stage=3)
                # feat2 = base.model(x2=feat2, stage=3)
                # #feat1, feat2 = base.model2(feat1, feat2, stage=3)
                # feat = torch.cat((feat1, feat2),dim=2)

            gall = feat

            gall_feat[ptr:ptr + batch_num, :, :] = gall.detach().cpu().numpy()
            ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        distmat = gait_dist(query_feat, gall_feat).cpu().numpy()
        cmc2, mAP2 = evaluate_vcm(distmat, q_pids, g_pids, q_camids,g_camids)

        all_cmc  = cmc2
        all_mAP  = mAP2

        # if t2v:
        #     str = print_metrics(
        #         all_cmc, all_mAP,
        #         prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
        #     )
        # else:
        #     str = print_metrics(
        #         all_cmc, all_mAP,
        #         prefix='{:<3}->{:<3}:  '.format('RGB', 'IR')
        #     )

        str = print_metrics(
            all_cmc, all_mAP,
            prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
        )
        
    return all_cmc, all_mAP, str



















def test_vcm_gait(base, loader, t2v=True):
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    q_pids = []
    q_camids = []

    # if config.vcm_test_mode == 't2v':
    #     t2v = True
    # else:
    #     t2v = False

    if t2v:
        query_loader = loader.query_loader
        query_feat = np.zeros((loader.samples.num_query_tracklets, 256,70))
    else:
        query_loader = loader.query_loader_1
        query_feat = np.zeros((loader.samples.num_query_tracklets_1, 256,70))

    with torch.no_grad():
        for batch_idx, (input, label,c_label) in enumerate(query_loader):
            batch_num = input.size(0)
            q_pids.extend(label)
            q_camids.extend(c_label)
            input = Variable(input.cuda())
            input = process_vedio(input)

            if t2v:
                feat1 = base.shape_model(x2=input, stage=1)
                feat2 = base.model(x2=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)

                # feat1, feat2 = base.model2(feat1, feat2, stage=2)

                feat1 = base.shape_model(x2=feat1, stage=3)
                feat2 = base.model(x2=feat2, stage=3)
                feat1, feat2 = base.model2(feat1, feat2, stage=3)
                feat = torch.cat((feat1, feat2),dim=2)

            else:
                feat1 = base.shape_model(x1=input, stage=1)
                feat2 = base.model(x1=input, stage=1)

                #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                
                # feat1, feat2 = base.model2(feat1, feat2, stage=2)

                feat1 = base.shape_model(x1=feat1, stage=3)
                feat2 = base.model(x1=feat2, stage=3)
                feat1, feat2 = base.model2(feat1, feat2, stage=3)
                feat = torch.cat((feat1, feat2),dim=2)

            query = feat

            query_feat[ptr:ptr + batch_num, :, :] = query.detach().cpu().numpy()
            ptr = ptr + batch_num
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print('Extracting Gallery Feature...')

    if loader.dataset == 'vcm':
        g_pids = []
        g_camids = []

        ptr = 0
        if t2v:
            gall_loader = loader.gallery_loader
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets, 256, 70))
        else:
            gall_loader = loader.gallery_loader_1
            gall_feat = np.zeros((loader.samples.num_gallery_tracklets_1, 256, 70))

        with torch.no_grad():
            for batch_idx, (input, label,c_label) in enumerate(gall_loader):
                batch_num = input.size(0)
                g_pids.extend(label)
                g_camids.extend(c_label)
                input = Variable(input.cuda())
                input = process_vedio(input)

                if t2v:
                    feat1 = base.shape_model(x1=input, stage=1)
                    feat2 = base.model(x1=input, stage=1)

                    #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                    embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x1=feat1, stage=2)
                    feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x1=feat2, stage=2)

                    #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                    feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                    feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                    
                    # feat1, feat2 = base.model2(feat1, feat2, stage=2)

                    feat1 = base.shape_model(x1=feat1, stage=3)
                    feat2 = base.model(x1=feat2, stage=3)
                    feat1, feat2 = base.model2(feat1, feat2, stage=3)
                    feat = torch.cat((feat1, feat2),dim=2)

                else:
                    feat1 = base.shape_model(x2=input, stage=1)
                    feat2 = base.model(x2=input, stage=1)

                    #feat1, feat2 = base.model2(feat1, feat2, stage=1)

                    embed_1, embed_2, embed_4, embed_8, embed_16 = base.shape_model(x2=feat1, stage=2)
                    feature_1 , feature_2, feature_4, feature_8, feature_16 = base.model(x2=feat2, stage=2)

                    #feat1, feat2 = base.model2(embed_1, embed_2, embed_4, embed_8, embed_16,      feature_1 , feature_2, feature_4, feature_8, feature_16, stage=2)
                    feat1 = torch.cat([embed_1, embed_2, embed_4, embed_8, embed_16], dim=2)
                    feat2 = torch.cat([feature_1, feature_2, feature_4, feature_8, feature_16], dim=2)
                    
                    # feat1, feat2 = base.model2(feat1, feat2, stage=2)
                    
                    feat1 = base.shape_model(x2=feat1, stage=3)
                    feat2 = base.model(x2=feat2, stage=3)
                    feat1, feat2 = base.model2(feat1, feat2, stage=3)
                    feat = torch.cat((feat1, feat2),dim=2)

                gall = feat

                gall_feat[ptr:ptr + batch_num, :, :] = gall.detach().cpu().numpy()
                ptr = ptr + batch_num

        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        distmat = gait_dist(query_feat, gall_feat).cpu().numpy()
        cmc2, mAP2 = evaluate_vcm(distmat, q_pids, g_pids, q_camids,g_camids)

        all_cmc  = cmc2
        all_mAP  = mAP2

        if t2v:
            str = print_metrics(
                all_cmc, all_mAP,
                prefix='{:<3}->{:<3}:  '.format('IR', 'RGB')
            )
        else:
            str = print_metrics(
                all_cmc, all_mAP,
                prefix='{:<3}->{:<3}:  '.format('RGB', 'IR')
            )
    return all_cmc, all_mAP, str



def gait_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()

    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
        dist += torch.sqrt(F.relu(_dist))
    return dist / num_bin





def attn(shape_feature_map,app_feature_map=None, stage=None):

    if stage == 2:
        b,c,h = shape_feature_map.size()
        shape_feature_map = shape_feature_map.view(b//6, c, 6, h)
        app_feature_map = app_feature_map.view(b//6, c, 6, h)

        shape = torch.zeros_like(shape_feature_map)
        app = torch.zeros_like(app_feature_map)
        for i in range(shape_feature_map.size(2)):
            s = shape_feature_map[:,:,:,i]
            a = app_feature_map[:,:,:,i]

            s,att = Cross_Non_local_s(s, a)
            a,att2 = Cross_Non_local_a(s, a)

            shape[:,:,:,i] = s
            app[:,:,:,i] = a

        shape_feature_map = shape
        app_feature_map = app
        return shape_feature_map,app_feature_map



def Cross_Non_local_a(s,a):

    batch_size = a.size(0)
    s = s.permute(0,2,1)
    f = torch.matmul(a, s)
    N = f.size(-1)
    f_div_C = f / N

    y = torch.matmul(f_div_C, a)

    z = y + a

    return z,f_div_C



def Cross_Non_local_s(s,a):
    
    batch_size = a.size(0)
    a = a.permute(0,2,1)
    f = torch.matmul(s, a)
    N = f.size(-1)
    f_div_C = f / N

    y = torch.matmul(f_div_C, s)

    z = y + s

    return z,f_div_C
