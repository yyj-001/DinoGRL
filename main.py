
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ast
import torch
import random
import argparse
import numpy as np
from torch.cuda import amp
from data_loader.loader import Loader
from core import Base, train, test,test_vcm,test_bupt_gait,test_vcm_gait
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")

# import setproctitle

best_map_v2t = 0
best_rank1_v2t = 0
best_map_t2v = 0
best_rank1_t2v = 0

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_best_model(cmc, mAP, cmc2, map2,   best_map, best_rank1, best_map_v2t, best_rank1_v2t):
    int_cmc1 = int(cmc[0])
    int_map = int(mAP)
    int_cmc2 = int(cmc2[0])
    int_map2 = int(map2)

    delta_cmc1_int = int_cmc1 - int(best_rank1)
    delta_map1_int = int_map - int(best_map)
    delta_cmc2_int = int_cmc2 - int(best_rank1_v2t)
    delta_map2_int = int_map2 - int(best_map_v2t)

    if  (delta_cmc1_int > 0   and   delta_cmc2_int >= 0   and   delta_map1_int >= 0  and  delta_map2_int >= 0) or \
        (delta_cmc2_int > 0   and   delta_cmc1_int >= 0   and   delta_map1_int >= 0  and  delta_map2_int >= 0) or \
        (delta_map1_int > 0   and   delta_map2_int >= 0   and   delta_cmc1_int >= 0  and  delta_cmc2_int >= 0) or \
        (delta_map2_int > 0   and   delta_map1_int >= 0   and   delta_cmc1_int >= 0  and  delta_cmc2_int >= 0) or \
        cmc[0] >= best_rank1  and cmc2[0]>=best_rank1_v2t and   mAP >= best_map      and  map2 >= best_map_v2t:
        is_best = True
    else:
        is_best = False

    return is_best



def main(config):
    global best_map_v2t
    global best_rank1_v2t
    global best_map_t2v
    global best_rank1_t2v

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    if config.mode == 'train':
        if config.resume_train_epoch >= 0:
            model.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:
            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pth', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                    indexes[-1]))

        scaler = amp.GradScaler()
        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            for lr_schedulers in model.lr_schedulers:
                lr_schedulers.step(current_epoch)

            if current_epoch < config.total_train_epoch:
                _, result = train(model, loaders, config,scaler)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))

            if current_epoch + 1 >= 0 and (current_epoch + 1) % config.eval_epoch == 0:
                if config.dataset=='bupt':
                    cmc, mAP, eval_str = test_bupt_gait(model, loaders, config)
                    cmc_v2t = cmc[0]
                    map_v2t = mAP[0]
                    cmc_t2v = cmc[1]
                    map_t2v = mAP[1]

                    is_best_rank = save_best_model(cmc_t2v, map_t2v, cmc_v2t, map_v2t,   best_map_t2v, best_rank1_t2v, best_map_v2t, best_rank1_v2t)
                    if is_best_rank:
                        best_rank1_v2t = cmc_v2t[0]
                        best_map_v2t = map_v2t
                        best_rank1_t2v = cmc_t2v[0]
                        best_map_t2v = map_t2v
                    
                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))

                elif config.dataset=='vcm':
                    cmc, mAP, eval_str_t2v = test_vcm_gait(model, loaders, t2v=True)
                    cmc2, map2, eval_str_v2t = test_vcm_gait(model, loaders, t2v=False)

                    is_best_rank = save_best_model(cmc, mAP, cmc2, map2,   best_map_t2v, best_rank1_t2v, best_map_v2t, best_rank1_v2t)
                    if is_best_rank:
                        best_rank1_t2v = cmc[0]
                        best_map_t2v = mAP
                        best_rank1_v2t = cmc2[0]
                        best_map_v2t = map2

                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))

    elif config.mode == 'test':
        model.resume_model(config.resume_test_model)
        cmc, mAP, eval_str_t2v = test_vcm(model, loaders, t2v=True)
        _, _, eval_str_v2t = test_vcm(model, loaders, t2v=False)
        # cmc, mAP, mINP = test(model, loaders, config)
        # logger('Time: {}; Test on Dataset: {}, \nmINP: {} \nmAP: {} \n Rank: {}'.format(time_now(),config.dataset,mINP, mAP, cmc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--module', type=str, default='B_shape', help='B')
    parser.add_argument('--dataset', default='vcm', help='dataset name: vcm , bupt]')
    parser.add_argument('--vcm_data_path', type=str, default='/dataset/Inversing_dataset/TensoIR-VCM/')
    parser.add_argument('--bupt_data_path', type=str, default='/data1/ls/data/BUPTCampus/')
    parser.add_argument('--regdb_data_path', type=str, default='/opt/data/private/data/RegDB/')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')

    parser.add_argument('--img_w', default=128, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
    parser.add_argument('--seq_lenth', type=int, default=6)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=500)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--in_dim', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.00035)
    parser.add_argument('--c_learning_rate', type=float, default=0.0007)
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--lower', type=float, default=0.02)
    parser.add_argument('--upper', type=float, default=0.4)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--output_path', type=str, default='/data/yyj/MEAG_2/output/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=100, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=False)
    parser.add_argument('--total_train_epoch', type=int, default=300)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--vcm_test_mode', default='t2v', help='dataset name: regdb or sysu,vcm]')
    parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')

    ###bupt
    parser.add_argument('--train_frame_sample', type=str, default='random')
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--random_flip', action='store_false', default=True)
    parser.add_argument('--fake', action='store_true', default=False)
    parser.add_argument('--test_frame_sample', type=str, default='uniform')
    parser.add_argument('--test_sampler', type=str,
                             default='ConsistentModalitySampler', help='None for no shuffle')
    parser.add_argument('--test_bs', type=int, default=64)  # Please don't change it.
    parser.add_argument('--train_sampler', type=str,
                             default='RandomIdentitySampler', help='None for shuffle')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--train_sampler_nc', type=int, default=2)
    parser.add_argument('--train_sampler_nt', type=int, default=1)
    parser.add_argument('--max_rank', type=int, default=20)
    #loss
    parser.add_argument('--weight_a', type=float, default=0.0)
    #model
    parser.add_argument('--dinov2_model', type=str, default='/data/yyj/OpenGait-master/opengait_model/dinov2_vits14_pretrain.pth')
    parser.add_argument('--mask_model', type=str, default='/data/yyj/OpenGait-master/opengait_model/MaskBranch_vits14.pt')
    
    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)







# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cuda', type=str, default='cuda')
#     parser.add_argument('--mode', type=str, default='train', help='train, test')
#     parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
#     parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
#     parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
#     parser.add_argument('--module', type=str, default='B_shape', help='B')
#     parser.add_argument('--dataset', default='bupt', help='dataset name: vcm , bupt]')
#     parser.add_argument('--vcm_data_path', type=str, default='./MITML-main/MITML/data/VCM-HITSZ/')
#     parser.add_argument('--bupt_data_path', type=str, default='/dataset/Inversing_dataset/')
#     parser.add_argument('--regdb_data_path', type=str, default='/opt/data/private/data/RegDB/')
#     parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
#     parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
#     parser.add_argument('--num_pos', default=4, type=int,
#                         help='num of pos per identity in each modality')

#     parser.add_argument('--img_w', default=128, type=int, metavar='imgw', help='img width')
#     parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
#     parser.add_argument('--seq_lenth', type=int, default=6)
#     parser.add_argument('--test_batch', type=int, default=32)
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--pid_num', type=int, default=1074)
#     parser.add_argument('--steps', type=int, default=200)
#     parser.add_argument('--in_dim', type=int, default=2048)
#     parser.add_argument('--learning_rate', type=float, default=0.00035)
#     parser.add_argument('--c_learning_rate', type=float, default=0.0007)
#     parser.add_argument('--num_workers', default=8, type=int,
#                         help='num of pos per identity in each modality')
#     parser.add_argument('--lower', type=float, default=0.02)
#     parser.add_argument('--upper', type=float, default=0.4)
#     parser.add_argument('--ratio', type=float, default=0.3)
#     parser.add_argument('--weight_decay', type=float, default=0.0005)
#     parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
#                         help='milestones for the learning rate decay')
#     parser.add_argument('--output_path', type=str, default='/data/yyj/output_bupt/',
#                         help='path to save related informations')
#     parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
#     parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
#     parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=False)
#     parser.add_argument('--total_train_epoch', type=int, default=200)
#     parser.add_argument('--eval_epoch', type=int, default=2)
#     parser.add_argument('--vcm_test_mode', default='t2v', help='dataset name: regdb or sysu,vcm]')
#     parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')

#     ###bupt
#     parser.add_argument('--train_frame_sample', type=str, default='random')
#     parser.add_argument('--sequence_length', type=int, default=6)
#     parser.add_argument('--random_flip', action='store_false', default=True)
#     parser.add_argument('--fake', action='store_true', default=False)
#     parser.add_argument('--test_frame_sample', type=str, default='uniform')
#     parser.add_argument('--test_sampler', type=str,
#                              default='ConsistentModalitySampler', help='None for no shuffle')
#     parser.add_argument('--test_bs', type=int, default=64)
#     parser.add_argument('--train_sampler', type=str,
#                              default='RandomIdentitySampler', help='None for shuffle')
#     parser.add_argument('--train_bs', type=int, default=16)
#     parser.add_argument('--train_sampler_nc', type=int, default=2)
#     parser.add_argument('--train_sampler_nt', type=int, default=1)
#     parser.add_argument('--max_rank', type=int, default=20)
#     #loss
#     parser.add_argument('--weight_a', type=float, default=0.0)
#     #model
#     parser.add_argument('--dinov2_model', type=str, default='/data/yyj/OpenGait-master/opengait_model/dinov2_vits14_pretrain.pth')
#     parser.add_argument('--mask_model', type=str, default='/data/yyj/OpenGait-master/opengait_model/MaskBranch_vits14.pt')
    
#     config = parser.parse_args()
#     seed_torch(config.seed)
#     main(config)
