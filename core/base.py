import os
import torch
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_right
from network import RGB_Model, IR_Model, Shared_Model, Classifier,Dinov2,Model,Shape_Model,gcn, attn
from tools import os_walk, TripletLoss_WRT,SupConLoss,CrossEntropyLoss,TripletLoss




class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num

        self.module = config.module

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.c_learning_rate = config.c_learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones
        self.steps = config.steps

        self.img_h = config.img_h
        self.img_w = config.img_w
        self.weight_a = config.weight_a

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()
        self._init_optimzer_and_lr_scheduler()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):
        self.model = Model(self.config)
        self.model = self.model.to(self.device)
        #self.model = nn.DataParallel(self.model).to(self.device)

        self.model2 = attn()
        self.model2 = self.model2.to(self.device)

        # self.rgb_model = RGB_Model()
        # self.rgb_model = nn.DataParallel(self.rgb_model).to(self.device)

        # self.ir_model = IR_Model()
        # self.ir_model = nn.DataParallel(self.ir_model).to(self.device)

        # self.shared_model = Shared_Model()
        # self.shared_model = nn.DataParallel(self.shared_model).to(self.device)

        # self.classifier = Classifier(self.pid_num)
        # self.classifier = nn.DataParallel(self.classifier).to(self.device)

        if 'shape' in self.config.module:
            self.shape_model = Shape_Model(self.config)
            self.shape_model = self.shape_model.to(self.device)
        #     self.shape_model = nn.DataParallel(self.shape_model).to(self.device)

        #     self.dinov2 = Dinov2(self.config)
        #     #self.dinov2 = self.dinov2.to(self.device)
        #     self.dinov2 = nn.DataParallel(self.dinov2).to(self.device)



    def _init_creiteron(self):
        self.pid_creiteron = nn.CrossEntropyLoss()
        self.tri_creiteron = TripletLoss_WRT()
        self.con_creiteron = SupConLoss(self.device)

        self.ce_loss = CrossEntropyLoss()
        self.tr_loss = TripletLoss()

    def _init_optimzer_and_lr_scheduler(self):
        optimizers = []
        lr_schedulers = []
        keys_dict = self.__dict__
        for key in keys_dict:
            if 'optimizer' in key:
                optimizers.append(keys_dict[key])
            if 'lr_scheduler' in key:
                lr_schedulers.append(keys_dict[key])
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers


    def _init_optimizer(self):
        model_params_group = [{'params': self.model.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        model2_params_group = [{'params': self.model2.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]

        # rgb_model_params_group = [{'params': self.rgb_model.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        # ir_model_params_group = [{'params': self.ir_model.parameters(), 'lr': self.learning_rate, 'weight_decay': self.weight_decay}]
        # shared_model_params_group = [{'params': self.shared_model.parameters(), 'lr': self.learning_rate,'weight_decay': self.weight_decay}]
        # classifier_params_group = [{'params': self.classifier.parameters(),'lr': self.c_learning_rate, 'weight_decay': self.weight_decay}]
        if 'shape' in self.config.module:
            shape_model_params_group = [{'params': self.shape_model.parameters(), 'lr': self.learning_rate,'weight_decay': self.weight_decay}]
        #     dinov2_params_group = [{'params': self.dinov2.parameters(), 'lr': self.learning_rate,'weight_decay': self.weight_decay}]


        self.model_optimizer = optim.Adam(model_params_group)
        self.model_lr_scheduler = WarmupMultiStepLR(self.model_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        self.model2_optimizer = optim.Adam(model2_params_group)
        self.model2_lr_scheduler = WarmupMultiStepLR(self.model2_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        # self.rgb_model_optimizer = optim.Adam(rgb_model_params_group)
        # self.rgb_model_lr_scheduler = WarmupMultiStepLR(self.rgb_model_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        # self.ir_model_optimizer = optim.Adam(ir_model_params_group)
        # self.ir_model_lr_scheduler = WarmupMultiStepLR(self.ir_model_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        # self.shared_model_optimizer = optim.Adam(shared_model_params_group)
        # self.shared_model_lr_scheduler = WarmupMultiStepLR(self.shared_model_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        # self.classifier_optimizer = optim.Adam(classifier_params_group)
        # self.classifier_lr_scheduler = WarmupMultiStepLR(self.classifier_optimizer, self.milestones,gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        if 'shape' in self.config.module:
            self.shape_model_optimizer = optim.Adam(shape_model_params_group)
            self.shape_model_lr_scheduler = WarmupMultiStepLR(self.shape_model_optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

        #     self.dinov2_optimizer = optim.Adam(dinov2_params_group)
        #     self.dinov2_lr_scheduler = WarmupMultiStepLR(self.dinov2_optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)


    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'best_model.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

            model2_file_path = os.path.join(self.save_model_path, 'best_model2.pth'.format(save_epoch))
            torch.save(self.model2.state_dict(), model2_file_path)

            # rgb_file_path = os.path.join(self.save_model_path, 'best_rgb_model.pth'.format(save_epoch))
            # torch.save(self.rgb_model.state_dict(), rgb_file_path)

            # ir_file_path = os.path.join(self.save_model_path, 'best_ir_model.pth'.format(save_epoch))
            # torch.save(self.ir_model.state_dict(), ir_file_path)

            # shared_file_path = os.path.join(self.save_model_path, 'best_shared_model.pth'.format(save_epoch))
            # torch.save(self.shared_model.state_dict(), shared_file_path)

            # class_file_path = os.path.join(self.save_model_path, 'best_class_model.pth'.format(save_epoch))
            # torch.save(self.classifier.state_dict(), class_file_path)

            if 'shape' in self.config.module:
                shape_file_path = os.path.join(self.save_model_path, 'best_shape.pth'.format(save_epoch))
                torch.save(self.shape_model.state_dict(), shape_file_path)

            #     dinov2_file_path = os.path.join(self.save_model_path, 'best_dinov2.pth'.format(save_epoch))
            #     torch.save(self.dinov2.state_dict(), dinov2_file_path)


    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pth' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pth', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, 'best_model.pth'.format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=True)

        model2_path = os.path.join(self.save_model_path, 'best_model2.pth'.format(resume_epoch))
        self.model2.load_state_dict(torch.load(model2_path), strict=True)

        if 'shape' in self.config.module:
            shape_path = os.path.join(self.save_model_path, 'best_shape.pth'.format(resume_epoch))
            self.shape_model.load_state_dict(torch.load(shape_path), strict=True)

        #     dinov2_path = os.path.join(self.save_model_path, 'best_dinov2.pth'.format(resume_epoch))
        #     self.dinov2.load_state_dict(torch.load(dinov2_path), strict=True)

        print('Successfully resume shared_model from {}'.format(model_path))


    def set_train(self):
        self.model = self.model.train()
        self.model2 = self.model2.train()

        # self.rgb_model = self.rgb_model.train()
        # self.ir_model = self.ir_model.train()
        # self.shared_model = self.shared_model.train()
        # self.classifier = self.classifier.train()
        if 'shape' in self.config.module:
            self.shape_model = self.shape_model.train()
        #     self.dinov2 = self.dinov2.train()


        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.model2 = self.model2.eval()
        
        # self.rgb_model = self.rgb_model.eval()
        # self.ir_model = self.ir_model.eval()
        # self.shared_model = self.shared_model.eval()
        # self.classifier = self.classifier.eval()
        if 'shape' in self.config.module:
            self.shape_model = self.shape_model.eval()
        #     self.dinov2 = self.dinov2.eval()


        self.training = False

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
