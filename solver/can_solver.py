import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from discrepancy.class_domain_da import class_da_regulizer_soft
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy

class CANSolver(BaseSolver):
    def __init__(self, net, dpn, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(CANSolver, self).__init__(net, dataloader, \
                      bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert('categorical' in self.train_data)

        num_layers = len(self.net.module.FC) + 1
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                  num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES, 
                  intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS, 
                                        self.opt.CLUSTERING.FEAT_KEY, 
                                        self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
		len(self.history['ts_center_dist']) < 1 or \
		len(self.history['target_labels']) < 2:
           return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1], 
			target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True: 
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                #self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers 
                center_change = self.clustering.center_change 
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
	            	self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
	            	self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                              self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'], 
                                                self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break
                
                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)
            self.loop += 1

        print('Training Done!')
        
    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net, 
		source_dataloader, self.opt.DATASET.NUM_CLASSES, 
                self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(
		target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
		chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
			samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                      for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]
        
        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert(self.selected_classes == 
               [labels[0].item() for labels in  samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums, source_sample_labels
            
    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def get_one_hot_encoding(self, labels, num_classes):
        y = torch.eye(num_classes).cuda()
        return y[labels]

    def get_proper_onehot(self, nums):
        num_classes = len(nums)
        start = end = 0
        labels_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            labels_list.extend([c]*nums[c])
        return self.get_one_hot_encoding(labels_list, num_classes)

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
                     iter(self.train_data[self.source_name]['loader'])
        self.train_data['categorical']['iterator'] = \
                     iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            self.dpn.train()
            self.net.zero_grad()
            self.dpn.zero_grad()

            loss = 0
            ce_loss_iter = 0
            cdd_loss_iter = 0

            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name) 
            source_data, source_gt = source_sample['Img'],\
                          source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_preds = self.net(source_data)['logits']

            # compute the cross-entropy loss
            ce_loss = self.CELoss(source_preds, source_gt)
            ce_loss.backward()

            ce_loss_iter += ce_loss
            loss += ce_loss
         
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                       target_samples_cls, target_nums_cls, source_sample_labels = self.CAS()

                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples) 
                            for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.net(source_cls_concat)
                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.net(target_cls_concat)

                # prepare the features
                feats_toalign_S = self.prepare_feats(feats_source)
                feats_toalign_T = self.prepare_feats(feats_target) 

                domain_logits = self.dpn(feats_source["feats"])
                domain_logits = domain_logits.reshape(domain_logits.shape[0],self.opt.DATASET.NUM_CLASSES,-1)
                domain_prob_s = torch.zeros(domain_logits.shape,dtype=torch.float32).cuda()

                kl_loss = 0
                entropy_loss = 0
                num_active_classes = 0
                for i in range(self.opt.DATASET.NUM_CLASSES):
                    indexes = source_sample_labels == i
                    if indexes.sum()==0:
                        continue
                    entropy_loss_cl, domain_prob_s_cl = self.entropy_loss(domain_logits[indexes,i])
                    kl_loss += -self.get_domain_entropy(domain_prob_s_cl)
                    entropy_loss += entropy_loss_cl
                    domain_prob_s[indexes,i] = domain_prob_s_cl
                    num_active_classes+=1

                entropy_loss = entropy_loss * 0.01 / num_active_classes
                kl_loss = kl_loss * 0.01 / num_active_classes

                cdd_loss = class_da_regulizer_soft(False, feats_toalign_S, feats_toalign_T, 5, self.get_proper_onehot(source_nums_cls), self.get_proper_onehot(target_nums_cls), domain_prob_s, None)

                cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T, 
                               source_nums_cls, target_nums_cls)[self.discrepancy_key]

                total_loss = cdd_loss*self.opt.CDD.LOSS_WEIGHT + entropy_loss + kl_loss
                total_loss.backward()

                cdd_loss_iter += total_loss
                loss += total_loss

            # update the network
            self.optimizer.step()

            if self.opt.TRAIN.LOGGING and (update_iters+1) % \
                      (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_preds, source_gt)
                cur_loss = {'ce_loss': ce_loss_iter, 'cdd_loss': cdd_loss_iter,
			'total_loss': loss}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    print('Test at (loop %d, iters: %d) with %s: %.4f.' % (self.loop, 
                              self.iters, self.opt.EVAL_METRIC, accu))

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and \
		(update_iters+1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    def entropy_loss(self, output):
        criterion = HLoss().cuda()
        return criterion(output)

    def get_domain_entropy(self, domain_probs):
        bs, num_domains = domain_probs.size()
        domain_prob_sum = domain_probs.sum(0)/bs
        mask = domain_prob_sum.ge(0.000001)
        domain_prob_sum = domain_prob_sum*mask + (1-mask.int())*1e-5
        return -(domain_prob_sum*(domain_prob_sum.log())).mean()

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        domain_prob = F.softmax(x, dim=1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b, domain_prob + 1e-5