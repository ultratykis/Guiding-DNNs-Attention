
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.mixture import GaussianMixture
from lib.data import SubsetSequentialSampler

from pytorch_grad_cam import GradCAM


class Uncertainty:
    def __init__(self, model, added_number, train_set, unlabeled_set, test_set, val_set, batch, subset_number, sample_type, num_classes):
        self.added_number = added_number
        self.train_set = train_set
        self.unlabeled_set = unlabeled_set
        self.batch = batch
        self.sample_type = sample_type
        self.num_classes = num_classes
        indices = list(range(len(train_set)))
        random.shuffle(indices)
        self.labeled_set_id = indices[:self.added_number]
        self.unlabeled_set_id = indices[self.added_number:]

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch,
                                       sampler=SubsetRandomSampler(self.labeled_set_id), pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=self.batch)
        self.val_loader = DataLoader(
            val_set, batch_size=self.batch, shuffle=True)

        self.model = model
        torch.backends.cudnn.benchmark = True
        self.subset_number = subset_number
        # initialize grad-cam class
        self.cam = GradCAM(model=self.model, target_layers=[
            self.model.layer4[-1]], use_cuda=True)
        self.cam.batch_size = self.batch
        # intialize gmm
        self.gmm = GaussianMixture(
            n_components=2, random_state=0, covariance_type='diag')

    def set_hyperparams(self, epoch, lr, milestones, momentum=0.9, wdecay=5e-4):
        self.epoch = epoch
        self.lr = lr
        self.milestones = milestones
        self.momentum = momentum
        self.wdecay = wdecay
        self.cycle = 0

    def learn(self, checkpoint_dir):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=self.lr, weight_decay=self.wdecay)
        self.scheduler = lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones)

        if self.cycle != 0:
            # Training and test
            self.train(checkpoint_dir)
            self.best_acc = self.test()

        #  Update the labeled dataset via loss prediction-based uncertainty measurement

        # Randomly sample 10000 unlabeled data points
        random.shuffle(self.unlabeled_set_id)
        subset_id = self.unlabeled_set_id[:self.subset_number]

        self.unlabeled_loader = DataLoader(self.unlabeled_set, batch_size=self.batch,
                                           sampler=SubsetSequentialSampler(
                                               subset_id),
                                           # more convenient if we maintain the order of subset
                                           pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = self.get_uncertainty()

        # Index in ascending order
        if self.sample_type == "entropy":
            arg = np.argsort(-uncertainty)
        else:
            arg = np.argsort(uncertainty)
        self.uncertainty_set_id = list(torch.tensor(
            subset_id)[arg][-self.added_number:].numpy())

        # Update the labeled dataset and the unlabeled dataset, respectively
        self.labeled_set_id += list(torch.tensor(subset_id)
                                    [arg][-self.added_number:].numpy())
        self.unlabeled_set_id = list(torch.tensor(subset_id)[
                                     arg][:-self.added_number].numpy()) + self.unlabeled_set_id[self.subset_number:]

        self.cycle += 1

    def update_train_loader(self):
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch,
                                       sampler=SubsetRandomSampler(
                                           self.labeled_set_id),
                                       pin_memory=True)

    def test(self):
        self.model.eval()

        total = 0
        correct = 0
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))

        with torch.no_grad():
            for (inputs, labels, _, _, _) in self.test_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores = self.model(inputs)
                batch_total, batch_correct, batch_class_total, batch_class_correct = self.get_accuracy(
                    scores, labels, self.num_classes)

                total += batch_total
                correct += batch_correct

                for i in range(self.num_classes):
                    class_total[i] += batch_class_total[i]
                    class_correct[i] += batch_class_correct[i]
        test_acc = []
        test_acc.append(correct / total)
        for i in range(self.num_classes):
            test_acc.append(class_correct[i] / class_total[i])

        return test_acc

    def get_accuracy(self, outputs, labels, num_classes):
        _, predicted = torch.max(outputs, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()

        batch_class_correct = list(0. for i in range(num_classes))
        batch_class_total = list(0. for i in range(num_classes))

        c = (predicted == labels).squeeze()
        if batch_total == 1:
            for i in range(outputs.size(0)):
                label = labels[i]
                batch_class_total[label] += 1
                batch_class_correct[label] += c.item()
        else:
            for i in range(outputs.size(0)):
                label = labels[i]
                batch_class_total[label] += 1
                batch_class_correct[label] += c[i].item()

        return batch_total, batch_correct, batch_class_total, batch_class_correct

    def train(self, checkpoint_dir):

        savepath_best = os.path.join(
            checkpoint_dir, 'al_best.pth'.format(self.cycle))
        savepath_final = os.path.join(
            checkpoint_dir, 'al_latest.pth'.format(self.cycle))

        print("Training...")
        for epoch in range(self.epoch):
            self.model.train()
            self.L2_loss = nn.MSELoss()
            for data in self.train_loader:
                inputs = data[0].cuda()
                labels = data[1].cuda()

                attention_gt = data[3]
                attention_peak = data[4]
                attention_loss = self.L2_loss(
                    attention_gt/224, attention_peak/224)

                click_nega_loss = data[2].cuda().to(torch.float32)

                loss_g = click_nega_loss + attention_loss

                self.optimizer.zero_grad()

                predication = self.model(inputs)
                loss_c = self.criterion(predication, labels)

                loss = loss_c + 3*loss_g
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, savepath_final)

    def get_uncertainty(self):
        self.model.eval()
        uncertainty = torch.tensor([]).cuda()
        softmax = nn.Softmax(dim=1)
        labeled_preds = None

        if self.sample_type == "attention":
            # inputs_l, labels_t, _, _, _ = next(
            #     iter(self.train_loader))
            # attention_map_l = self.cam(
            #     input_tensor=inputs_l.cuda(), targets=None)

            for (inputs, labels, _, _, _) in self.unlabeled_loader:
                attention_map_unl = self.cam(
                    input_tensor=inputs.cuda(), targets=None)
                uncertainty = torch.cat(
                    (uncertainty, self.attention_distribution_based(attention_map_unl).cuda()), 0)
        else:
            with torch.no_grad():
                for (inputs, labels, _, _, _) in self.unlabeled_loader:

                    if self.sample_type == "entropy":
                        inputs = inputs.cuda()
                        preds = self.model(inputs)
                        preds = softmax(preds)
                        uncertainty = torch.cat(
                            (uncertainty, self.entropy_based(preds)), 0)
                        uncertainty = torch.abs(torch.sub(uncertainty, 0.5))

                    elif self.sample_type == "diversity":
                        labeled_preds = torch.tensor([]).cuda()
                        inputs_l, labels_t, _, _, _ = next(
                            iter(self.train_loader))
                        inputs_l = inputs_l.cuda()
                        preds_l = self.model(inputs_l[-(inputs_l.shape[0]):])
                        preds_l = softmax(preds_l)
                        labeled_preds = preds_l
                        inputs = inputs.cuda()
                        preds = self.model(inputs)
                        preds = softmax(preds)
                        uncertainty = torch.cat(
                            (uncertainty, self.diversity_based(labeled_preds, preds)), 0)
                    else:  # random sampling
                        inputs = inputs.cuda()
                        uncertainty = torch.cat(
                            (uncertainty, self.random_based(inputs)), 0)

        return uncertainty.cpu()

    def random_based(self, prob_dist):
        random_indices = torch.randperm(prob_dist.shape[0]).cuda()

        return random_indices

    def entropy_based(self, prob_dist):
        log_probs = prob_dist * torch.log2(prob_dist)
        raw_entropy = 0 - torch.sum(log_probs, 1)
        normalized_entropy = raw_entropy / torch.log2(torch.from_numpy(
            np.full(prob_dist.shape[0], prob_dist.shape[1]).astype(np.float32)).clone()).cuda()

        return normalized_entropy

    def diversity_based(self, labeled, unlabeled):
        '''use coreset'''
        pdist = nn.PairwiseDistance(p=2)
        dists = pdist(labeled, unlabeled)

        return dists

    def attention_distribution_based(self, unlabeled):
        data = np.reshape(unlabeled, (unlabeled.shape[0], -1))
        self.gmm.fit(data)
        scores = self.gmm.score_samples(data)
        # normalize the scores
        scores = np.abs(scores / np.max(scores))

        return torch.from_numpy(scores)
