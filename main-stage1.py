from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from long_tailed_cifar10 import train_long_tail
from dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
from sample_dirichlet import clients_indices

import numpy as np
from torch import max, eq, no_grad
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
import time
import os


from sklearn.metrics import classification_report

class Global(object):
    def __init__(self,
                num_classes: int,
                device: str,
                args,
                num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature

        self.feature_syn = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                        requires_grad=True, device=args.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                        requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.optimizer_feature = SGD([self.feature_syn, ], lr=args.lr_feature)  # optimizer_img for synthetic data
        
        self.criterion = CrossEntropyLoss().to(args.device)
        # global model
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        self.feature_net = nn.Linear(256, 10).to(args.device)


    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def global_eval_acc(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        labels_true_test, labels_pred_test = [], []
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
                labels_true_test.extend(labels.data.cpu().numpy().tolist())
                labels_pred_test.extend(predicts.data.cpu().numpy().tolist())
            # accuracy = num_corrects / len(data_test)
            CLASS_NAME = [str(s_label)+'/'+str(s_label) for s_label in range(10)] 
            report_test = classification_report(labels_true_test, labels_pred_test, 
                                                target_names=CLASS_NAME, 
                                                digits=4, output_dict=True)
        return float(report_test["accuracy"]), float(report_test["9/9"]["recall"])   # TODO 9/9

    def download_params(self):
        return self.syn_model.state_dict()


class Local(object):
    def __init__(self, data_client, class_list: int):
        args = args_parser()
        self.device = args.device

        self.data_client = data_client
        self.class_compose = class_list

        self.criterion = CrossEntropyLoss().to(args.device)

        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(
            args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def local_train(self, args, global_params):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                    batch_size=args.batch_size_local_training,
                                    shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                _, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()


def Fedavg():
    args = args_parser()
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature}\n '
        'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_net=args.lr_net,
            lr_feature=args.lr_feature,
            num_of_feature=args.num_of_feature,
            match_epoch=args.match_epoch,
            crt_epoch=args.crt_epoch))
    random_state = np.random.RandomState(args.seed)
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                        args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                            args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                                args.num_classes)
    
    global_model = Global(num_classes=args.num_classes,
                            device=args.device,
                            args=args,
                            num_of_feature=args.num_of_feature)   # 100 特征重训练时每个class有100个样本
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)

    re_trained_acc_all = []
    re_trained_acc_few = []

    temp_model = nn.Linear(256, 10).to(args.device)
    syn_params = temp_model.state_dict()
    
    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()        

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        
        list_dicts_local_params = []   
        list_nums_local_data = []    
        
        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client,
                                class_list=original_dict_per_client[client])

            local_params = local_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))

        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)

        one_re_train_acc_all, one_re_train_acc_few = global_model.global_eval_acc(global_params, data_global_test, args.batch_size_test)
        re_trained_acc_all.append(one_re_train_acc_all)
        re_trained_acc_few.append(one_re_train_acc_few)

        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))

        if r % 10 == 0:
            MODEL_SAVE_PATH = os.path.join(args.save_path, f"Stage1_{args.imb_factor}ratio_epoch{r}.pth")
            global_params = global_model.download_params()
            torch.save(global_params, MODEL_SAVE_PATH)
            print(re_trained_acc_all)


if __name__ == '__main__':
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    args = args_parser()
    Fedavg()


