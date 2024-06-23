import torch
import numpy as np
import random 
import copy
import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


GPU_ID = 0   
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else 'cpu'

SELECT_LABELS = list(range(10))     

seed = 7
np.random.seed(seed)    
random.seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--path_cifar10', type=str, default="")
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--load_path', type=str, default='result/')
parser.add_argument('--save_path', type=str, default='result/')
parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
args = parser.parse_args()

"""
1. 数据集加载   
"""
from torchvision import datasets
from torchvision.transforms import transforms

train_batch_size = 32  
test_batch_size = 64

transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data_local_train = datasets.CIFAR10(args.path_cifar10, train=True, download=True, 
                                        transform=transform_all)
data_global_test = datasets.CIFAR10(args.path_cifar10, train=False,
                                            transform=transform_all)
length_test = len(data_global_test)
test_loader = DataLoader(data_global_test, test_batch_size) 


from long_tailed_cifar10 import train_long_tail
from dataset import classify_label, show_clients_data_distribution
from sample_dirichlet import clients_indices
list_label2indices = classify_label(data_local_train, num_classes=10)  
_, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), num_classes=10,
                                                    imb_factor=args.imb_factor, imb_type="exp")
list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), num_classes=10,
                                        num_clients=args.num_clients, non_iid_alpha=0.5, seed=seed)
original_dict_per_client = show_clients_data_distribution(data_local_train, list_client2indices,
                                                            num_classes=10)


"""
2. 模型初始化与训练
"""
from train_frame import train_local
from Resnet8 import ResNet_cifar


print("training stage 2 F2C")  

print(f"loading model from {args.load_path}")


global_model = ResNet_cifar(resnet_size=8, scaling=4, save_activations=False,
                    group_norm_num_groups=None, freeze_bn=False, 
                    freeze_bn_affine=False, num_classes=10).to(DEVICE)
w_glob = torch.load(args.load_path)
global_model.load_state_dict(w_glob, strict=False)


F_label_all = []
F_mu_all = []
F_sigma_all = []
Sample_num = []
for idx in range(args.num_clients):
    local = train_local(data=data_local_train, idxs=list_client2indices[idx], device=DEVICE)
    f_label, f_mu, f_sigma, num_i = local.get_mu_sigma_CIFAR(net=copy.deepcopy(global_model).to(DEVICE), class_num=10)
    # 先求全局平均F
    F_label_all.extend(f_label)
    F_mu_all.extend(f_mu)
    F_sigma_all.extend(f_sigma)
    Sample_num.extend(num_i)


Global_mu = [[0] * 256] * 10
Global_sigma = [0] * 10
Global_num = [0] * 10
for c_i in set(F_label_all):
    g_mu = []
    g_sigma_1 = []
    g_sigma_2 = []
    count = []
    for idx in range(len(F_label_all)):
        if c_i == F_label_all[idx]:
            count.append(Sample_num[idx])
            g_mu.append(Sample_num[idx] * F_mu_all[idx]) 
            g_sigma_1.append(((Sample_num[idx]-1) * F_sigma_all[idx]))
            g_sigma_2.append(Sample_num[idx] * np.dot(F_mu_all[idx], F_mu_all[idx]))
    Global_num[c_i] = sum(count)
    Global_mu[c_i] = sum(g_mu) / Global_num[c_i]
    Global_sigma[c_i] = sum(g_sigma_1) / (Global_num[c_i]-1) + sum(g_sigma_2) / (Global_num[c_i]-1)

linalg_mu = []
for c_i in range(10):
    linalg_mu.append(np.linalg.norm(Global_mu[c_i], 2))
print("linalg2: ", linalg_mu)

Global_sigma_new = [0] * 10
weight_sim = [1, 0.89, 0.78, 0.67, 0.56, 0.45, 0.34, 0.23, 0.12, 0.01]  # IF 0.01
for c_i in range(10):
    temp = Global_sigma[c_i] - (Global_num[c_i]/(Global_num[c_i]-1)) * np.dot(Global_mu[c_i], Global_mu[c_i])
    Global_sigma_new[c_i] = temp   

Gen_feature = []
Gen_label = []
Gen_num = 1000    
for c_i in range(10):   
    Gussain_f = np.random.multivariate_normal(Global_mu[c_i], Global_sigma_new[c_i], Gen_num)
    Gen_feature.extend(Gussain_f)
    Gen_label.extend([c_i] * Gen_num)

F_center_all = Gen_feature
F_label_all = Gen_label

class FeatureDataset(Dataset):
    def __init__(self, train_data, train_label):
        self.features = train_data
        self.labels = train_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

class Bal_FeatureDataset(Dataset):
    def __init__(self, train_data, train_label):
        self.features = train_data
        self.labels = train_label

        final_idx = []
        check_label = [0] * 10       
        label_per_class = [[] for _ in range(10)]
        for i in range(len(self.labels)):
            check_label[self.labels[i]] += 1
            label_per_class[self.labels[i]].append(i)
        sample_per_class = 100  
        for c_i in range(10):
            if len(label_per_class[c_i]) > 0:
                _label_per_class = np.random.choice(label_per_class[c_i], sample_per_class)
                final_idx.extend(_label_per_class)
        self.idxs = final_idx

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feature = self.features[self.idxs[item]]
        label = self.labels[self.idxs[item]]
        return feature, label

train_dataloader_F = DataLoader(FeatureDataset(F_center_all, F_label_all), batch_size=32, shuffle=True)

from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(256, 10)
        
    def forward(self, x):
        logits = self.linear(x)
        return logits


model_F = NeuralNetwork().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_F.parameters(), lr=0.01)   

from test_frame import AccuarcyCompute
for iter in range(100):   
    batch_loss = []
    batch_ac = []
    batch_whole = []
    for batch_idx, (fea, labels) in enumerate(train_dataloader_F):
        fea, labels = fea.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.long)
        model_F.zero_grad()
        log_probs = model_F(fea)
        ac, whole = AccuarcyCompute(log_probs, labels)

        loss = loss_func(log_probs, labels)

        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
        batch_ac.append(ac)
        batch_whole.append(whole)
    if (iter+1) % 10 == 0:
        print(f"Epoch{iter}: acc {sum(batch_ac) / sum(batch_whole)}  loss {sum(batch_loss) / len(batch_loss)}")



model_F_param = model_F.state_dict()
new_C_w_tensor = model_F_param["linear.weight"]
new_C_b_tensor = model_F_param["linear.bias"]
w_glob['classifier.weight'].copy_(new_C_w_tensor)  
w_glob['classifier.bias'].copy_(new_C_b_tensor)  

# stage2模型路径
model_stage2 = f"Stage2_{args.imb_factor}ratio.pth" 
MODEL_SAVE_PATH = os.path.join(args.save_path, model_stage2)

print(f"saving model in {MODEL_SAVE_PATH}\n")
torch.save(w_glob, MODEL_SAVE_PATH)
print("Done!\n" + "-" * 80)



from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from sklearn.metrics import classification_report
CLASS_NAME = [str(i) for i in range(10)]

model_param = torch.load(MODEL_SAVE_PATH)
model = ResNet_cifar(resnet_size=8, scaling=4, save_activations=False,
                    group_norm_num_groups=None, freeze_bn=False, 
                    freeze_bn_affine=False, num_classes=10).to(DEVICE)
model.load_state_dict(model_param)

model.eval()

labels_true_test, labels_pred_test = [], []
with no_grad():
    num_corrects = 0
    for data_batch in test_loader:
        images, labels = data_batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, outputs = model(images)
        _, predicts = max(outputs, -1)
        num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
        labels_true_test.extend(labels.data.cpu().numpy().tolist())
        labels_pred_test.extend(predicts.data.cpu().numpy().tolist())
    accuracy = num_corrects / length_test

print(accuracy)
report_test = classification_report(labels_true_test, labels_pred_test, 
                                    target_names=CLASS_NAME, 
                                    digits=4) 
print("Test Set:")
print(report_test)



