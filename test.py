import torch
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from sklearn.metrics import classification_report
from Resnet8 import ResNet_cifar

CLASS_NAME = [str(i) for i in range(10)]
GPU_ID = 1   
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else 'cpu'

MODEL_LOAD_PATH = "" # load model to test


from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from get_dataset import CIFAR10_client_longtail, CIFAR10_check_LT_FL

train_batch_size = 32   # 训练集和测试集的加载batch设置
test_batch_size = 64

PATH_CIFAR10 = "/home/CZX/Data/CIFAR10"

transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
data_local_train = datasets.CIFAR10(PATH_CIFAR10, train=True, download=True, 
                                        transform=transform_all)
data_global_test = datasets.CIFAR10(PATH_CIFAR10, train=False,
                                            transform=transform_all)
length_test = len(data_global_test)
test_loader = DataLoader(data_global_test, test_batch_size) 


model_param = torch.load(MODEL_LOAD_PATH)
model = ResNet_cifar(resnet_size=8, scaling=4, save_activations=False,
                    group_norm_num_groups=None, freeze_bn=False, 
                    freeze_bn_affine=False, num_classes=10).to(DEVICE)
model.load_state_dict(model_param)

model.eval()

labels_true_test, labels_pred_test = [], []
with torch.no_grad():
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