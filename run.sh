python main-stage1.py --num_classes=10 \
--path_cifar10=/home/Data/CIFAR10 \
--num_clients=100 \
--num_online_clients=8 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--lr_local_training=0.1 \
--non_iid_alpha=0.5 \
--imb_factor=0.02 \
--save_path=/home/Fed-temp/result


python main-stage2.py --path_cifar10=xxx/xxx/CIFAR10 \
--num_clients=100 \
--imb_factor=0.02 \
--load_path=/home/Fed-temp/result/xxxx.pth \
--save_path=/home/Fed-temp/result

