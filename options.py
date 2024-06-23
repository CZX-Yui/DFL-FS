import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default="")
    parser.add_argument('--path_cifar100', type=str, default="")
    
    parser.add_argument('--num_classes', type=int, default=100)   
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--head_ratio', default=1, type=float, help='head factor')

    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--init_belta', type=float, default=0.97)
    args = parser.parse_args()
    return args
