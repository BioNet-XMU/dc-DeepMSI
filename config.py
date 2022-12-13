import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='dc-DeepMSI for MSI interactive segmentation')

    parser.add_argument('--nChannel', default=100, type=int, help='number of channels in feature clustering')
    parser.add_argument('--lr_dr', default=0.01, type=float, help='learing rate of dimension reduction')
    parser.add_argument('--lr_fc', default=0.01, type=float, help='learing rate of feature clustering')
    parser.add_argument('--momentum_dr', default=0.9, type=float, help='momentum of dimension reduction')
    parser.add_argument('--momentum_fc', default=0.9, type=float, help='momentum of feature clustering')
    parser.add_argument('--epoch_dr', default=1000, type=int, help='learing rate of dimension reduction')
    parser.add_argument('--epoch_fc', default=200, type=int, help='learing rate of feature clustering')
    parser.add_argument('--meanNetStep', default=0.9, type=float, help='ensembling momentum')
    parser.add_argument('--use_gpu', default=True, type=bool, help='use GPU for accelerating')
    parser.add_argument('--use_umap',  default=True, type=bool, help='use Parameter-umap for joint optimization')
    parser.add_argument('--nConv', default=5, type=int, help='number of convolutional layers in feature clustering')
    parser.add_argument('--stepsize_sim', default=1, type=float, help='the weight of similarity loss')
    parser.add_argument('--stepsize_tv', default=1, type=float, help='the weight of total variation loss')
    parser.add_argument('--stepsize_sta', default=0.01, type=float, help='the weight of stability loss')

    return parser