from config import get_arguments
import os
from trainer import *
import argparse

parser = get_arguments()

parser.add_argument('--input_file',required= True,help = 'path to inputting msi data')
parser.add_argument('--input_shape',type = int, nargs = '+', help='input file shape',)
parser.add_argument('--mode',
                    help = 'spat-contig mode for Spatially contiguous ROI, spat-spor for Spatially sporadic ROI',
                    default= 'spat-contig')
parser.add_argument('--output_file', default='output',help='output file name')

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()

    data_image = np.loadtxt(args.input_file)

    Dimension_Reduction(data_image, args)

    im_Average2target = Feature_Clustering(data_image, args)

    np.savetxt(args.output_file + '.txt', im_Average2target)
