import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='My Benchmark')
    
    parser.add_argument('--datatsetstype', default='ZedCamera', type=str,  help='DataType')
    parser.add_argument('--ctrtype', default='train', type=str,  help='DataControlblock work mode')
    parser.add_argument('--modeltypes', default='pointnet2_part_seg_msg', type=str,  help='modelType')
    parser.add_argument('--device', default='gpu', type=str,  help='device to run the code')
    parser.add_argument('--gpus', default='0', type=str,  help='gups which is used for runing the code')
    parser.add_argument('--set_datactr_func', default='', type=str,  help='func to set datactr(func_dir and fuc_name)')
    parser.add_argument('--set_modelctr_func', default='', type=str,  help='func to set modelctr(func_dir and fuc_name)')
    parser.add_argument('--log', default='logs', type=str,  help='print cache info dir')
    parser.add_argument('--cfg', default='', type=str,  help='path stored the config file')
    
    # parser.add_argument('--datadir', default='ZedCamera', type=str,  help='DataDir')
    # parser.add_argument('--datafilename', default='ZedCamera', type=str,  help='DataFileName')
    # parser.add_argument('--dataargset', default=None, type=str,  help='Set Data Args')
    
    # parser.add_argument('--modelname', default='pointnet2', type=str,  help='ModelName')
    # parser.add_argument('--modelfiledir', default='model/pointnet2/part_seg_msg/checkpoints', type=str,  help='ModelFileDir')
    # parser.add_argument('--modelfilename', default='best_model.pth', type=str,  help='ModelFileName')
    # parser.add_argument('--modelargset', default=None, type=str,  help='Set Model Args')
    
    # parser.add_argument('--visargset', default=None, type=str,  help='Set Visualizer Args')
        
    # parser.add_argument('--dataset', required=True,
    #                     choices=['cifar10', 'cifar100'], help='Dataset')
    
    # parser.add_argument('--name', default='0', type=str, help='name of run')
    
    # parser.add_argument('--net_both', default=None, type=str,
    #                     help='checkpoint path of both networks')
    
    # parser.add_argument('--step_size', default=0.1, type=float, help='')
    
    # parser.add_argument('--loss_type', default='CE', type=str,
    #                     choices=['CE', 'Focal', 'LDAM'],
    #                     help='Type of loss for imbalance')
    
    # parser.add_argument('--ratio', default=100, type=int, help='max/min')
    
    # parser.add_argument('--smote', '-s', action='store_true', help='oversampling')
    
    # parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')

    return parser.parse_args()
ARGS = parse_args()
if ARGS.cfg != '':
    with open(ARGS.cfg) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    for k, v in config['common'].items():
        setattr(ARGS, k, v)
pass