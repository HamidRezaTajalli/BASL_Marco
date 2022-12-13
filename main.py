import argparse
from pathlib import Path
from train_and_validation import sl, sl_simple


parser = argparse.ArgumentParser(description='BASL_Autoencoder')
parser.add_argument('--dataname', type=str, default='fmnist',
                    choices=['mnist', 'cifar10', 'fmnist'],
                    help='The dataset to use')
parser.add_argument('--model', type=str, default='resnet9', choices=[
                    'resnet9', 'resnet18', 'cnn6', 'lenet', 'stripnet'], help='The model to use')

parser.add_argument('--exp_num', type=float, default=2,
                    help='number of each experiment being repeated')
parser.add_argument('--pos', type=str, default="upper-mid",
                    choices=["upper-left", "upper-mid", "upper-right", "mid-left", "mid-mid", "mid-right", "lower-left",
                             "lower-mid", "lower-right"],
                    help='The position of the trigger')
parser.add_argument('--trigger_size', type=float, default=4,
                    help='The size of the trigger')
parser.add_argument('--trigger_label', type=int, default=0,
                    help='The label of the target/objective class. The class to be changed to.')
parser.add_argument('--batch_size', type=int,
                    default=128, help='Train batch size')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--datadir', type=str, default='./data',
                    help='path to save downloaded data')

parser.add_argument('--save_path', type=str, default='.',
                    help='path to save training results')
parser.add_argument('--cutlayer', type=int, default=1, help='cut layer')
parser.add_argument('--num_clients', type=int, default=1, help='number of innocent clients')
parser.add_argument('--fixed_alpha', action='store_true', help='alpha to be fixed or not during training')
parser.add_argument('--tb_inj', action='store_true', help='training to convergence before injecting the backdoor?')
args = parser.parse_args()


def main():
    exp_num = args.exp_num
    trig_size = args.trigger_size
    trig_pos = args.pos
    dataset = args.dataname
    trig_shape = 'square'
    trig_samples = 100
    bd_label = args.trigger_label
    arch_name = args.model
    bd_opacity = 1.0
    base_path = Path(args.save_path)
    tp_name = 'BASL_Autoencoder'
    cut_layer = args.cutlayer
    batch_size = args.batch_size
    num_clients = args.num_clients
    alpha_fixed = args.fixed_alpha
    tb_inj = args.tb_inj

    # sl_simple.sl_training_procedure(tp_name=tp_name, dataset=dataset, arch_name=arch_name, cut_layer=cut_layer,
    #                                 base_path=base_path, exp_num=exp_num, batch_size=batch_size, num_clients=num_clients)

    sl.sl_training_procedure(tp_name=tp_name, dataset=dataset, arch_name=arch_name, cut_layer=cut_layer, base_path=base_path, exp_num=exp_num, batch_size=batch_size,
                             alpha_fixed=alpha_fixed, num_clients=num_clients, bd_label=bd_label, tb_inj=tb_inj)




if __name__ == '__main__':
    main()
