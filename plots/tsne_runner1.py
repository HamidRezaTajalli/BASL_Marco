from pathlib import Path
import argparse
import time
from tsne import tsne_plot, tsne_plot_per_client


parser = argparse.ArgumentParser(description='BASL_Marco')
parser.add_argument('--datadir', type=str, default='.',
                    help='parent path to load and save required data')
parser.add_argument('--expname', type=str, default=str(time.time_ns()),
                    help='experiment name which will be added at beginning of plots\' name so plots will be distinguishable from each other')
parser.add_argument('--dimension', type=str, default='2d',
                    help='2d or 3d for plotting tsne')

args = parser.parse_args()



def main():
    plt_mode = args.dimension
    num_of_clients = 10
    epoch_list = ['99', '89', '79', '69', '59', '49', '39', '29', '19', '9']

    for epoch_num in epoch_list:
        for mode in ['FW', 'BW']:
            smsh_address = Path(args.datadir).joinpath('10clients', '61', f'{epoch_num}', f'{mode}')
            lbl_address = Path(args.datadir).joinpath('10clients', '61', f'{epoch_num}')
            if not smsh_address.exists() or not lbl_address.exists():
                raise Exception('Path does not exist')
            tsne_plot(args.expname, smsh_address, num_of_clients=num_of_clients, plt_mode=plt_mode)
            # tsne_plot_per_client(args.expname, smsh_address=smsh_address, lbl_address=lbl_address,
            #                      num_of_clients=num_of_clients, plt_mode=plt_mode)


if __name__ == '__main__':
    main()