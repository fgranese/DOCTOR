from test_wrapper import test_tbb_pbb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', '-d_name', help='Name of the dataset', default='cifar10')
parser.add_argument('--scenario', '-sc', help='Once between TBB and PBB', default='tbb')
parser.add_argument('--ood', '-ood', help='True for testing with ood samples False otherwise', type=bool, default=False)


def main():
    args = parser.parse_args()
    test_tbb_pbb(args.dataset_name, args.scenario, args.ood)

if __name__ == "__main__":
    main()