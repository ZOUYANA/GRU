
import argparse

parser = argparse.ArgumentParser()

# data path
parser.add_argument('--data_file', default='ETTh1.csv', help='ETTh1.csv')
parser.add_argument('--save_path', default='./saved_model/')
parser.add_argument('--results_path', default='./results/')
# preprocess data
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--do_normalization', default=True)

# devices
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--use_parallel', default=False)

# modeling
parser.add_argument('--enc_in', default=7) 
parser.add_argument('--d_model', default=1024)
parser.add_argument('--n_layers', default=2)

# training
parser.add_argument('--learning_rate', default=0.0001)
parser.add_argument('--training_epoch', default=10000)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--early_stop', default=5)

# parse arguments
args = parser.parse_args()
