import argparse
import os
from utils.helper import str2bool


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='test',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--gpus', type=str, default='0', help='set CUDA_VISIBLE_DEVICES')
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cocotalk_final.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc',
                        help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--sg_data_dir', type=str, default='data/coco_img_sg',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--loader_num_workers', type=int, default=4,
                        help='num of processes to use for BlobFetcher')
    parser.add_argument('--bag_info_path', type=str, default='data/aligned_triplets_final.json',
                        help='num of processes to use for BlobFetcher')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                        help='num of processes to use for BlobFetcher')
    parser.add_argument('--embed_dim', type=int, default=300,
                        help='num of processes to use for BlobFetcher')
    parser.add_argument('--glove_dir', type=str, default='data/glove_dir',
                        help='num of processes to use for BlobFetcher')

    # load model and settings
    parser.add_argument('--resume_from', type=str, default=None,
                        help="continuing training from this experiment id")
    parser.add_argument('--resume_from_best', type=str2bool, default=False,
                        help='resume from best model, True: use best_model.pth, False: use model.pth')
    parser.add_argument('--load_best_score', type=int, default=1,
                        help='Do we load previous best score when resuming training.')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Do we load previous best score when resuming training.')

    # Model settings
    parser.add_argument('--obj_num_classes', type=int, default=305, help='number of obj classes')
    parser.add_argument('--rel_num_classes', type=int, default=1000, help='number of relationship classes')
    parser.add_argument('--in_channels', type=int, default=2048, help='number of channel obj feature')
    parser.add_argument('--hidden_dim', type=int, default=512, help='number of channel obj feature')
    parser.add_argument('--pooling_dim', type=int, default=4096, help='number of channel obj feature')
    parser.add_argument('--obj_layer', type=int, default=1, help='number of channel obj feature')
    parser.add_argument('--rel_layer', type=int, default=1, help='number of channel obj feature')

    # parser.add_argument('--num_layers', type=int, default=1,
    #                     help='number of layers in the RNN')
    # parser.add_argument('--input_encoding_size', type=int, default=1000,
    #                     help='the encoding size of each token in the vocabulary, and the image.')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--accumulate_number', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1,  # 5.,
                        help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                        help='')
    parser.add_argument('--reduce_on_plateau', action='store_true',
                        help='')

    # learning rate
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                        help='at what epoch to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=-1,
                        help='how often to save a model checkpoint (in iterations)? (-1 = every epoch)')
    parser.add_argument('--checkpoint_root', type=str, default='log',
                        help='root directory to store checkpointed models')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='directory to store current checkpoint, \
                         if not set, it will be assigned as (args.checkpoint_root, args.id) by default. ')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L/SPICE? requires coco-caption code from Github.')
    parser.add_argument('--log_loss_every', type=int, default=10,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

    args = parser.parse_args()

    if args.checkpoint_path == '':
        args.checkpoint_path = os.path.join(args.checkpoint_root, args.id)
    if not os.path.exists(args.checkpoint_root):
        os.mkdir(args.checkpoint_root)
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    if args.resume_from:
        path = os.path.join(args.checkpoint_root, args.resume_from)
        assert os.path.exists(path), "%s not exists" % args.resume_from

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # os.environ["CUDA_VISIBLE_DEVICES"] = 0

    print("[INFO] set CUDA_VISIBLE_DEVICES = %s" % args.gpus)

    # Check if args are valid
    # assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"

    return args
