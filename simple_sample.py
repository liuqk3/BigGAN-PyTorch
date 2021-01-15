import torch
import torchvision
import utils
import random
import argparse
import os
import json

def parse_args():
    usage = 'Parser for sample images'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='I128',
        help='which dataset the model be trained?')
    parser.add_argument(
        '--weights_path', type=str, default='/home/t-qiankunliu/Program/BigGAN-PyTorch/weights/BigGAN_I128_hdf5_seed0_Gch96_Dch96_bs256_nDa8_nGa8_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/G_ema.pth',
        help='the path of model weight')
    parser.add_argument(
        '--not_strict', action='store_true', default=False,
        help='do not load the state dict strictly')
    parser.add_argument(
        '--class_info', type=str, default='data/cache/I128_imgs_class_info.json',
        help='the class info used to train the model'
    )

    # args for sample
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size used for test sample.')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Parallelize G?')  
    parser.add_argument(
        '--num_classes', type=int, default=8,
        help='number of random classes, if -1, sample all classes')
    parser.add_argument(
        '--num_per_classes', type=int, default=8,
        help='number of samples for each class')  
    parser.add_argument(
        '--sample_type', type=str, default='sample_folder',
        choices=['sample_folder', 'sample_sheet'],
        help='which type of sample to perform')
    parser.add_argument(
        '--samples_dir', type=str, default='samples/debug')

    args = parser.parse_args()

    if not os.path.exists(args.class_info):
        raise ValueError('No class info found, please run python calculate_inception_moments.py --dataset I128 to generate one')
    args.class_info = json.load(open(args.class_info, 'r'))

    return args


def get_config(args):
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())

    resolution = utils.imsize_dict[args.dataset]
    attn_dict = {128: '64', 256: '128', 512: '64'}
    dim_z_dict = {128: 120, 256: 140, 512: 128}

    # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
    config["resolution"] = resolution
    config["n_classes"] = utils.nclass_dict[args.dataset]
    config["G_activation"] = utils.activation_dict["inplace_relu"]
    config["D_activation"] = utils.activation_dict["inplace_relu"]
    config["G_attn"] = attn_dict[resolution]
    config["D_attn"] = "64"
    config["G_ch"] = 96
    config["D_ch"] = 96
    config["hier"] = True
    config["dim_z"] = dim_z_dict[resolution]
    config["shared_dim"] = 128
    config["G_shared"] = True
    config = utils.update_config_roots(config)
    config["skip_init"] = True
    config["no_optim"] = True
    config["device"] = "cuda"
    return config

def get_generater(args, config):
    # Import the model.
    model = __import__(config["model"])
    G = model.Generator(**config).to(config["device"])
    utils.count_parameters(G)

    utils.load_state_dict(G, torch.load(args.weights_path), strict=not args.strict)
    G.eval()
    return G

def sample_folder(args, config, G):
    (z_, y_) = utils.prepare_z_y(
        args.batch_size,
        G.dim_z,
        config["n_classes"],
        device=config["device"],
        fp16=config["G_fp16"],
        z_var=config["z_var"],
        num_categorie_to_sample=args.num_classes,
        per_category_to_sample=args.num_per_classes
    )

    out_path = os.path.join(args.samples_dir, 'sample_folder')
    utils.make_sure_dir(out_path)
    with torch.no_grad():
        z_.sample_()
        y_.sample_()
        if y_.next:
            image_tensors = G(z_, G.shared(y_)) # batch_size, 3, h, w
            a = 1
            # torchvision.utils.save_image(
            #     image_tensors,
            #     out_path,
            #     nrow=int(G_batch_size ** 0.5),
            #     normalize=True,
            # )


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)

    # Seed RNG.
    utils.seed_rng(config["seed"])
    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True

    G = get_generater(args, config)

    if args.sample_type == 'sample_folder':
        sample_folder(args, config, G)