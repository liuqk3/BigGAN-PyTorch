import torch
import torchvision
import utils
import random
import argparse
import os
import json
import random
import torch.nn as nn


def parse_args():
    usage = 'Parser for sample images'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='I128',
        help='which dataset the model be trained?')
    parser.add_argument(
        '--weights_path', type=str, default='weights/BigGAN_I128_hdf5_seed0_Gch96_Dch96_bs256_nDa8_nGa8_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/G_ema.pth',
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
        '--batch_size', type=int, default=5,
        help='Batch size used for test sample.')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Parallelize G?')
    parser.add_argument(
        '--sample_type', type=str, default='sample_interplation',
        choices=['sample_folder', 'sample_interplation'],
        help='which type of sample to perform')
    parser.add_argument(
        '--samples_dir', type=str, default='samples/debug')

    ## args for samples in a class-wise manner
    parser.add_argument(
        '--num_classes', type=int, default=3,
        help='number of random classes, if -1, sample all classes')
    parser.add_argument(
        '--num_per_classes', type=int, default=8,
        help='number of samples for each class')
    
    ## args for samples in a interplate manner
    parser.add_argument(
        '--num_pairs', type=int, default=20,
        help='the number of pairs to be generated. For each pair, we will generate num_midpoints interplated samples')
    parser.add_argument(
        '--num_midpoints', type=int, default=0,
        help='for each pair of combination, how many interplated sample are generated?, if less than 0, will generate 1 interplate sample with a random weight')
    parser.add_argument(
        '--fix_z', action='store_true', default=False,
        help='while interplating, fix the noize z, and change the class embedding y'
    )
    parser.add_argument(
        '--fix_y', action='store_true', default=False,
        help='while interplating, fix the class embedding y, and change the noize z'
    )

    args = parser.parse_args()

    if not os.path.exists(args.class_info):
        raise ValueError(
            'No class info found, please run python calculate_inception_moments.py --dataset I128 to generate one')
    if args.fix_z and args.fix_y:
        raise ValueError('While interplating, one of noise z and class embedding y should not be fixed!')
    args.class_info = json.load(open(args.class_info, 'r'))

    return args


def save_samples(args, sample, save_dir,
                 normalize=True, value_range=None, scale_each=False, meta=None):
    # this is modified from make_grid() in torchvision.utils.py
    from PIL import Image
    if normalize is True:
        sample = sample.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(ranvalue_rangege, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for s in sample:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(sample, value_range)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    sample = sample.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    if args.sample_type == 'sample_folder':
        y = meta['y'] # the class label
        for i in range(len(y)):
            cat_idx = int(y[i])
            one_sample = Image.fromarray(sample[i])
            if cat_idx not in args.sample_count:
                args.sample_count[cat_idx] = 0
            args.sample_count[cat_idx] += 1
            cat_name = args.class_info['idx_to_class'][str(cat_idx)]
            save_path = os.path.join(save_dir, cat_name, '{}.jpg'.format(args.sample_count[cat_idx]))
            utils.make_sure_dir(save_path, str_type='file')
            one_sample.save(save_path)
            print('saved in {}'.format(save_path))
    elif args.sample_type == 'sample_interplation':
        if args.fix_y:
            y = meta['y'] # the class label
            w = meta['w']
            for i in range(len(y)):
                cat_idx = int(y[i])
                one_sample = Image.fromarray(sample[i])
                cat_name = args.class_info['idx_to_class'][str(cat_idx)]
                if args.num_midpoints > 0:
                    weight = round(float(w[i]), 2)
                else:
                    weight = round(float(w), 2)
                args.sample_count += 1
                save_path = os.path.join(save_dir, '{}_{}_{}.jpg'.format(args.sample_count, cat_name, weight))
                utils.make_sure_dir(save_path, str_type='file')
                one_sample.save(save_path)
                print('saved in {}'.format(save_path))
        else:
            y1 = meta['y1'] # the class label
            y2 = meta['y2']
            w = round(float(meta['w']), 2)
            for i in range(len(y1)):
                cat_idx1 = int(y1[i])
                cat_idx2 = int(y2[i])
                one_sample = Image.fromarray(sample[i])
                cat_name1 = args.class_info['idx_to_class'][str(cat_idx1)]
                cat_name2 = args.class_info['idx_to_class'][str(cat_idx2)]
                if args.num_midpoints > 0:
                    weight = round(float(w[i]), 2)
                else:
                    weight = round(float(w), 2)
                args.sample_count += 1
                save_path = os.path.join(save_dir, '{}_{}_{}_{}.jpg'.format(args.sample_count, cat_name1, weight, cat_name2))
                utils.make_sure_dir(save_path, str_type='file')
                one_sample.save(save_path)
                print('saved in {}'.format(save_path))
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

    utils.load_state_dict(G, torch.load(args.weights_path),
                          strict=not args.not_strict)
    G.eval()
    return G


def sample_folder(args, config, G):
    args.sample_count = {}

    (z_, y_) = utils.prepare_z_y(
        args.batch_size,
        G.dim_z,
        config["n_classes"],
        device=config["device"],
        fp16=config["G_fp16"],
        z_var=config["z_var"],
        num_categories_to_sample=args.num_classes,
        per_category_to_sample=args.num_per_classes
    )

    out_dir = os.path.join(args.samples_dir, 'sample_folder')
    with torch.no_grad():
        count = 0
        while y_.next:
            z_.sample_()
            y_.sample_()
            if z_.shape[0] > y_.shape[0]:
                z_ = z_[y_.shape[0], :]
            if z_.shape[0] < y_.shape[0]:
                y_ = y_[z_.shape[0]]
            if args.parallel:
                nn.parallel.data_parallel(G, (z_, y_))
            else:
                image_tensors = G(z_, G.shared(y_))  # batch_size, 3, h, w
            save_samples(args, sample=image_tensors, save_dir=out_dir, meta={'y': y_})


def get_random_w(margin=0):
    """
    get a random float number in [margin, 1-margin]
    """
    return round(random.random() * (1- 2*margin) + margin, 2)


def sample_interplation(args, config, G):
    args.sample_count = 0

    interp_style = 'interplate' + ('Z' if not args.fix_z else '') + ('Y' if not args.fix_y else '')
    assert interp_style != 'interplate', 'one of noise and class embedding should not be fixed!'
    out_dir = os.path.join(args.samples_dir, interp_style)

    pair_count = 0
    if args.num_midpoints > 0:
        batch_size = max(1, args.batch_size // args.num_midpoints)
    else:
        batch_size = args.batch_size
    while pair_count < args.num_pairs:
        if args.num_midpoints <= 0:
            w = torch.tensor([get_random_w(margin=0.1)], device=config["device"])
        else:
            w = torch.linspace(0, 1.0, num_midpoints + 2, device=config["device"]) # num_midpoints
        # Prepare zs and ys
        if args.fix_z:  # If fix Z, only sample 1 z per pair
            if args.num_midpoints > 0:
                zs = torch.randn(batch_size, 1, G.dim_z, device=config["device"])
                zs = zs.repeat(1, args.num_midpoints + 2, 1).view(-1, G.dim_z)
            else:
                zs = torch.randn(batch_size, G.dim_z, device=config["device"])
        else:
            if args.num_midpoints > 0:
                z1 = torch.randn(batch_size, 1, G.dim_z, device=config["device"])
                z2 = torch.randn(batch_size, 1, G.dim_z, device=config["device"])
                zs = z1 * w.view(1, -1, 1) + z2 * (1 - w.view(1, -1, 1))
                zs = zs.view(-1, G.dim_z)
            else:
                z1 = torch.randn(batch_size, G.dim_z, device=config["device"])
                z2 = torch.randn(batch_size, G.dim_z, device=config["device"])
                zs = z1 * w.view(1, -1) + (1 - w.view(1, -1)) * z2

        if args.fix_y:  # If fix y, only sample 1 y per pair
            y = utils.sample_1hot(batch_size, config["n_classes"]) # batch_size
            if args.num_midpoints > 0:
                ys = G.shared(y).view(batch_size, 1, -1)
                ys = ys.repeat(1, args.num_midpoints + 2, 1).view(batch_size * (args.num_midpoints + 2), -1)
                y = y.repeat(1, args.num_midpoints + 2).view(-1)
            else:
                ys = G.shared(y).view(batch_size, -1)
        else:
            y1 = utils.sample_1hot(batch_size, config["n_classes"])
            y2 = utils.sample_1hot(batch_size, config["n_classes"])
            if args.num_midpoints > 0:
                ys = G.shared(y1).view(batch_size, 1, -1) * w.view(1, -1, 1) + G.shared(y2).view(batch_size, 1, -1) * w.view(1, -1, 1)
                ys = ys.view(batch_size * (args.num_midpoints+2), -1)
                y1 = y1.repeat(1, args.num_midpoints + 2).view(-1)
                y2 = y2.repeat(1, args.num_midpoints + 2).view(-1)
            else:
                ys = G.shared(y1).view(batch_size, -1) * w.view(1, -1) + (1 - w.view(1, -1)) * G.shared(y2).view(batch_size, -1)
        # Run the net--note that we've already passed y through G.shared.
        if G.fp16:
            zs = zs.half()
        with torch.no_grad():
            if args.parallel:
                out_ims = nn.parallel.data_parallel(G, (zs, ys)).data.cpu()
            else:
                out_ims = G(zs, ys).data.cpu()
        if not args.fix_y:
            meta = {
                'y1': y1,
                'y2': y2,
                'w': w
            }
        else:
            meta = {
                'y': y,
                'w': w
            }
        save_samples(args, sample=out_ims, save_dir=out_dir, meta=meta)
        pair_count += batch_size


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
    if args.sample_type == 'sample_interplation':
        sample_interplation(args, config, G)
