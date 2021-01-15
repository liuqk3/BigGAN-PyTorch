import torch
import torchvision
import utils

parser = utils.prepare_parser()
parser = utils.add_sample_parser(parser)
config = vars(parser.parse_args())

# See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
config["resolution"] = utils.imsize_dict["I128"]
config["n_classes"] = utils.nclass_dict["I128"]
config["G_activation"] = utils.activation_dict["inplace_relu"]
config["D_activation"] = utils.activation_dict["inplace_relu"]
config["G_attn"] = "64"
config["D_attn"] = "64"
config["G_ch"] = 96
config["D_ch"] = 96
config["hier"] = True
config["dim_z"] = 120
config["shared_dim"] = 128
config["G_shared"] = True
config = utils.update_config_roots(config)
config["skip_init"] = True
config["no_optim"] = True
config["device"] = "cuda"

# Seed RNG.
utils.seed_rng(config["seed"])

# Set up cudnn.benchmark for free speed.
torch.backends.cudnn.benchmark = True

# Import the model.
model = __import__(config["model"])
G = model.Generator(**config).to(config["device"])
utils.count_parameters(G)

# Load weights with ema.
weights_path = "/home/t-qiankunliu/Program/BigGAN-PyTorch/weights/BigGAN_I128_hdf5_seed0_Gch96_Dch96_bs256_nDa8_nGa8_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/G_ema.pth"  # Change this.
G.load_state_dict(torch.load(weights_path))

# Update batch size setting used for G.
G_batch_size = max(config["G_batch_size"], config["batch_size"])
(z_, y_) = utils.prepare_z_y(
    G_batch_size,
    G.dim_z,
    config["n_classes"],
    device=config["device"],
    fp16=config["G_fp16"],
    z_var=config["z_var"],
)

G.eval()

out_path = "/home/t-qiankunliu/Program/BigGAN-PyTorch/samples/debug/random_samples.jpg"  # Change this.
with torch.no_grad():
    z_.sample_()
    y_.sample_()
    image_tensors = G(z_, G.shared(y_))
    torchvision.utils.save_image(
        image_tensors,
        out_path,
        nrow=int(G_batch_size ** 0.5),
        normalize=True,
    )