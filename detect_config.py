from models_1.detect_net import WpodNet
import torch
image_root = 'train_enhance'
batch_size = 96
weight = 'weights/wpod_net.pt'
# weight = ''
output_weight = 'weights/wpod_net_new.pt'
epoch = 1000
net = WpodNet
device = 'cuda:0'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')



