import torch
import numpy as np
from src.args import parse_arguments

# Config
datasets = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'
args.openclip_cachedir = f'./'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

for dataset in datasets:
    finetuned_state_dict = torch.load(f'checkpoints/{model}/{dataset}/finetuned.pt').state_dict()
    print('\n')
    print(f'{dataset} sum: {finetuned_state_dict["model.visual.proj"][100,0]}')
    print('\n')

    # tmp_weight_list = []
    # for l in layer_ids:
    #     tmp_layer_weight = finetuned_state_dict[f'model.visual.transformer.resblocks.{l}.mlp.c_proj.weight'].numpy().flatten()
    #     tmp_layer_bias = finetuned_state_dict[f'model.visual.transformer.resblocks.{l}.mlp.c_proj.bias'].numpy().flatten()
    #     tmp_weight_list.append(
    #         np.concatenate([tmp_layer_weight, tmp_layer_bias])
    #         )
    # tmp_weight_list = np.vstack(tmp_weight_list)
    # np.save(f'/Users/dg/Projects/task_vectors/mlp_weights/{dataset}_weight_layers.npy', tmp_weight_list)
