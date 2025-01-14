import sys
import tqdm
import torch
from src.args import parse_arguments
from src.datasets.registry import get_dataset
from src.task_vectors import TaskVector, weighted_sum
from src.datasets.common import get_features


# Config
dataset_name = "SVHN"
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.device = 'mps'
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model.cache_dir = './'
# Apply the resulting task vector
# Save activations
pretrained_model.eval()
dataset = get_dataset(
    dataset_name,
    pretrained_model.val_preprocess,
    location=args.data_location,
    batch_size=16,
)
feat = get_features(is_train=False, image_encoder=pretrained_model, dataset=dataset, device=args.device)
