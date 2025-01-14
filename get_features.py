import sys
import tqdm
import torch
from src.args import parse_arguments
from src.datasets.registry import get_dataset
from src.task_vectors import TaskVector, weighted_sum
from src.datasets.common import get_features


# Config
datasets = ["SVHN", "SVHN"]
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.device = 'cpu'
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]
merged_task_vector = weighted_sum(task_vectors, [0.5, 0.5])
# Apply the resulting task vector
image_encoder = merged_task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
image_encoder.cache_dir = 'cache'
# Save activations
image_encoder.eval()
dataset = get_dataset(
    datasets[0],
    image_encoder.val_preprocess,
    location=args.data_location,
    batch_size=args.batch_size,
)

feat = get_features(is_train=False, image_encoder=image_encoder, dataset=dataset, device=args.device)
