import torch
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments

# Config
datasets = ['MNIST', 'RESISC45']
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'path/to/data'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]
# Sum the task vectors
task_vector_sum = sum(task_vectors)
# Apply the resulting task vector
image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0.8)

# Evaluate
for dataset in datasets:
    eval_single_dataset(image_encoder, dataset, args)