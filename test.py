import copy
import numpy as np
from src.task_vectors import TaskVector, weighted_sum
from src.eval import eval_single_dataset
from src.args import parse_arguments

# Config
datasets = ['MNIST', 'RESISC45']
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]

# Test loop
task_accs = {}
for i, weight in enumerate(np.arange(0.0,1.1,0.1)):
    alpha_vector = weighted_sum(task_vectors, [0.5, 0.5])
    tmp_alphas = [weight, 1 - weight]
    merged_task_vector = weighted_sum(task_vectors, tmp_alphas)
    # Exchange weights of the last linear layer
    merged_task_vector.vector['model.visual.transformer.resblocks.23.mlp.c_proj.weight'] = alpha_vector.vector['model.visual.transformer.resblocks.23.mlp.c_proj.weight']
    merged_task_vector.vector['model.visual.transformer.resblocks.23.mlp.c_proj.bias'] = alpha_vector.vector['model.visual.transformer.resblocks.23.mlp.c_proj.bias']

    # Apply the resulting task vector
    image_encoder = merged_task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    # Evaluate
    task_accuracies = []
    for dataset in datasets:
        task_accuracies.append(
            eval_single_dataset(image_encoder, dataset, args)['top1']
        )
    task_accuracies.append(np.mean(task_accuracies))
    task_accs[str(weight)] = task_accuracies
np.save('axcept_last_class_layer_task_accs.npy', task_accs)
