import numpy as np
from src.task_vectors import TaskVector, weighted_sum
from src.eval import eval_single_dataset
from src.args import parse_arguments

# Config
datasets = ['SVHN', 'RESISC45']
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
for weight in np.arange(0,1.0,0.1):
    tmp_alphas = [weight, 1 - weight]
    print(f'Tmp. alphas: {tmp_alphas}')
    merged_task_vector = weighted_sum(task_vectors, tmp_alphas)
    # Apply the resulting task vector
    image_encoder = merged_task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    # Evaluate
    task_accuracies = []
    for dataset in datasets:
        task_accuracies.append(
            eval_single_dataset(image_encoder, dataset, args)['top1']
        )
    for i, acc in enumerate(task_accuracies):
        print(f'Test accuracy task {datasets[i]}: {acc:.4f}%')
    print(f'Multitask accuracy: {np.mean(task_accuracies):.4f}% \n')