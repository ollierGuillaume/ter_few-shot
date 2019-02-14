"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--train-batches', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)

# parser.add_argument('--inner-train-steps', default=1, type=int)
# parser.add_argument('--inner-val-steps', default=3, type=int)

args = parser.parse_args()

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3
else:
    raise(ValueError('Unsupported dataset'))

param_str = f'{args.dataset}__n={args.n}_k={args.k}_batch={args.batch_size}_'
#            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
print(param_str)


###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.train_batches, n=args.n, k=args.k, q=None,
                                   num_tasks=args.batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=None,
                                   num_tasks=args.batch_size),
    num_workers=8
)


############
# Training #
############
print(f'Training semantic classifier on {args.dataset}...')
model = FewShotClassifier(num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss().to(device)


def prepare_batch(n, k, batch_size):
    def prepare_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the model
        x = x.reshape(batch_size, n*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        # y = create_nshot_task_label(k, q).cuda().repeat(batch_size)
        return x, y

    return prepare_batch_


callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_batch(args.n, args.k, args.q, args.meta_batch_size),
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/semantic_classifier/{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc'
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
    CSVLogger(PATH + f'/logs/semantic_classifier/{param_str}.csv'),
]


fit(
    model,
    optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_batch(args.n, args.k, args.batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k,
                         'train': True,
                         'device': device, 'inner_train_steps': args.inner_train_steps,
                         'lr': args.lr},
)
