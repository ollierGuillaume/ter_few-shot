"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import BasicSampler, create_nshot_task_label, EvaluateFewShot, prepare_nshot_task
from few_shot.models import SemanticBinaryClassifier
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from few_shot.train import fit, gradient_step
from config import PATH
import numpy as np
import matplotlib.pyplot as plt

setup_dirs()
assert torch.cuda.is_available()

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batches', default=100, type=int)
    parser.add_argument('--size-binary-layer', default=10, type=int)
    # parser.add_argument('--eval-batches', default=20, type=int)

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
        raise (ValueError('Unsupported dataset'))

    param_str = str(args.dataset) + '__n=' + str(args.n) + '_k=' + str(args.k) \
                + '_epochs=' + str(args.epochs) + '__lr=' + str(args.lr) + '__size_binary_layer=' \
                + str(args.size_binary_layer)
    #            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
    print(param_str)

    ###################
    # Create datasets #
    ###################
    validation_split = .2

    split = int(np.floor(validation_split * args.n))

    background = dataset_class('background')

    classes = np.random.choice(background.df['class_id'].unique(), size=args.k)
    for i in classes:
        background.df[background.df['class_id'] == i] = background.df[background.df['class_id'] == i].sample(frac=1)

    train_dataloader = DataLoader(
        background,
        batch_sampler=BasicSampler(background, validation_split, True, classes, n=args.n),
        num_workers=8
    )

    eval_dataloader = DataLoader(
        background,
        batch_sampler=BasicSampler(background, validation_split, False, classes, n=args.n),
        num_workers=8
    )
    # evaluation = dataset_class('evaluation')
    # evaluation_taskloader = DataLoader(
    #     evaluation,
    #     batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k),
    #     num_workers=8
    # )

    ############
    # Training #
    ############
    print('Training semantic classifier on '+str(args.dataset)+'...')
    model = SemanticBinaryClassifier(num_input_channels, args.k, fc_layer_size, size_binary_layer=args.size_binary_layer).to(device,
                                                                                                         dtype=torch.double)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().to(device)


    def prepare_batch(n, k):
        def prepare_batch_(batch):
            x, y = batch
            x = x.double().cuda()
            # Create dummy 0-(num_classes - 1) label
            y = create_nshot_task_label(k, n).cuda()
            # for e in x:
            #     plt.imshow(e.cpu().squeeze().numpy())
            #     plt.show()
            return x, y

        return prepare_batch_


    progressbar = ProgressBarLogger()
    progressbar.set_params({'num_batches': args.k * args.n, 'metrics': ['categorical_accuracy'], 'loss': loss_fn,
                            'verbose': 1})
    evalmetrics = EvaluateMetrics(eval_dataloader)
    evalmetrics.set_params({'metrics': ['categorical_accuracy'],
                            'prepare_batch': prepare_batch(args.n, args.k),
                            'loss_fn': loss_fn})

    callbacks = [
        evalmetrics,
        progressbar,

        ModelCheckpoint(
            filepath=os.path.join(PATH, 'models', 'semantic_classifier', str(param_str) + '.pth'),
            monitor='val_' + str(args.n) + '-shot_' + str(args.k) + '-way_acc'
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
        CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', str(param_str) + '.csv'))
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=args.epochs,
        dataloader=train_dataloader,
        prepare_batch=prepare_batch(args.n, args.k),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=gradient_step,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'device': device},
    )
