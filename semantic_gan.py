from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import BasicSampler, create_nshot_task_label, EvaluateFewShot, prepare_nshot_task
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from few_shot.train import fit, gradient_step
from config import PATH
import numpy as np
from few_shot.semantic_gan import SemanticBinaryEncoder, SemanticBinaryDecoder, SemanticBinaryDiscriminator, SemanticBinaryClassifierLatentSpace, fit_gan_few_shot
import torch
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

    parser.add_argument('--n', default=1, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--size-binary-layer', default=10, type=int)
    parser.add_argument('--size-continue-layer', default=10, type=int)
    parser.add_argument('--stochastic', action='store_true')
    # parser.add_argument('--eval-batches', default=20, type=int)

    # parser.add_argument('--inner-train-steps', default=1, type=int)
    # parser.add_argument('--inner-val-steps', default=3, type=int)

    args = parser.parse_args()

    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
    dataset = 'Omniglot'
    param_str = str(dataset) + '__n=' + str(args.n) + '_k=' + str(args.k) \
                  + '_epochs=' + str(args.epochs) + '__lr=' + '__size_binary_layer=' \
                  + str(args.size_binary_layer) + '__size_continue_layer=' + str(args.size_continue_layer) \
                  + ('__stochastic' if args.stochastic else '__deterministic')
    #            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'

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
    print('Training semantic GAN on ' + str(dataset) + '...')
    classifier = SemanticBinaryClassifierLatentSpace(args.size_binary_layer, args.k).to(device, dtype=torch.double)
    encoder = SemanticBinaryEncoder(num_input_channels, args.size_binary_layer, args.size_continue_layer,
                                    stochastic=args.stochastic).to(device, dtype=torch.double)
    discriminator = SemanticBinaryDiscriminator(num_input_channels, fc_layer_size).to(device, dtype=torch.double)
    generator = SemanticBinaryDecoder(args.size_binary_layer, args.size_continue_layer).to(device, dtype=torch.double)

    lr_fast = 0.0002
    lr_slow = 0.0001

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_slow)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_fast)
    c_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr_slow)
    e_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr_fast)


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
    progressbar.set_params({'num_batches': args.k * args.n, 'metrics': ['categorical_accuracy'],
                            'verbose': 1})
    evalmetrics = EvaluateMetrics(eval_dataloader)

    callbacks = [
        evalmetrics,
        progressbar,

        ModelCheckpoint(
            filepath=os.path.join(PATH, 'models', 'semantic_classifier', str(param_str) + '.pth'),
            monitor='val_' + str(args.n) + '-shot_' + str(args.k) + '-way_acc'
        ),
        CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', str(param_str) + '.csv'))
    ]


    fit_gan_few_shot(
        encoder,
        generator,
        classifier,
        discriminator,
        train_dataloader,
        param_str,
        args.k,
        args.n,
        args.epochs,
        prepare_batch(args.n, args.k),
        args.size_continue_layer,
        args.size_binary_layer,
        device,
        e_optimizer,
        g_optimizer,
        c_optimizer,
        d_optimizer,
        metrics=['categorical_accuracy'],
        callbacks=callbacks
    )
