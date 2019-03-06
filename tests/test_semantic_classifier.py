import unittest

from torch.utils.data import DataLoader
from few_shot.models import SemanticBinaryClassifier
import torch
from few_shot.datasets import OmniglotDataset
from few_shot.core import NShotTaskSampler
from few_shot.callbacks import *
from few_shot.core import BasicSampler, create_nshot_task_label, EvaluateFewShot, prepare_nshot_task
from few_shot.utils import setup_dirs
from few_shot.train import fit, gradient_step
from config import PATH
from torch import nn
import argparse
import numpy as np
import os


class TestSemanticBinaryClassifier(nn.Module):
    def __init__(self, k_way: int, semantic_model, size_binary_layer=10):

        super(TestSemanticBinaryClassifier, self).__init__()
        self.model = semantic_model
        self.logits = nn.Linear(size_binary_layer, k_way)

    def forward(self, x):
        _, x = self.model(x)
        return self.logits(x)


class TestSemanticClassifier(unittest.TestCase):
    # def test(self):
    #     # parser = argparse.ArgumentParser()
    #     #
    #     # parser.add_argument('--n', default=1, type=int)
    #     # parser.add_argument('--k', default=100, type=int)
    #     # parser.add_argument('--lr', default=0.01, type=float)
    #     # parser.add_argument('--epochs', default=10, type=int)
    #     # parser.add_argument('--size-binary-layer', default=10, type=int)
    #     # parser.add_argument('--stochastic', action='store_true')
    #     #
    #     # parser.add_argument('--epochs-test-model', default=10, type=int)
    #     # args = parser.parse_args()
    #     k = 200
    #     n = 5
    #     lr = 0.01
    #     epochs = 500
    #     size_binary_layer = 50
    #     stochastic = False
    #
    #     model_name = 'omniglot__n='+str(n)+'_k='+str(k)+'_epochs='+str(epochs)+'__lr=0.01__size_binary_layer='\
    #                  +str(size_binary_layer)+('__stochastic' if stochastic else '__deterministic')
    #     validation_split = .2
    #
    #     setup_dirs()
    #     assert torch.cuda.is_available()
    #
    #     device = torch.device('cuda')
    #     torch.backends.cudnn.benchmark = True
    #
    #     model = SemanticBinaryClassifier(1, k, size_binary_layer=size_binary_layer, stochastic=stochastic)
    #     model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier",
    #                                                   model_name+".pth")))
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     evaluation = OmniglotDataset('evaluation')
    #
    #     classes = np.random.choice(evaluation.df['class_id'].unique(), size=k)
    #     for i in classes:
    #         evaluation.df[evaluation.df['class_id'] == i] = evaluation.df[evaluation.df['class_id'] == i].sample(frac=1)
    #
    #     train_dataloader = DataLoader(
    #         evaluation,
    #         batch_sampler=BasicSampler(evaluation, validation_split, True, classes, n=n),
    #         num_workers=8
    #     )
    #
    #     eval_dataloader = DataLoader(
    #         evaluation,
    #         batch_sampler=BasicSampler(evaluation, validation_split, False, classes, n=n),
    #         num_workers=8
    #     )
    #
    #     test_model = TestSemanticBinaryClassifier(k, model, size_binary_layer=size_binary_layer)\
    #         .to(device, dtype=torch.double)
    #     optimiser = torch.optim.Adam(test_model.parameters(), lr=lr)
    #     loss_fn = nn.CrossEntropyLoss().to(device)
    #
    #     def prepare_batch(n, k):
    #         def prepare_batch_(batch):
    #             x, y = batch
    #             x = x.double().cuda()
    #             # Create dummy 0-(num_classes - 1) label
    #             y = create_nshot_task_label(k, n).cuda()
    #             return x, y
    #         return prepare_batch_
    #
    #     progressbar = ProgressBarLogger()
    #     progressbar.set_params({'num_batches': k * n, 'metrics': ['categorical_accuracy'], 'loss': loss_fn,
    #                         'verbose': 1})
    #     evalmetrics = EvaluateMetrics(eval_dataloader)
    #     evalmetrics.set_params({'metrics': ['categorical_accuracy'],
    #                         'prepare_batch': prepare_batch(n, k),
    #                         'loss_fn': loss_fn})
    #
    #     callbacks = [
    #         evalmetrics,
    #         progressbar,
    #
    #         ModelCheckpoint(
    #             filepath=os.path.join(PATH, 'models', 'semantic_classifier', model_name + 'test_other_class.pth'),
    #             monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    #         ),
    #         ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
    #         CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', model_name + 'test_other_class.csv'))
    #     ]
    #
    #     fit(
    #         test_model,
    #         optimiser,
    #         loss_fn,
    #         epochs=100,
    #         dataloader=train_dataloader,
    #         prepare_batch=prepare_batch(n, k),
    #         callbacks=callbacks,
    #         metrics=['categorical_accuracy'],
    #         fit_function=gradient_step,
    #         fit_function_kwargs={'n_shot': n, 'k_way': k, 'device': device},
    #     )

    def test_view_binary(self):
        k = 200
        n = 5
        epochs = 500
        size_binary_layer = 50
        stochastic = False

        model_name = 'omniglot__n=' + str(n) + '_k=' + str(k) + '_epochs=' + str(
            epochs) + '__lr=0.01__size_binary_layer=' \
                     + str(size_binary_layer) + ('__stochastic' if stochastic else '__deterministic')
        setup_dirs()
        assert torch.cuda.is_available()

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        model = SemanticBinaryClassifier(1, k, size_binary_layer=size_binary_layer, stochastic=stochastic).to(device, dtype=torch.double)
        evaluation = OmniglotDataset('evaluation')

        classes = np.random.choice(evaluation.df['class_id'].unique(), size=20)
        for i in classes:
            evaluation.df[evaluation.df['class_id'] == i] = evaluation.df[evaluation.df['class_id'] == i].sample(frac=1)

        validation_split = 0

        eval_dataloader = DataLoader(
                 evaluation,
                 batch_sampler=BasicSampler(evaluation, validation_split, True, classes, n=5),
                 num_workers=8
             )

        model.eval()
        for batch_index, batch in enumerate(eval_dataloader):
            x, y = batch
            print("x shape:", x.shape)
            x = x.double().cuda()
            print("type x:", type(x))
            _, bin_x = model(x)
            print("bin x:", bin_x)
            # for e in x:
                # plt.imshow(e.cpu().squeeze().numpy())
                # plt.show()
            break