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

import numpy as np
import os


class TestSemanticBinaryClassifier(nn.Module):
    def __init__(self, num_input_channels: int, k_way: int, semantic_model,
                 size_binary_layer=10):

        super(TestSemanticBinaryClassifier, self).__init__()
        self.model = semantic_model
        self.logits = nn.Linear(size_binary_layer, k_way)

    def forward(self, x):
        _, x = self.model(x)
        return self.logits(x)


class TestSemanticClassifier(unittest.TestCase):
    def test(self):
        k = 200
        n = 5
        lr = 0.01
        epochs = 50
        model_name = "omniglot__n=5_k=200_epochs=500__lr=0.01__size_binary_layer=20"
        fc_layer_size = 64
        num_input_channels = 1
        size_binary_layer = 20
        validation_split = .2

        setup_dirs()
        assert torch.cuda.is_available()

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        model = SemanticBinaryClassifier(1, 100, size_binary_layer=size_binary_layer)
        model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier",
                                                      model_name+".pth")))
        for param in model.parameters():
            param.requires_grad = False

        evaluation = OmniglotDataset('evaluation')

        classes = np.random.choice(evaluation.df['class_id'].unique(), size=k)
        for i in classes:
            evaluation.df[evaluation.df['class_id'] == i] = evaluation.df[evaluation.df['class_id'] == i].sample(frac=1)

        train_dataloader = DataLoader(
            evaluation,
            batch_sampler=BasicSampler(evaluation, validation_split, True, classes, n=n),
            num_workers=8
        )

        eval_dataloader = DataLoader(
            evaluation,
            batch_sampler=BasicSampler(evaluation, validation_split, False, classes, n=n),
            num_workers=8
        )

        test_model = TestSemanticBinaryClassifier(num_input_channels, k, model, size_binary_layer=10)\
            .to(device,dtype=torch.double)
        optimiser = torch.optim.Adam(test_model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss().to(device)

        def prepare_batch(n, k):
            def prepare_batch_(batch):
                x, y = batch
                x = x.double().cuda()
                # Create dummy 0-(num_classes - 1) label
                y = create_nshot_task_label(k, n).cuda()
                return x, y
            return prepare_batch_

        progressbar = ProgressBarLogger()
        progressbar.set_params({'num_batches': k * n, 'metrics': ['categorical_accuracy'], 'loss': loss_fn,
                            'verbose': 1})
        evalmetrics = EvaluateMetrics(eval_dataloader)
        evalmetrics.set_params({'metrics': ['categorical_accuracy'],
                            'prepare_batch': prepare_batch(n, k),
                            'loss_fn': loss_fn})

        callbacks = [
            evalmetrics,
            progressbar,

            ModelCheckpoint(
                filepath=os.path.join(PATH, 'models', 'semantic_classifier', model_name + 'test_other_class.pth'),
                monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
            ),
            ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
            CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', model_name + 'test_other_class.csv'))
        ]

        fit(
            test_model,
            optimiser,
            loss_fn,
            epochs=epochs,
            dataloader=train_dataloader,
            prepare_batch=prepare_batch(n, k),
            callbacks=callbacks,
            metrics=['categorical_accuracy'],
            fit_function=gradient_step,
            fit_function_kwargs={'n_shot': n, 'k_way': k, 'device': device},
        )