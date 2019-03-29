import unittest

from torch.utils.data import DataLoader
from few_shot.models import SemanticBinaryClassifier, FewShotClassifier
import torch
from few_shot.datasets import OmniglotDataset
from few_shot.core import NShotTaskSampler
from few_shot.callbacks import *
from few_shot.core import BasicSampler, create_nshot_task_label, EvaluateFewShot, prepare_nshot_task
from few_shot.utils import setup_dirs
from few_shot.train import fit, gradient_step
from config import PATH
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from shutil import copyfile


class TestSemanticBinaryClassifier(nn.Module):
    def __init__(self, k_way: int, semantic_model, size_binary_layer=10):

        super(TestSemanticBinaryClassifier, self).__init__()
        self.model = semantic_model
        self.logits = nn.Linear(size_binary_layer, k_way)

    def forward(self, x):
        _, x = self.model(x)
        return self.logits(x)


class TestSemanticClassification(unittest.TestCase):
    def test(self):
        k = 200
        n = 5
        epochs = 20
        size_binary_layer = 30
        stochastic = True
        n_conv_layers = 2
        lr = 0.01

        model_name = 'omniglot__n=5_k=200_epochs=500__lr=0.01__size_binary_layer=30__stochastic__n_conv_layers=2'
        validation_split = .2

        setup_dirs()
        assert torch.cuda.is_available()

        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        model = SemanticBinaryClassifier(1, k, size_binary_layer=size_binary_layer, stochastic=stochastic,
                                         size_dense_layer_before_binary=None,
                                         n_conv_layers=n_conv_layers)
        model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier",
                                                     model_name+".pth")))

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
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

        test_model = TestSemanticBinaryClassifier(k, model, size_binary_layer=size_binary_layer).to(device, dtype=torch.double)
        loss_fn = nn.CrossEntropyLoss().to(device)

        def prepare_batch(n, k):
            def prepare_batch_(batch):
                x, y = batch
                x = x.double().cuda()
                # Create dummy 0-(num_classes - 1) label
                y = create_nshot_task_label(k, n).cuda()
                return x, y
            return prepare_batch_

        evalmetrics = EvaluateMetrics(eval_dataloader)
        evalmetrics.set_params({'metrics': ['categorical_accuracy'],
                            'prepare_batch': prepare_batch(n, k),
                            'loss_fn': loss_fn})

        callbacks = [
            evalmetrics,

            ModelCheckpoint(
                filepath=os.path.join(PATH, 'models', 'semantic_classifier', model_name + 'test_other_class.pth'),
                monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
            ),
            CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', model_name + 'test_other_class.csv'))
        ]

        #print(summary(model, (1, 28, 28)))
        for param in model.parameters():
            param.requires_grad = False
        fit(
            test_model,
            optimiser,
            loss_fn,
            epochs=100,
            dataloader=train_dataloader,
            prepare_batch=prepare_batch(n, k),
            callbacks=callbacks,
            metrics=['categorical_accuracy'],
            fit_function=gradient_step,
            fit_function_kwargs={'n_shot': n, 'k_way': k, 'device': device},
        )
    #
    # def test_view_binary(self):
    #     k = 200
    #     n = 5
    #     epochs = 500
    #     size_binary_layer = 30
    #     stochastic = False
    #
    #     model_name = 'omniglot__n=' + str(n) + '_k=' + str(k) + '_epochs=' + str(
    #         epochs) + '__lr=0.01__size_binary_layer=' \
    #                  + str(size_binary_layer) + ('__stochastic' if stochastic else '__deterministic')
    #     setup_dirs()
    #     assert torch.cuda.is_available()
    #
    #     device = torch.device('cuda')
    #     torch.backends.cudnn.benchmark = True
    #
    #     model = SemanticBinaryClassifier(1, k, size_binary_layer=size_binary_layer, stochastic=stochastic).to(device, dtype=torch.double)
    #
    #     model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier", model_name+".pth")))
    #     evaluation = OmniglotDataset('evaluation')
    #
    #     classes = np.random.choice(evaluation.df['class_id'].unique(), size=20)
    #     for i in classes:
    #         evaluation.df[evaluation.df['class_id'] == i] = evaluation.df[evaluation.df['class_id'] == i].sample(frac=1)
    #
    #     validation_split = 0
    #
    #     eval_dataloader = DataLoader(
    #              evaluation,
    #              batch_sampler=BasicSampler(evaluation, validation_split, True, classes, n=5),
    #              num_workers=8
    #          )
    #
    #     model.eval()
    #
    #     pd.options.display.max_colwidth = 200
    #     with torch.no_grad():
    #         for batch_index, batch in enumerate(eval_dataloader):
    #             x, y = batch
    #             print("x shape:", x.shape)
    #             x = x.double().cuda()
    #             _, bin_x = model(x)
    #             # print("x:",x)
    #             # print("bin x:", bin_x)
    #
    #             i = 0
    #             print(bin_x)
    #             for classe in classes:
    #                 print(evaluation.df[evaluation.df['class_id'] == classe]['filepath'][:n].to_string())
    #                 for j in range(n):
    #                     print(bin_x[i][3])
    #                     i += 1
    #             break

    # def test_view_conv(self):
    #     n = 5
    #     k = 200
    #
    #     setup_dirs()
    #     assert torch.cuda.is_available()
    #
    #     device = torch.device('cuda')
    #     torch.backends.cudnn.benchmark = True
    #
    #     model = FewShotClassifier(1, k).to(device, dtype=torch.double)
    #     background = OmniglotDataset('background')
    #
    #     classes = np.random.choice(background.df['class_id'].unique(), size=k)
    #     for i in classes:
    #         background.df[background.df['class_id'] == i] = background.df[background.df['class_id'] == i].sample(frac=1)
    #
    #     train_dataloader = DataLoader(
    #         background,
    #         batch_sampler=BasicSampler(background, 0.2, True, classes, n=n),
    #         num_workers=8
    #     )
    #
    #     eval_dataloader = DataLoader(
    #         background,
    #         batch_sampler=BasicSampler(background, 0.2, False, classes, n=n),
    #         num_workers=8
    #     )
    #
    #     def prepare_batch(n, k):
    #         def prepare_batch_(batch):
    #             x, y = batch
    #             x = x.double().cuda()
    #             # Create dummy 0-(num_classes - 1) label
    #             y = create_nshot_task_label(k, n).cuda()
    #             # for e in x:
    #             #     plt.imshow(e.cpu().squeeze().numpy())
    #             #     plt.show()
    #             return x, y
    #
    #         return prepare_batch_
    #
    #     optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    #     loss_fn = nn.CrossEntropyLoss().to(device)
    #
    #     evalmetrics = EvaluateMetrics(eval_dataloader)
    #     evalmetrics.set_params({'metrics': ['categorical_accuracy'],
    #                             'prepare_batch': prepare_batch(n, k),
    #                             'loss_fn': loss_fn})
    #     progressbar = ProgressBarLogger()
    #     progressbar.set_params({'num_batches': k * n, 'metrics': ['categorical_accuracy'], 'loss': loss_fn,
    #                             'verbose': 1})
    #     callbacks = [
    #         evalmetrics,
    #         progressbar,
    #
    #         ModelCheckpoint(
    #             filepath=os.path.join(PATH, 'models', 'semantic_classifier', 'test_k=200_few_shot_classifier1.pth'),
    #             monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    #         ),
    #         ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
    #         CSVLogger(os.path.join(PATH, 'logs', 'semantic_classifier', 'test_k=200_few_shot_classifier.csv'))
    #     ]
    #
    #     fit(
    #         model,
    #         optimiser,
    #         loss_fn,
    #         epochs=50,
    #         dataloader=train_dataloader,
    #         prepare_batch=prepare_batch(n, k),
    #         callbacks=callbacks,
    #         metrics=['categorical_accuracy'],
    #         fit_function=gradient_step,
    #         fit_function_kwargs={'n_shot': n, 'k_way': k, 'device': device},
    #     )

        # model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier", "test1.pth")))
        # body_model = [i for i in model.children()][0]
        # layer1 = body_model[0]
        # #tensor = layer1.weight.data.cpu().squeeze().numpy()
        # # for i in range(64):
        # #     plt.imshow(tensor[i], cmap='gray')
        # #     plt.show()
        #
        # model.eval()
        #
        # pd.options.display.max_colwidth = 200
        #
        # dic_filters_activations = []
        # for _ in range(4):
        #     layer = []
        #     for _ in range(64):
        #         layer.append({})
        #     dic_filters_activations.append(layer)
        #
        # with torch.no_grad():
        #     for batch_index, batch in enumerate(eval_dataloader):
        #         x, y = batch
        #         x = x.double().cuda()
        #         layers_out = model.view(x)
        #         # print("bin x:", bin_x)
        #
        #
        #         i = 0
        #         for classe in classes:
        #             for j in range(n):
        #
        #                 name = background.df[background.df['class_id'] == classe]['filepath'].iloc[j]
        #                 for layer in range(4):
        #                     feat_x = layers_out[layer][i]
        #                     for filter in range(64):
        #                         out_filter = feat_x[filter]
        #                         max_activation = torch.max(out_filter).item()
        #                         dic_filters_activations[layer][filter][name] = max_activation
        #                 i += 1
        #         break
        #     for layer in range(4):
        #         for filter in range(64):
        #             n = 0
        #             print("filter:", filter)
        #             for key, value in sorted(dic_filters_activations[layer][filter].items(), key=lambda kv: (kv[1], kv[0]),
        #                                      reverse=True):
        #                 print(key, value)
        #                 if not os.path.exists( os.path.join('view_model', str(layer), 'filter'+str(filter))):
        #                     os.makedirs(os.path.join('view_model', str(layer), 'filter'+str(filter)))
        #                 copyfile(key, os.path.join('view_model', str(layer), 'filter'+str(filter), str(n)+'.png'))
        #                 n += 1
        #                 if n == 20:
        #                     break
