import unittest

from torch.utils.data import DataLoader
from few_shot.models import SemanticBinaryClassifier
import torch
from few_shot.datasets import OmniglotDataset
from few_shot.core import NShotTaskSampler
import numpy as np
class TestSemanticClassifier(unittest.TestCase):
    def test(self):
        model = SemanticBinaryClassifier(1, 100, size_binary_layer=10)
        model.load_state_dict(torch.load("models\\semantic_classifier\\omniglot__n=10_k=100_epochs=10__lr=0.01.pth"))
        evaluation = OmniglotDataset('evaluation')
        evaluation_taskloader = DataLoader(
            evaluation,
            batch_sampler=NShotTaskSampler(evaluation, 1, n=10, k=20,
                                           fixed_k_classes=np.arange(20)),
            num_workers=8
        )
        for batch_index, batch in enumerate(evaluation_taskloader):
            x,y=batch
            print(y)