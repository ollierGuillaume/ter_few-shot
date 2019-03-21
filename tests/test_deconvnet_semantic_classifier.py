import unittest
import matplotlib.pyplot as plt
from few_shot.models import FewShotClassifier, FewShotDeconv
from config import DATA_PATH
import torch
from few_shot.utils import setup_dirs
import os
import numpy as np
from PIL import Image
from skimage import io


def vis_layer(activ_map):
    plt.ion()
    plt.imshow(activ_map[:, :, 0], cmap='gray')

def decon_img(layer_output):
    raw_img = layer_output.data.numpy()[0].transpose(1,2,0)
    img = (raw_img-raw_img.min())/(raw_img.max()-raw_img.min())*255
    img = img.astype(np.uint8)
    return img

def vis_grid(Xs):
  """ visualize a grid of images """
  (N, H, W, C) = Xs.shape
  A = int(ceil(sqrt(N)))
  G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
  G *= np.min(Xs)
  n = 0
  for y in range(A):
    for x in range(A):
      if n < N:
        G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
        n += 1
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G


class TestDeconvNetFewShot(unittest.TestCase):
    def test(self):

        img_filename = os.path.join( DATA_PATH ,'Omniglot',
                                     'images_background',
                                     'Japanese_(hiragana).0',
                                     'character07',
                                     '0494_01.108002.png')
        img = io.imread(img_filename)
        img = img[np.newaxis, np.newaxis, :, :]
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.from_numpy(img)
        print(img.size())
        n = 5
        k = 900
        setup_dirs()
        assert torch.cuda.is_available()

        device = torch.device('cpu')
        torch.backends.cudnn.benchmark = True

        model = FewShotClassifier(1, k).to(device, dtype=torch.double)
        model.load_state_dict(torch.load(os.path.join("models", "semantic_classifier",
                                                      "test_k=900_few_shot_classifier.pth")))

        conv_out = model(img)

        deconv_model = FewShotDeconv(model)

        conv_layer_indices = model.get_conv_layer_indices()

        plt.ion()  # remove blocking
        plt.figure(figsize=(10, 5))

        done = False
        while not done:
            layer = input('Layer to view (0-4, -1 to exit): ')
            try:
                layer = int(layer)
            except ValueError:
                continue
            print(model.feature_outputs)
            activ_map = model.feature_outputs[layer].data.numpy()
            activ_map = activ_map.transpose(1, 2, 3, 0)
            activ_map_grid = vis_grid(activ_map)
            vis_layer(activ_map_grid)

            # only transpose convolve from Conv2d or ReLU layers
            conv_layer = layer
            if conv_layer not in conv_layer_indices:
                conv_layer -= 1
                if conv_layer not in conv_layer_indices:
                    continue

            n_maps = activ_map.shape[0]

            marker = None
            while True:
                choose_map = input('Select map?  (y/[n]): ') == 'y'
                if marker != None:
                    marker.pop(0).remove()

                if not choose_map:
                    break

                _, map_x_dim, map_y_dim, _ = activ_map.shape
                map_img_x_dim, map_img_y_dim, _ = activ_map_grid.shape
                x_step = map_img_x_dim // (map_x_dim + 1)

                print('Click on an activation map to continue')
                x_pos, y_pos = plt.ginput(1)[0]
                x_index = x_pos // (map_x_dim + 1)
                y_index = y_pos // (map_y_dim + 1)
                map_idx = int(x_step * y_index + x_index)

                if map_idx >= n_maps:
                    print('Invalid map selected')
                    continue

                decon = deconv_model(model.feature_outputs[layer][0][map_idx][None, None, :, :], conv_layer, map_idx,
                                model.pool_indices)
                img = decon_img(decon)
                plt.subplot(121)
                marker = plt.plot(x_pos, y_pos, marker='+', color='red')
                plt.subplot(122)
                plt.imshow(img)


if __name__ == '__main__':
    unittest.main()
