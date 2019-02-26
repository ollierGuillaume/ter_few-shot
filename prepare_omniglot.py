"""
Run this script to prepare the Omniglot dataset from the raw Omniglot dataset that is found at
https://github.com/brendenlake/omniglot/tree/master/python.
This script prepares an enriched version of Omniglot the same as is used in the Matching Networks and Prototypical
Networks papers.
1. Augment classes with rotations in multiples of 90 degrees.
2. Downsize images to 28x28
3. Uses background and evaluation sets present in the raw dataset
"""
from skimage import io
from skimage import transform
import zipfile
import shutil
import os

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

import torchvision.transforms
import numpy as np
from random import uniform
from math import pi

no_preprocessing = False
# Parameters
dataset_zip_files = ['images_background.zip', 'images_evaluation.zip']
raw_omniglot_location = DATA_PATH + '/Omniglot_Raw/'
prepared_omniglot_location = DATA_PATH + '/Omniglot/'
output_shape = (28, 28)

id_img = 0


def handle_characters(alphabet_folder, character_folder, rotate=0, n_variations=None):
    global id_img
    for root, _, character_images in os.walk(character_folder):
        character_name = root.split('/')[-1]
        mkdir(str(alphabet_folder) + '.' + str(rotate) + '/' + str(character_name))
        for img_path in character_images:
            # print(root+'/'+img_path)
            img = io.imread(root + '/' + img_path)
            img = transform.rotate(img, angle=rotate)
            img = transform.resize(img, output_shape, anti_aliasing=True)
            img = (img - img.min()) / (img.max() - img.min())
            img = 1 - img
            if n_variations is not None:
                for _ in range(n_variations):
                    basename_img = img_path.split(".")[0]

                    trans = transform.AffineTransform(shear=uniform(-pi / 6, pi / 6),
                                                      rotation=uniform(-pi / 6, pi / 6))

                    shift_y, shift_x = np.array(img.shape[:2]) / 2.
                    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
                    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])

                    img_trans = transform.warp(img, (tf_shift + (trans + tf_shift_inv)).inverse)
                    io.imsave(str(alphabet_folder) + '.' + str(rotate) + '/' + str(character_name) + '/' + \
                              str(basename_img) + '.' + str(id_img) + '.png', img_trans)
                    id_img += 1
            else:
                io.imsave(str(alphabet_folder) + '.' + str(rotate) + '/' + str(character_name) + '/' + \
                              str(img_path), img)
            # return


def handle_alphabet(folder):
    print('{}...'.format(folder.split('/')[-1]))
    n_variations_character = 10
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomAffine(20,
                                            scale=(0.9, 1.1),
                                            shear=20),
        # torchvision.transforms.ToTensor(),
    ])

    for rotate in [0]:
        # Create new folders for each augmented alphabet
        mkdir(str(folder)+'.'+str(rotate))
        for root, character_folders, _ in os.walk(folder):
            for character_folder in character_folders:
                # For each character folder in an alphabet rotate and resize all of the images and save
                # to the new folder
                handle_characters(folder, root + '/' + character_folder,
                                  n_variations=n_variations_character, rotate=rotate)
                # return

    # Delete original alphabet
    rmdir(folder)


# Clean up previous extraction
rmdir(prepared_omniglot_location)
mkdir(prepared_omniglot_location)

# Unzip dataset
for root, _, files in os.walk(raw_omniglot_location):
    for f in files:
        if f in dataset_zip_files:
            print('Unzipping {}...'.format(f))
            zip_ref = zipfile.ZipFile(root + f, 'r')
            zip_ref.extractall(prepared_omniglot_location)
            zip_ref.close()

if not no_preprocessing:
    print('Processing background set...')
    for root, alphabets, _ in os.walk(prepared_omniglot_location + 'images_background/'):
        for alphabet in sorted(alphabets):
            handle_alphabet(root + alphabet)

    print('Processing evaluation set...')
    for root, alphabets, _ in os.walk(prepared_omniglot_location + 'images_evaluation/'):
        for alphabet in sorted(alphabets):
            handle_alphabet(root + alphabet)
