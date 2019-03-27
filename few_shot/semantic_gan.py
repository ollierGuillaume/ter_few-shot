import torch
from few_shot.models import *
from torch import nn
from few_shot.callbacks import CallbackList, DefaultCallback, ModelCheckpoint, Callback, ProgressBarLogger
from config import PATH
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union
from torch.autograd import Variable
import os
import torch.nn.functional as F

class SemanticBinaryEncoder(nn.Module):
    def __init__(self, num_input_channels: int, size_binary_layer: int = 10, size_continue_layer: int = 10,
                 stochastic: bool = True):
        super(SemanticBinaryEncoder, self).__init__()
        self.size_continue_layer = size_continue_layer
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.dense_bin = nn.Linear(3136, size_binary_layer)
        self.dense_cont = nn.Linear(3136, size_continue_layer)
        self.slope = 1.0

        if stochastic:
            self.binary_act = StochasticBinaryActivation(estimator='ST')
        else:
            self.binary_act = DeterministicBinaryActivation(estimator='ST')

    def forward(self, x):
        #  return latent space with binary semantic part and continue example variation
        # batch examples have the same class
        # maybe reduce hamming
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x1 = self.dense_bin(x)
        x2 = self.dense_cont(x)

        x1 = self.binary_act([x1, self.slope])
        x2 = F.relu(x2)

        return x1, x2

def normal_init(m, mean, std):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0.)


class SemanticBinaryClassifierLatentSpace(nn.Module):
    def __init__(self, latent_size, k):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, 100)
        self.dlayer = nn.Linear(100, k)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.dlayer(x))
        return x

class SemanticBinaryDecoder(nn.Module):
    def __init__(self, size_binary_layer: int = 10, size_continue_layer: int = 10):
        super(SemanticBinaryDecoder, self).__init__()
        self.latent_size = size_binary_layer + size_continue_layer
        d = 128
        self.deconv1 = nn.ConvTranspose2d(self.latent_size, d * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 4, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, 0)
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, c, v):
        x = torch.cat([c, v], dim=1)
        x = x.view(-1, self.latent_size, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv6(x))
        return x

class SemanticBinaryDiscriminator(nn.Module):
    def __init__(self, num_input_channels: int, final_layer_size: int):
        super(SemanticBinaryDiscriminator, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.fc1 = nn.Linear(final_layer_size, 100)
        self.logits = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.logits(x)

criterionBCE = nn.BCELoss()
criterionClass = nn.CrossEntropyLoss()
criterionReconstr = nn.MSELoss()

def train_discriminator(encoder, generator, discriminator, images, d_optimizer, device, noisy=False, latent_size=0, images_for_fake=None):
    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()

    real_labels = Variable(torch.ones(images.size(0)).to(device, dtype=torch.double))

    outputs = discriminator(images).to(device, dtype=torch.double)
    # print (outputs.shape)
    real_loss = criterionBCE(outputs, real_labels).to(device, dtype=torch.double)
    real_score = outputs

    c, v = encoder(images_for_fake)
    if noisy:
        noisy1 = Variable(torch.randn(images_for_fake.size(0), latent_size).to(device, dtype=torch.double))
        fake_images = generator(c, noisy1)
    else:
        fake_images = generator(c, v)

    fake_labels = Variable(torch.zeros(images_for_fake.size(0)).to(device, dtype=torch.double))

    outputs = discriminator(fake_images).to(device, dtype=torch.double)
    fake_loss = criterionBCE(outputs, fake_labels).to(device, dtype=torch.double)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score


def train_classifier(encoder, classifier, images, class_labels, c_optimizer, e_optimizer):
    classifier.zero_grad()
    encoder.zero_grad()

    z1, z2 = encoder(images)
    outputs = classifier(z1)
    cl_loss = criterionClass(outputs, class_labels)
    cl_score = outputs
    cl_loss.backward()
    c_optimizer.step()
    e_optimizer.step()
    return cl_loss, cl_score


def train_ae(encoder, generator, images, e_optimizer, g_optimizer):
    encoder.zero_grad()
    generator.zero_grad()
    z1, z2 = encoder(images)
    images_reconstr = generator(z1, z2)
    ed_loss = criterionReconstr(images, images_reconstr)
    ed_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return ed_loss


def train_dae(encoder, generator, discriminator, images, device, e_optimizer, g_optimizer):

    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()
    z1, z2 = encoder(images)
    reconstr_images = generator(z1, z2)
    outputs = discriminator(reconstr_images).to(device, dtype=torch.double)
    real_labels = Variable(torch.ones(images.size(0)).to(device, dtype=torch.double))
    dae_loss = criterionBCE(outputs, real_labels).to(device, dtype=torch.double)
    dae_score = outputs
    dae_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return dae_loss


def train_dae_noisy(encoder, generator, discriminator, images, latent_size, device, e_optimizer, g_optimizer):
    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()

    c, v = encoder(images)
    noisy1 = Variable(torch.randn(images.size(0), latent_size).to(device, dtype=torch.double))
    new_images = generator(c, noisy1)

    outputs = discriminator(new_images)
    real_labels = Variable(torch.ones(images.size(0)).to(device, dtype=torch.double))
    dae_loss = criterionBCE(outputs, real_labels).to(device)
    dae_score = outputs
    dae_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return dae_loss


def train_cae_noisy(encoder, generator, classifier, images, class_labels, latent_size, device, e_optimizer, g_optimizer):
    classifier.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()
    c, v = encoder(images)
    noisy1 = Variable(torch.randn(images.size(0), latent_size).to(device, dtype=torch.double))
    new_images = generator(c, noisy1)

    outputs = classifier(c)
    cae_noisy_loss = criterionClass(outputs, class_labels)
    cae_noisy_score = outputs
    cae_noisy_loss.backward()
    e_optimizer.step()
    g_optimizer.step()

    return cae_noisy_loss


images_old = None

def gradient_step_gan_few_shot(e_optimizer: Optimizer, g_optimizer: Optimizer, c_optimizer: Optimizer, d_optimizer: Optimizer,
                               x: torch.Tensor, y: torch.Tensor, device,
                               encoder: Module, generator: Module, classifier: Module, discriminator: Module,
                               latent_sizeC: int, latent_sizeB: int, epoch: int):

    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    global images_old
    max_batch = 500
    cl_losses = []
    cl_scores = []

    # d_losses = []
    # real_scores = []
    # fake_scores = []

    dae_losses = []
    ae_losses = []
    cae_losses = []

    for i in range(0, y.shape[0], max_batch):
        y_sub_batch = y[i:i+max_batch]
        x_sub_batch = x[i:i+max_batch]
        #### Classifier
        images = Variable(x_sub_batch.to(device))
        class_labels = Variable(y_sub_batch.to(device))
        cl_loss, cl_score = train_classifier(encoder, classifier, images, class_labels, c_optimizer, e_optimizer)
        cl_losses.append(cl_loss)
        cl_scores.append(cl_score)

        #### Adversarial discriminator
        if (images_old is not None):  # pas la premiere iteration
            # Train the discriminator
            d_loss, real_score, fake_score = train_discriminator(encoder, generator, discriminator, images,
                                                                 d_optimizer, device,
                                                                 noisy=False, latent_size=latent_sizeC, images_for_fake=images_old)

            d_loss, real_score, fake_score = train_discriminator(encoder, generator, discriminator, images,
                                                                 d_optimizer, device,
                                                                 noisy=True, latent_size=latent_sizeC, images_for_fake=images_old)



        #### Autoencoder reconstruction
        ae_loss = cl_loss
        if (epoch < 5):
            ae_loss = train_ae(encoder, generator, images, e_optimizer, g_optimizer)
        ae_losses.append(ae_loss)

        ### Autoencoder Discriminator
        dae_loss = train_dae_noisy(encoder, generator, discriminator, images, latent_sizeC, device, e_optimizer, g_optimizer)
        dae_losses.append(dae_loss)

        #### Autoencoder noisy classifier
        class_labels = Variable(class_labels.to(device))
        cae_noisy_loss = train_cae_noisy(encoder, generator, classifier, images, class_labels, latent_sizeC, device, e_optimizer, g_optimizer)
        cae_losses.append(cae_noisy_loss)

        images_old = images

    return torch.stack(cl_losses).mean(), torch.stack(cl_scores).mean(), torch.stack(dae_losses).mean(), torch.stack(ae_losses).mean(), torch.stack(cae_losses).mean()



def fit_gan_few_shot(encoder: Module, generator: Module, classifier: Module, discriminator: Module,
        dataloader: DataLoader, params_str: str, k: int, n: int, epochs: int,
        prepare_batch: Callable, latent_sizeC: int, latent_sizeB: int, device,
        e_optimizer: Optimizer, g_optimizer: Optimizer, c_optimizer: Optimizer, d_optimizer: Optimizer,
        metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool = True):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])


    class EncoderClassifier(nn.Module):
        def forward(self, x):
            c, v = encoder(x)
            return classifier(c)

    encoderclassifier = EncoderClassifier()

    callbacks.set_model(encoderclassifier)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': criterionClass
    })

    checkpoint_e = ModelCheckpoint(
        filepath=os.path.join(PATH, 'models', 'semantic_gan', str(params_str) + '_encoder.pth'),
        monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    )

    checkpoint_g = ModelCheckpoint(
        filepath=os.path.join(PATH, 'models', 'semantic_gan', str(params_str) + '_generator.pth'),
        monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    )

    checkpoint_c = ModelCheckpoint(
        filepath=os.path.join(PATH, 'models', 'semantic_gan', str(params_str) + '_classifier.pth'),
        monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    )

    checkpoint_d = ModelCheckpoint(
        filepath=os.path.join(PATH, 'models', 'semantic_gan', str(params_str) + '_discriminator.pth'),
        monitor='val_' + str(n) + '-shot_' + str(k) + '-way_acc'
    )

    checkpoint_e.set_model(encoder)
    checkpoint_g.set_model(generator)
    checkpoint_c.set_model(classifier)
    checkpoint_d.set_model(discriminator)

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()


    for epoch in range(1, epochs + 1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)
            cl_loss, cl_score, dae_loss, ae_loss, cae_loss = gradient_step_gan_few_shot(e_optimizer, g_optimizer, c_optimizer, d_optimizer,
                                                                     x, y, device,
                                                                     encoder, generator, classifier, discriminator,
                                                                     latent_sizeB, latent_sizeC, epoch)
            batch_logs['cl_loss'] = cl_loss.item()
            #batch_logs['d_loss'] = d_loss.item()
            batch_logs['dae_loss'] = dae_loss.item()
            batch_logs['ae_loss'] = ae_loss.item()
            batch_logs['cae_loss'] = cae_loss.item()

            batch_logs['cl_score'] = cl_score.item()
            #batch_logs['real_score'] = real_score.item()
            #batch_logs['fake_score'] = fake_score.item()

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)
        checkpoint_e.on_epoch_end(epoch, epoch_logs)
        checkpoint_c.on_epoch_end(epoch, epoch_logs)
        checkpoint_g.on_epoch_end(epoch, epoch_logs)
        checkpoint_d.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()

