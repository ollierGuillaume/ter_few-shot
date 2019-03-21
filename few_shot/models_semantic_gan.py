class SemanticBinaryAutoEncoder(nn.Module):
    def __init__(self, num_input_channels: int, size_binary_layer: int = 10, size_continue_layer: int = 10,
                 stochastic: bool = True):
        super(SemanticBinaryAutoEncoder, self).__init__()
        self.size_continue_layer = size_continue_layer
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.dense_bin = nn.Linear(3136, size_binary_layer)
        self.dense_cont = nn.Linear(3136, size_continue_layer)

        self.deconv1 = deconv_block(64, 64)
        self.deconv2 = deconv_block(64, 64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3)

        if stochastic:
            self.binary_act = StochasticBinaryActivation(estimator='ST')
        else:
            self.binary_act = DeterministicBinaryActivation(estimator='ST')

        # self.discriconv1 = conv_block(num_input_channels, 64)
        # self.discriconv2 = conv_block(64, 64)
        # self.discriconv3 = conv_block(64, 64)
        # self.discriconv4 = conv_block(64, 64)
        # self.dlogits = nn.Linear(64, 2)

    def encoder(self, x):
        #  return latent space with binary semantic part and continue example variation
        # batch examples have the same class
        # maybe reduce hamming
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.dense_bin(x)
        x2 = self.dense_cont(x)

        x1 = self.binary_act(x1)
        x2 = F.relu(x2)

        return x1, x2

    def decoder(self, latent_space):
        x = self.deconv1(latent_space)
        x = self.deconv2(x)
        x = F.tanh(self.deconv3(x))
        return x

    # def binary_space_decoder(self, binary_latent_space, n_noise):
    #     # construct an example of the same class with a binary_latent_space and a noise vector
    #     n = torch.distributions.Uniform(0, 1)
    #     noise = n.sample((n_noise, self.size_continue_layer))
    #     latent_space = torch.cat((binary_latent_space, noise))
    #     return self.decoder(latent_space)

    def forward(self, x):
        c, v = self.encoder(x)
        x = self.decoder(torch.cat((c, v)))
        return x, c, v

    # def discriminator(self, x1, x2):
    #     x = torch.cat((x1, x2))
    #     x = self.discriconv1(x)
    #     x = self.discriconv2(x)
    #     x = self.discriconv3(x)
    #     x = self.discriconv4(x)
    #
    #     x = x.view(x.size(0), -1)
    #
    #     return self.dlogits(x)


class SemanticBinaryEncoder(nn.Module):
    def __init__(self, num_input_channels: int, size_binary_layer: int = 10, size_continue_layer: int = 10,
                 stochastic: bool = True):
        super(SemanticBinaryAutoEncoder, self).__init__()
        self.size_continue_layer = size_continue_layer
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.dense_bin = nn.Linear(3136, size_binary_layer)
        self.dense_cont = nn.Linear(3136, size_continue_layer)

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

        x1 = self.dense_bin(x)
        x2 = self.dense_cont(x)

        x1 = self.binary_act(x1)
        x2 = F.relu(x2)

        return x1, x2


class SemanticBinaryDecoder(nn.Module):
    def __init__(self, num_input_channels: int, size_binary_layer: int = 10, size_continue_layer: int = 10,
                 stochastic: bool = True):
        self.deconv1 = deconv_block(64, 64)
        self.deconv2 = deconv_block(64, 64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3)
        self.latent_size = size_continue_layer + size_binary_layer

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, c, v):
        x = torch.cat([c, v], dim=1)
        x = x.view(-1, self.latent_size, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = F.tanh(self.deconv3(x))
        return x


def train_discriminator(encoder, generator, discriminator, images, noisy=False, latent_size=0, images_for_fake=None):
    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()

    real_labels = Variable(torch.ones(images.size(0)).to(device))

    outputs = discriminator(images)
    # print (outputs.shape)
    real_loss = criterionBCE(outputs, real_labels)
    real_score = outputs

    c, v = encoder(images_for_fake)
    if noisy:
        noisy1 = Variable(torch.randn(images_for_fake.size(0), latent_size).to(device))
        fake_images = generator(c, noisy1)
    else:
        fake_images = generator(c, v)

    fake_labels = Variable(torch.zeros(images_for_fake.size(0)).to(device))

    outputs = discriminator(fake_images)
    fake_loss = criterionBCE(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score


def train_generator(generator, discriminator_outputs):
    generator.zero_grad()
    Variable(torch.ones(images.size(0)).to(device))
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss


def train_classifier(encoder, classifier, images, class_labels):
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


def train_ae(encoder, generator, images):
    encoder.zero_grad()
    generator.zero_grad()
    z1, z2 = encoder(images)
    images_reconstr = generator(z1, z2)
    ed_loss = criterionReconstr(images, images_reconstr)
    ed_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return ed_loss


def train_dae(encoder, generator, discriminator, images):

    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()
    z1, z2 = encoder(images)
    reconstr_images = generator(z1, z2)
    outputs = discriminator(reconstr_images)
    real_labels = Variable(torch.ones(images.size(0)).to(device))
    dae_loss = criterion(outputs, real_labels)
    dae_score = outputs
    dae_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return dae_loss


def train_dae_noisy(encoder, generator, discriminator, images, latent_size):
    discriminator.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()

    c, v = encoder(images)
    noisy1 = Variable(torch.randn(images.size(0), latent_size).to(device))
    new_images = generator(c, noisy1)

    outputs = discriminator(new_images)
    real_labels = Variable(torch.ones(images.size(0)).to(device))
    dae_loss = criterionBCE(outputs, real_labels)
    dae_score = outputs
    dae_loss.backward()
    e_optimizer.step()
    g_optimizer.step()
    return dae_loss


def train_cae_noisy(encoder, generator, classifier, images, class_labels, latent_size):
    classifier.zero_grad()
    encoder.zero_grad()
    generator.zero_grad()
    c, v = encoder(images)
    noisy1 = Variable(torch.randn(images.size(0), latent_size).to(device))
    new_images = generator(c, noisy1)

    outputs = classifier(c)
    cae_noisy_loss = criterionClass(outputs, class_labels)
    cae_noisy_score = outputs
    cae_noisy_loss.backward()
    e_optimizer.step()
    g_optimizer.step()

    return cae_noisy_loss


def gradient_step_gan_few_shot(optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor,
                               encoder: Module, generator: Module, classifier: Module, discriminator: Module,
                               latent_sizeC: int, latent_sizeB: int, epoch: int **kwargs):

    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    max_batch = 500
    losses = []
    predictions = []
    for i in range(0, y.shape[0], max_batch):
        y_sub_batch = y[i:i+max_batch]
        x_sub_batch = x[i:i+max_batch]
        #### Classifier
        images = Variable(x_sub_batch.to(device))
        class_labels = Variable(y_sub_batch.to(device))
        cl_loss, cl_score = train_classifier(encoder, classifier, images, class_labels)

        #### Adversarial discriminator
        if (images_old is not None):  # pas la premiere iteration
            # Train the discriminator
            d_loss, real_score, fake_score = train_discriminator(encoder, generator, discriminator, images, noisy=False,
                                                                 latent_size=latent_sizeC, images_for_fake=images_old)
            d_loss, real_score, fake_score = train_discriminator(encoder, generator, discriminator, images, noisy=True,
                                                                 latent_size=latent_sizeC, images_for_fake=images_old)

        #### Autoencoder reconstruction
        ae_loss = cl_loss
        if (epoch < 5):
            ae_loss = train_ae(encoder, generator, images)

        ### Autoencoder Discriminator
        dae_loss = train_dae_noisy(encoder, generator, discriminator, images, latent_sizeC)

        #### Autoencoder noisy classifier
        class_labels = Variable(class_labels.to(device))
        cae_noisy_loss = train_cae_noisy(encoder, generator, classifier, images, class_labels, latent_sizeC)


def fit_gan(encoder: Module, generator: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        latent_sizeC: int, latent_sizeB: int
        verbose: bool = True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

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

            loss, y_pred = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()

