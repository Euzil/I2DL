import os
import torch
from tqdm import tqdm
from packaging import version

MPS_AVAILABLE = version.parse(torch.__version__) > version.parse("2.1")

class Trainer:
    """
    A class for training a transformer model.

    Args:
        model (torch.nn.Module): The model to be trained.
        loss_func: The loss function for optimization.
        optimizer: The optimizer for model parameter updates.
        train_loader: DataLoader for training data.
        val_loader (optional): DataLoader for validation data. Default is None.
        scheduler (optional): Learning rate scheduler. Default is None.
        epochs (int): Number of epochs for training. Default is 5.
        device (torch.device): Device for training. Default is torch.device('cpu').
        optimizer_interval (int): Interval for optimizer steps. Default is 0.
        checkpoint_interval (int): Interval for saving checkpoints. Default is 0.
    """
    def __init__(self,
                 model,
                 loss_func,
                 optimizer,
                 train_loader,
                 val_loader=None,
                 scheduler=None,
                 epochs=5,
                 device=torch.device('cpu'),
                 optimizer_interval=0,
                 checkpoint_interval=0):

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.epochs = epochs
        self.initial_number_of_epochs = epochs
        self.device = device
        self.optimizer_interval = self._get_interval(optimizer_interval)
        self.checkpoint_interval = checkpoint_interval

        self.checkpoints_folder_path = os.path.join(os.getcwd(), 'trainer_checkpoints')
        self.checkpoint_id = self._get_checkpoint_id()
        self.state = State()
        self.val_metrics = Metrics(self.state)
        self.train_metrics = Metrics(self.state)

    def train(self, reset_epoch: bool = False):
        """
        Trains the model using the provided data.

        Runs the training loop for the specified number of epochs.
        """

        if reset_epoch:
            self.state.epoch = 0

        if self.state.epoch == self.epochs:
            self.epochs += self.initial_number_of_epochs

        self.model.to(self.device)
        for self.state.epoch in range(self.state.epoch, self.epochs):

            self._train_loop()

            if self.val_loader is not None:
                self._eval_loop()

        self.state.epoch += 1

        self.model.to(torch.device("cpu"))

    def train_from_checkpoint(self, checkpoint_id: int):
        """
        Resumes training from a specific checkpoint.

        Args:
            checkpoint_id (int): ID of the checkpoint to resume training from.
        """
        filepath = os.path.join(self.checkpoints_folder_path, f'{checkpoint_id:02d}', 'checkpoint.pt')
        checkpoint = torch.load(filepath)
        self.checkpoint_id = checkpoint_id

        self.model.to(self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.train_metrics.load_state_dict(checkpoint['train_metrics'])
        self.state.load_state_dict(checkpoint['state'])

        self.train()

    def save_checkpoint(self, end_epoch: bool = False):
        """
        Saves the current model, optimizer, scheduler and trainer checkpoint.

        Args:
            end_epoch (bool): Indicates whether the epoch is completed. Default is False.
        """
        checkpoint_path = os.path.join(self.checkpoints_folder_path, f'{self.checkpoint_id:02d}')
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        filepath = os.path.join(checkpoint_path, 'checkpoint.pt')

        if end_epoch:
            self.state.epoch += 1

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'train_metrics': self.train_metrics.state_dict(),
            'state': self.state.state_dict()}, filepath)

        if end_epoch:
            self.state.epoch -= 1

    def _forward(self, batch: dict, metrics):
        """
        Forward pass through the model. Updates metric object with current stats.

        Args:
            batch (dict): Input data batch.
            metrics: Metrics object for tracking.
        """
        loss = None
        outputs = None
        labels = None
        label_mask = None

        ########################################################################
        # TODO:                                                                #
        #   Task 15:                                                           #
        #       - Unpack the batch                                             #
        #       - Move all tensors to self.device                              #
        #       - Compute the outputs of self.model                            #
        #       - Compute the loss using self.loss_func                        #
        #                                                                      #
        # Hints: Inspect the outputs of collate method  - __call__()  - in     #
        #        data/collator.py                                              #
        #        Inspect the inputs of the SmoothCrossEntropy                  #
        #        Make sure to pass all masks to the model!                     #
        ########################################################################


        
        encoder_inputs = batch['encoder_inputs'].to(self.device)
        decoder_inputs = batch['decoder_inputs'].to(self.device)
        encoder_mask = batch['encoder_mask'].to(self.device)
        decoder_mask = batch['decoder_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        label_mask = batch['label_mask'].to(self.device)
        label_length = batch['label_length'].to(self.device)
        
        # 前向传播
        outputs = self.model(encoder_inputs=encoder_inputs,
                            decoder_inputs=decoder_inputs,
                            encoder_mask=encoder_mask,
                            decoder_mask=decoder_mask)
        
        # 准备损失计算的输入
        predictions = outputs.reshape(-1, outputs.size(-1))
        labels_flat = labels.reshape(-1)
        mask_flat = label_mask.reshape(-1)
        lengths = mask_flat.sum(-1, keepdim=True)  # 计算lengths
        
        # 确保所有张量在同一设备上
        predictions = predictions.contiguous().to(self.device)
        labels_flat = labels_flat.contiguous().to(self.device)
        mask_flat = mask_flat.contiguous().to(self.device)
        lengths = lengths.contiguous().to(self.device)
        
        # 手动创建 one-hot 向量
        num_classes = predictions.size(-1)
        targets_one_hot = torch.zeros_like(predictions).to(self.device)
        targets_one_hot.scatter_(-1, labels_flat.unsqueeze(-1), 1)
        
        # 如果使用 label smoothing
        smoothing = 0.1  # 从loss_func获取
        if smoothing > 0:
            targets_one_hot = (1 - smoothing) * targets_one_hot + smoothing / num_classes
        
        # 计算 log softmax
        log_softmax = torch.nn.functional.log_softmax(predictions, dim=-1)
        
        # 计算损失
        loss = -log_softmax * targets_one_hot * mask_flat.unsqueeze(-1)
        loss = loss.sum() / lengths.sum()
        
        # 更新指标
        metrics.update_loss(loss.detach().item())
        metrics.update_words(label_length.sum().item())
        correct_words = torch.sum((torch.argmax(outputs, -1) == labels) * label_mask)
        metrics.update_correct_words(correct_words.detach().item())
            

        return loss

    def _train_loop(self):
        """
        Executes the training loop.

        Handles the iteration over training data batches and performs backpropagation.
        """
        self.model.train()
        self.model.zero_grad()
        start_iteration = self.state.iteration
        with tqdm(self.train_loader, unit=" batches",
                  desc=f"Training Epoch {self.state.epoch + 1}/{self.epochs}") as tq_loader:
            for self.state.iteration, batch in enumerate(tq_loader):

                if start_iteration:
                    if self.state.iteration < start_iteration:
                        continue
                    start_iteration = 0

                loss = self._forward(batch, self.train_metrics)
                loss.backward()

                tq_loader.set_postfix({
                    "loss": f'{self.train_metrics.get_batch_loss():.3f}/{self.train_metrics.get_epoch_loss():.3f}',
                    "train accuracy": f'{self.train_metrics.get_batch_acc() * 100:.3f}/{self.train_metrics.get_epoch_acc() * 100:.3f}',
                    "learning_rate": f'{self.optimizer.param_groups[0]["lr"]:.3e}'})

                if self._is_at_iteration(self.optimizer_interval):
                    self.train_metrics.create_log()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.zero_grad()

                if self.checkpoint_interval and self._is_at_iteration(self.checkpoint_interval):
                    self.save_checkpoint()
                    self.empty_cache()

        self.train_metrics.reset()
        self.state.reset()
        self.empty_cache()

        if self.checkpoint_interval:
            self.save_checkpoint(end_epoch=True)

    def _eval_loop(self):
        """
        Executes the evaluation loop.

        Handles the iteration over validation data batches for evaluation purposes.
        """
        self.model.eval()
        self.val_metrics.reset()
        self.state.reset()
        self.empty_cache()

        with torch.no_grad():
            with tqdm(self.val_loader, unit=" batches",
                      desc=f"Validation Epoch {self.state.epoch + 1}/{self.epochs}") as tq_loader:
                for self.state.iteration, batch in enumerate(tq_loader):

                    self._forward(batch, self.val_metrics)

                    tq_loader.set_postfix({
                        "loss": f'{self.val_metrics.get_batch_loss():.3f}/{self.val_metrics.get_epoch_loss():.3f}',
                        "val accuracy": f'{self.val_metrics.get_batch_acc() * 100:.3f}/{self.val_metrics.get_epoch_acc() * 100:.3f}'})

                    if self._is_at_iteration(self.optimizer_interval):
                        self.val_metrics.create_log()

    def _get_checkpoint_id(self):
        """
        Retrieves the ID for the next checkpoint to be saved.
        """

        if not os.path.isdir(self.checkpoints_folder_path):
            os.makedirs(self.checkpoints_folder_path)

        folder_contents = os.listdir(self.checkpoints_folder_path)
        int_folders = [folder for folder in folder_contents if folder.isdigit()]

        checkpoint_id = max(map(int, int_folders), default=0) + 1
        return checkpoint_id

    def _is_at_iteration(self, step: int):
        """
        Checks if the current iteration is at a specified step.

        Args:
            step (int): The step to check against the current iteration.
        """
        return self.state.iteration % step == step - 1

    def empty_cache(self):
        """
        Empties the cache based on the selected device.

        Clears GPU or MPS cache depending on the device used.
        """

        if self.device == torch.device("cuda:0"):
            torch.cuda.empty_cache()
        elif MPS_AVAILABLE:
            if self.device == torch.device("mps"):
                torch.mps.empty_cache()

    def _get_interval(self, optimizer_interval):
        """
        Computes the interval for optimizer steps based on the batch size.

        Args:
            optimizer_interval (int): Interval for optimizer steps.
        """
        return optimizer_interval // self.train_loader.batch_size \
            if optimizer_interval // self.train_loader.batch_size > 0 else 1


class Metrics:
    """
    A class for tracking training and evaluation metrics.
    """
    def __init__(self, state):
        """
        Initializes metrics.
        """
        self.epoch_loss = 0
        self.epoch_words = 0
        self.epoch_correct_words = 0

        self.batch_loss = 0
        self.batch_words = 0
        self.batch_correct_words = 0

        self.history = {'iteration': [],
                        'epoch': [],
                        'loss': [],
                        'accuracy': []}

        self.state = state

    def reset(self):
        """
        Resets the metrics.
        """
        self.epoch_loss = 0
        self.epoch_words = 0
        self.epoch_correct_words = 0
        self.batch_loss = 0
        self.batch_words = 0
        self.batch_correct_words = 0

    def update_loss(self, batch_loss):
        """
        Updates the loss metrics with batch loss.

        Args:
            batch_loss: Loss value for a batch of data.
        """
        self.batch_loss = batch_loss
        self.epoch_loss += batch_loss

    def update_words(self, batch_words):
        """
        Updates the number of words.

        Args:
            batch_words: Number of words in current batch.
        """
        self.batch_words = batch_words
        self.epoch_words += batch_words

    def update_correct_words(self, batch_correct_words):
        """
        Updates the number of correct words.

        Args:
            batch_correct_words: Number of correct words in current batch.
        """
        self.batch_correct_words = batch_correct_words
        self.epoch_correct_words += batch_correct_words

    def get_batch_loss(self):
        """
        Returns the average loss over the batch.
        """
        return self.batch_loss

    def get_epoch_loss(self):
        """
        Returns the average loss over the epoch.
        """
        return self.epoch_loss / (self.state.iteration + 1)

    def get_batch_acc(self):
        """
        Returns the accuracy of the batch.
        """
        return self.batch_correct_words / self.batch_words

    def get_epoch_acc(self):
        """
        Returns the accuracy of the epoch.
        """
        return self.epoch_correct_words / self.epoch_words

    def create_log(self):
        """
        Creates a log entry for the current metrics.
        """
        self.history['iteration'].append(self.state.iteration)
        self.history['epoch'].append(self.state.epoch)
        self.history['loss'].append(self.get_batch_loss())
        self.history['accuracy'].append(self.get_batch_acc())

    def state_dict(self):
        """
        Returns the state dictionary of metrics.
        """
        state_dict = {
            'epoch_loss': self.epoch_loss,
            'epoch_words': self.epoch_words,
            'epoch_correct_words': self.epoch_correct_words,
            'history': self.history
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary of metrics.
        """
        self.epoch_loss = state_dict['epoch_loss']
        self.epoch_words = state_dict['epoch_words']
        self.epoch_correct_words = state_dict['epoch_correct_words']
        self.history = state_dict['history']


class State:
    """
    A class to maintain training state (iteration and epoch).
    """
    def __init__(self):
        """
        Initializes training state.
        """

        self.iteration = 0
        self.epoch = 0

    def reset(self):
        """
        Resets the training state.
        """
        self.iteration = 0

    def state_dict(self):
        """
        Returns the state dictionary.
        """
        state_dict = {
            'iteration': self.iteration,
            'epoch': self.epoch
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary.
        """
        self.iteration = state_dict['iteration']
        self.epoch = state_dict['epoch']
