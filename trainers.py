import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST 
from torchvision import transforms
from tqdm import tqdm

from fashionmnist_utils import mnist_reader
from metrics import MetricLogger


class Trainer(ABC):
    """Provides a library-independent API for training and evaluating machine learning classifiers."""

    def __init__(self, model):
        """Creates a new model instance with a unique name and underlying model.

        :param model: Model object to be used in training/prediction/evaluation.A
        """
        self.model = model
        self.name = f'{type(model).__name__}-{datetime.now().strftime("%m-%d--%H-%M-%S")}'

    @abstractmethod
    def predict(self, input):
        ...

    @abstractmethod
    def train(self, *args):
        """Completely trains self.model using internal training data
        """
        ...

    @abstractmethod
    def evaluate(self) -> MetricLogger:
        """Evaluate model on the internal testing data.

        :returns: MetricLogger object with results.
        """
        ...

    @abstractmethod
    def save(self):
        """Save the model object in "models". The filename is given by self.name.
        """
        ...

    @staticmethod
    @abstractmethod
    def load(path: str):
        """Load the model object at the specified file location.

        :param path: Path in "models" directory to load from.
        """
        ...


class SKLearnTrainer(Trainer):
    """Implements the Model API for scikit-learn models."""

    def __init__(self, algorithm):
        super().__init__(algorithm)
        X, y = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='train')
        
        # Load and split datasets into training, validation, and test set.
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test, self.y_test = mnist_reader.load_mnist('data/FashionMNIST/raw', kind='t10k')
        self.logger = MetricLogger(one_hot=False)

    def predict(self, input):
        return self.model.predict(input)

    def train(self):
        self.model = self.model.fit(self.X_train, self.y_train)
        prediction = self.model.predict(self.X_val)
        self.logger.log(prediction, self.y_val)
        print("Accuracy: {}\nPrecision: {}\nRecall: {}\n".format(self.logger.accuracy, self.logger.precision, self.logger.recall))
        
    def evaluate(self):
        prediction = self.model.predict(self.X_test)
        self.logger.log(prediction, self.y_test)
        return self.logger

    def save(self):
        with open(os.path.join('models', self.name + '.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load(path: str) -> Trainer:
        new = SKLearnTrainer(None)
        with open(path, 'rb') as file:
            new.model = pickle.load(file)
            new.name = os.path.basename(path).split('.')[0]
            return new


def get_data(transform, train=True):
    return FashionMNIST(os.path.join(os.getcwd(), 'data'), train=train, transform=transform, download=True)


class PyTorchTrainer(Trainer):
    """Implements the Model API for PyTorch (torch) models."""

    def __init__(self, nn_module: nn.Module, transform: Callable, optimizer: torch.optim.Optimizer, batch_size: int):
        """Initialize model.

        :param nn_module: torch.nn.Module to use for the model.
        :param transform: torchvision.transforms.Transform to apply to dataset images.
        :param optimizer: torch.optim.Optimizer
        :param batch_size: Batch size to use for datasets.
        """
        super().__init__(nn_module)

        self.transform = transform
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Load datasets
        self.train_data, self.val_data, self.test_data = None, None, None
        self.init_data()

        # Create logger for TensorBoard
        self.logger = SummaryWriter()

    def init_data(self):
        """Method for loading datasets.
        """
        data = get_data(self.transform, True)
        test_data = get_data(self.transform, False)
        val_len = int(len(data) * 0.2)

        torch.manual_seed(42)
        train_data, val_data = random_split(data, [len(data) - val_len, val_len])

        self.train_data = DataLoader(train_data, self.batch_size)
        self.val_data = DataLoader(val_data, self.batch_size)
        self.test_data = DataLoader(test_data, self.batch_size)

    def predict(self, input):
        return self.model(input).argmax(dim=1)

    def train(self, epochs: int):
        trainlogger = MetricLogger()
        validlogger = MetricLogger()
        loss_fn = nn.CrossEntropyLoss()
        print("Training...")
        for e in range(epochs):
            running_loss = 0.0
            for i, (x,y) in enumerate(self.train_data, 0):
                
                y_pred = self.model(x)
                trainlogger.log(y_pred, y)
                
                loss = loss_fn(y_pred, y)
                running_loss += loss
                if i % 50 == 49:
                    self.logger.add_scalar("Accuracy", trainlogger.accuracy, i)
                    self.logger.add_scalar("Loss", running_loss.item()/50, i)
                    running_loss = 0.0
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for (x,y) in self.val_data:
                out = self.model(x)
                validlogger.log(out, y)
            
            self.logger.add_scalar("Validation Accuracy", validlogger.accuracy, e)
            validlogger.reset()
            trainlogger.reset()

            print(f"\rFinished Epoch {e}", end="")
        print()


    def evaluate(self) -> MetricLogger:
        logger = MetricLogger()
        for (x,y) in self.test_data:
            out = self.model(x)
            logger.log(out, y)
        
        return logger

    def save(self):
        self.train_data, self.val_data, self.test_data = None, None, None
        self.logger = None

        file_name = os.path.join('models', self.name)
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str) -> Trainer:
        with open(path, 'rb') as file:
            new = pickle.load(file)
            new.init_data()
            return new
