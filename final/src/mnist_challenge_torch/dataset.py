from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dataset
import torchvision.transforms as transforms


def get_MNIST_set():
    train_data = dataset.MNIST(root="mnist",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
    test_data = dataset.MNIST(root="mnist",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
    return train_data, test_data


def get_MNIST_loader():
    train_data, test_data = get_MNIST_set()
    train_loader = DataLoader(train_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=128)
    return train_loader, test_loader