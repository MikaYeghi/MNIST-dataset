import torch
from torch import optim
from matplotlib import pyplot as plt

def create_loss_fn():
    return torch.nn.CrossEntropyLoss()

def create_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr)

def create_scheduler(optimizer, SCHEDULER_GAMMA):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=SCHEDULER_GAMMA
    )
    return scheduler

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, masks_batch):
        model.train()
        yhat = model(images_batch)
        loss = loss_fn(masks_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def evaluate(model, images):
    model.eval()
    with torch.no_grad():
        return model(images)

def preprocess_batch(images):
    def preprocess(image):
        image /= 255.
        return image

    images = images.float()
    for i in range(len(images)):
        images[i] = preprocess(images[i])
    return images

def OH_encode(labels, NUM_CLASSES):
    return torch.nn.functional.one_hot(labels % NUM_CLASSES).float()

def plot_training_info(train_losses, val_losses, LRs):
    plt.figure()
    plt.subplot(311)
    plt.plot(train_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training loss")
    plt.subplot(312)
    plt.plot(LRs, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Learning rate value")
    plt.title("Learning rate")
    plt.subplot(313)
    plt.plot(val_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Validation loss")
    plt.show()