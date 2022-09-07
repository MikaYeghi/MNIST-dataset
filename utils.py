import torch
from torch import optim

def create_loss_fn():
    return torch.nn.CrossEntropyLoss()

def create_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr)

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