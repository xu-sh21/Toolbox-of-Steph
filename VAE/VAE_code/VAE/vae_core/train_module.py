'''Define the training function for training process!!!'''
#######################################################################
import os
import time
from tqdm import tqdm
import torch

from others.utils import shuffle
#######################################################################

# Check the work state.
def check(config):
    if config['settings']['is_train'] is False:
        raise ValueError('[Error]: Not Training Process!!! Please Check!!!')


# Begin
def begin_training(config):
    model_name = config['settings']['model_name']
    print("-------------------------------------")
    print("[INFO] Begin Training!!!")
    print("[INFO] Model Name: " + model_name)


# Make training directory and result directory if not exists.
def prepare_dir(config):
    train_dir = config['settings']['train_dir']
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    result_dir = config['settings']['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    return train_dir, result_dir


def train_single_epoch(device, model, loss_func, optimizer, config, training_data):
    model.train()
    loss = 0.0
    st, ed, times = 0, config['params']['batch_size'], 0
    while st < len(training_data) and ed <= len(training_data):
        optimizer.zero_grad()
        X_batch = torch.from_numpy(training_data[st:ed]).float().to(device)
        X_new, mu, sigma = model(X_batch)

        loss_ = loss_func(X_new, X_batch, mu, sigma)
        loss_.backward()
        optimizer.step()
        loss += loss_.cpu().data.numpy()
        st, ed = ed, ed + config['params']['batch_size']
        times += 1
    loss /= times

    return loss


def valid_single_epoch(device, model, loss_func, config, validation_data):
    model.eval()
    loss = 0.0
    st, ed, times = 0, config['params']['batch_size'], 0
    while st < len(validation_data) and ed <= len(validation_data):
        X_batch = torch.from_numpy(validation_data[st:ed]).float().to(device)
        X_new, mu, sigma = model(X_batch)

        loss_ = loss_func(X_new, X_batch, mu, sigma)
        loss += loss_.cpu().data.numpy()
        st, ed = ed, ed + config['params']['batch_size']
        times += 1
    loss /= times

    return loss


# Save the model if it is the best model till now.
def save_model(model, train_dir, config):
    model_path = train_dir + f"checkpoint_{config['settings']['inference_version']}.pth.tar"
    with open(model_path, 'wb') as fout:
        torch.save(model, fout)


# Print the results.
def output_result(start_time, epoch, epoch_nums, train_loss, val_loss, best_val_loss, best_epoch, test_loss, config):
    epoch_time = time.time() - start_time
    print("-------------------------------------")
    print(f"[INFO] Training Epoch {epoch}/{epoch_nums}:")
    print(f"[INFO] Time {epoch_time}.")
    print(f"[INFO] Learning Rate {config['params']['lr']}")
    print(f"[INFO] Training Loss {train_loss}")
    print(f"[INFO] Validation Loss {val_loss}")
    print(f"[INFO] Best Validation Accuracy {best_val_loss} at Epoch {best_epoch}")
    print(f"[INFO] Testing Loss {test_loss}")

    return best_val_loss, best_epoch


# Save the training result.
def save_data_each_epoch(result_dir, epoch, train_loss, config):
    with open(result_dir + f'/{config["settings"]["model_name"]}_train.txt', "a") as f:
            f.write(f'Epoch {epoch}/{config["params"]["epoch_nums"]}:\n')
            f.write(f'Training Loss: {train_loss:.4f}\n')
            f.write('-' * 50 + '\n')


# Adjust learning rate.
def adjust_lr(train_loss, optimizer, pre_losses):
    if train_loss > max(pre_losses):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9995
    pre_losses = pre_losses[1:] + [train_loss]


# End training.
def end_training(best_val_loss, best_epoch):
    print("-------------------------------------")
    print("[INFO] Training Finished!!!")
    print(f"[INFO] Best Validation Accuracy {best_val_loss} at Epoch {best_epoch}")
    print("-------------------------------------")


# Train and valid cirlces.
def train_job(device, model, loss_func, optimizer, config, training_data, validation_data, test_data):

    # Prepare parameters for training.
    pre_losses = [1e18] * 3
    best_val_loss = 1e18
    epoch_nums = config['params']['epoch_nums']
    train_dir, result_dir = prepare_dir(config)

    for epoch in range(1, epoch_nums):
        start_time = time.time()

        train_loss = train_single_epoch(device, model, loss_func, optimizer, config, training_data)

        # training_data = shuffle(training_data, 1)

        val_loss = valid_single_epoch(device, model, loss_func, config, validation_data)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            test_loss = valid_single_epoch(device, model, loss_func, config, test_data)
            # Save the model if it is the best model till now.
            save_model(model, train_dir, config)

        best_val_loss, best_epoch = output_result(start_time, epoch, epoch_nums, train_loss, val_loss, best_val_loss, best_epoch, test_loss, config)
        save_data_each_epoch(result_dir, epoch, train_loss, config)

        # Ajust learning rate.
        adjust_lr(train_loss, optimizer, pre_losses)
    
    return best_val_loss, best_epoch