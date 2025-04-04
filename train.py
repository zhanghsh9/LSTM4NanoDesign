import torch
import torch.nn.functional as F
import torch.nn as nn

from datetime import datetime
# import matplotlib.pyplot as plt
import time
import os
import random

from parameters import EPOCHS, VALID_FREQ, RESULTS_PATH, MODEL_PATH


def train_one_epoch_forward(training_loader, optimizer, model, loss_fn=nn.MSELoss(), device=torch.device('cuda')):
    """

    :param device:
    :param training_loader: DataLoader
    :param optimizer: torch.optim.Adam
    :param model: torch.nn.Module
    :param loss_fn: torch.nn.MSELoss
    :return: loss
    """
    start_time = time.time()

    running_loss = 0.
    last_loss = 0.
    epoch_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.float().to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        epoch_loss += loss.item()
        optimizer.step()

        # Print loss info per 64 batch
        if i % 64 == 0:
            if i == 0:
                last_loss = running_loss

            else:
                last_loss = running_loss / 64

            running_loss = 0.
            print('{}: batch {} loss: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), i + 1, last_loss))

    end_time = time.time()
    print('{}: Time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                     time.strftime('%H h %M m %S s ', time.gmtime(end_time - start_time))))

    return epoch_loss / (i + 1)


def train_epochs_forward(training_loader, valid_loader, model, loss_fn, optimizer, scheduler,
                         attention, timestamp, epochs=EPOCHS, start_epoch=0, results_path=RESULTS_PATH,
                         device=torch.device('cuda')):
    """
    Train transformer for epochs
    :param device:
    :param results_path:
    :param start_epoch: int
    :param timestamp: str
    :param attention: double
    :param scheduler: torch.optim.lr_scheduler.StepLR
    :param epochs: int
    :param valid_loader: torch.utils.data.DataLoader
    :param training_loader: torch.utils.data.DataLoader
    :param model: torch.nn.Module
    :param loss_fn: torch.nn.CrossEntropyLoss
    :param optimizer: torch.optim.Adam
    :return: model and loss
    """

    # Loss record
    vloss_record = []
    loss_record = []
    x_axis_loss = []
    x_axis_vloss = []

    # Save model path
    model_save_path = os.path.join(results_path, timestamp, MODEL_PATH)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    '''
    # Set up interactive mode for matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    train_line, = ax.plot([], [], label='Training Loss')
    val_line, = ax.plot([], [], label='Validation Loss')
    plt.legend()
    '''

    # Train
    for epoch in range(start_epoch, epochs):
        print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Forward EPOCH {epoch + 1}:')
        model.train(True)
        avg_loss = train_one_epoch_forward(training_loader=training_loader, model=model, loss_fn=loss_fn,
                                           optimizer=optimizer, device=device)
        scheduler.step()
        print(
            f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Epoch: {epoch + 1}  Learning Rate: {scheduler.get_last_lr()}')

        # See https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/5
        # if device == 'cuda':
        # torch.cuda.empty_cache()

        # Eval
        if epoch % VALID_FREQ == 0 or epoch + 1 == epochs:

            # Evaluation
            model.eval()
            running_vloss = 0.0
            for i, vdata in enumerate(valid_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)

                voutputs, _ = model(vinputs)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: LOSS train: {avg_loss}, valid: {avg_vloss}')
            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)

            if epoch == start_epoch:
                best_vloss = avg_vloss
                best_vloss_epoch=epoch+1

            elif avg_vloss < best_vloss:
                # Save model
                model_name = 'Forward_mse_vloss_best_attn_{}.pth'.format(attention)
                if os.path.exists(os.path.join(model_save_path, model_name)):
                    os.remove(os.path.join(model_save_path, model_name))
                torch.save(model, os.path.join(model_save_path, model_name))
                best_vloss = avg_vloss
                best_vloss_epoch = epoch+1

        else:
            print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Training Loss:{avg_loss}, best vloss: {best_vloss}')

        # Save checkpoint
        model_checkpoint_path = os.path.join(model_save_path, 'forward_checkpoint.pth')
        torch.save(model, model_checkpoint_path)

        checkpoint = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_loss,
            'loss_fn': loss_fn,
            'attention': attention,
            'timestamp': timestamp,
            'vloss_record': vloss_record,
            'x_axis_vloss': x_axis_vloss
        }
        checkpoint_path = os.path.join(model_save_path, 'forward_states.pth')
        torch.save(checkpoint, checkpoint_path)

        print()
        loss_record.append(float(avg_loss))
        x_axis_loss.append(epoch + 1)

        '''
        # Update the plot
        train_line.set_xdata(x_axis_loss)
        train_line.set_ydata(loss_record)
        val_line.set_xdata(x_axis_vloss)
        val_line.set_ydata(vloss_record)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close()
    '''
    return model, x_axis_loss, x_axis_vloss, loss_record, vloss_record, best_vloss_epoch


def train_one_epoch_backward(training_loader, optimizer, backward_model, forward_model, loss_fn=nn.MSELoss(),
                             device=torch.device('cuda')):
    """

    :param device:
    :param training_loader:
    :param optimizer:
    :param backward_model:
    :param forward_model:
    :param loss_fn:
    :return:
    """

    start_time = time.time()
    running_loss = 0.
    last_loss = 0.
    epoch_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.float().to(device)

        outputs, _ = backward_model(labels)
        outputs_forward, _ = forward_model(outputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs_forward, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 64 == 0:
            if i == 0:
                last_loss = running_loss
            else:
                last_loss = running_loss / 64

            running_loss = 0.
            print('{}: batch {} loss: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), i + 1, last_loss))

    end_time = time.time()
    print('{}: Time used: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                     time.strftime('%H h %M m %S s ', time.gmtime(end_time - start_time))))

    return epoch_loss / (i + 1)


def train_epochs_backward(training_loader, test_loader, forward_model, backward_model, loss_fn, optimizer, scheduler,
                          timestamp, epochs=EPOCHS, start_epoch=0, results_path=RESULTS_PATH,
                          device=torch.device('cuda')):
    """
    Train backward model for epochs
    :param start_epoch:
    :param forward_model:
    :param device:
    :param results_path:
    :param timestamp: str
    :param backward_model: torch.nn.Module
    :param scheduler: torch.optim.lr_scheduler.StepLR
    :param epochs: int
    :param test_loader: torch.utils.data.DataLoader
    :param training_loader: torch.utils.data.DataLoader
    :param loss_fn: torch.nn.CrossEntropyLoss
    :param optimizer: torch.optim.Adam
    :return: model and loss
    """
    torch.autograd.set_detect_anomaly(True)
    # Loss record
    vloss_record = []
    loss_record = []
    x_axis_loss = []
    x_axis_vloss = []

    # Save model path
    model_save_path = os.path.join(results_path, timestamp, MODEL_PATH)
    forward_model.train(True)

    # Freeze parameters
    for paras in forward_model.parameters():
        paras.requires_grad = False

    # Train
    best_vloss = 1e5
    for epoch in range(start_epoch, epochs):
        print('{}: Backward EPOCH {}:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), epoch + 1))
        backward_model.train(True)
        avg_loss = train_one_epoch_backward(training_loader=training_loader, backward_model=backward_model,
                                            forward_model=forward_model, loss_fn=loss_fn, optimizer=optimizer,
                                            device=device)
        scheduler.step()
        print(
            f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Epoch: {epoch + 1}  Learning Rate: {scheduler.get_last_lr()}')

        # See https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/5
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Eval
        if epoch % VALID_FREQ == 0 or epoch + 1 == epochs:

            # Evaluation
            backward_model.eval()
            running_vloss = 0.0
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)

                voutputs, _ = backward_model(vlabels)
                voutputs_forward, _ = forward_model(voutputs)

                vloss = loss_fn(voutputs_forward, vlabels).item()
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('{}: LOSS train: {}, valid: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                                         avg_loss, avg_vloss))
            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)

            if epoch == start_epoch:
                best_vloss = avg_vloss

            elif avg_vloss < best_vloss:
                # Save model
                model_name = 'Backward_mse_vloss_best.pth'
                if os.path.exists(os.path.join(model_save_path, model_name)):
                    os.remove(os.path.join(model_save_path, model_name))
                torch.save(backward_model, os.path.join(model_save_path, model_name))
                best_vloss = avg_vloss

        else:
            print(f'{time.strftime("%Y%m%d  %H:%M:%S", time.localtime())}: Training Loss:{avg_loss}, best vloss: {best_vloss}')

            # Save checkpoint
            model_checkpoint_path = os.path.join(model_save_path, 'backward_checkpoint.pth')
            torch.save(backward_model, model_checkpoint_path)

            checkpoint = {
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'loss_fn': loss_fn,
                'timestamp': timestamp,
                'vloss_record': vloss_record,
                'x_axis_vloss': x_axis_vloss
            }
            checkpoint_path = os.path.join(model_save_path, 'backward_states.pth')
            torch.save(checkpoint, checkpoint_path)

        print()
        loss_record.append(float(avg_loss))
        x_axis_loss.append(epoch + 1)
    return backward_model, x_axis_loss, x_axis_vloss, loss_record, vloss_record


def load_checkpoint(checkpoint_path, forward):
    """
    Load the model and optimizer state from a checkpoint file.

    :param checkpoint_path: str, path to the checkpoint file
    :param forward: Whether is loading forward model (True) or backward model (False)
    :return: loaded model and states
    """
    if forward:
        model = torch.load(os.path.join(checkpoint_path, 'forward_checkpoint.pth'))
        checkpoint = torch.load(os.path.join(checkpoint_path, 'forward_states.pth'))
    else:
        model = torch.load(os.path.join(checkpoint_path, 'backward_checkpoint.pth'))
        checkpoint = torch.load(os.path.join(checkpoint_path, 'backward_states.pth'))

    optimizer = torch.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['avg_loss']
    loss_fn = checkpoint['loss_fn']
    attention = checkpoint['attention']
    timestamp = checkpoint['timestamp']
    vloss_record = checkpoint['vloss_record']
    x_axis_vloss = checkpoint['x_axis_vloss']

    return model, optimizer, epoch, loss, loss_fn, attention, timestamp, vloss_record, x_axis_vloss


def custom_loss(forward_output, backward_output, min_vals, max_vals, alpha=1.0):
    # Transform the output of the backward network to match the valid range
    max_vals=[170/100.99, 170/100.99, 300/176.06, (300-180)/70.71, ]
    transformed_backward_output = constrained_transform(backward_output, min_vals, max_vals)

    # Compute the loss as the MSE between the transformed output and forward output
    mse_loss = torch.mean((forward_output - transformed_backward_output) ** 2)

    # Additional penalty for the backward output being out of the valid range can be applied if needed
    penalty_loss = torch.mean(range_penalty(transformed_backward_output, min_vals, max_vals, alpha))

    total_loss = mse_loss + penalty_loss
    return total_loss