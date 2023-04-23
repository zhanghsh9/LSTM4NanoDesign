import torch
import torch.nn.functional as F

from datetime import datetime
import time
import os
import random

from parameters import EPOCHS, VALID_FREQ, RESULTS_PATH, MODEL_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch_forward(training_loader, optimizer, model, loss_fn):
    """

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
        running_loss += loss.item()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

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


def train_epochs_forward(training_loader, test_loader, model, loss_fn, optimizer, scheduler, attention, epochs=EPOCHS):
    """
    Train transformer for epochs
    :param attention: double
    :param scheduler: torch.optim.lr_scheduler.StepLR
    :param epochs: int
    :param test_loader: torch.utils.data.DataLoader
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
    timestamp = datetime.now().strftime('%Y%m%d')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)

    # Train
    for epoch in range(epochs):
        print('{}: Forward EPOCH {}:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), epoch + 1))
        model.train(True)
        avg_loss = train_one_epoch_forward(training_loader=training_loader, model=model, loss_fn=loss_fn,
                                           optimizer=optimizer)
        scheduler.step()

        # See https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/5
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Eval
        if epoch % VALID_FREQ == 0 or epoch + 1 == epochs:

            # Evaluation
            model.eval()
            running_vloss = 0.0
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.float().to(device), vlabels.float().to(device)

                voutputs, _ = model(vinputs)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('{}: LOSS train: {}, valid: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                                         avg_loss, avg_vloss))
            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)

            if epoch == 0:
                best_vloss = avg_vloss

            elif avg_vloss < best_vloss:
                # Save model
                model_name = 'Forward_mse_vloss_best_attn_{}.pth'.format(attention)
                if os.path.exists(os.path.join(model_save_path, model_name)):
                    os.remove(os.path.join(model_save_path, model_name))
                torch.save(model, os.path.join(model_save_path, model_name))
                best_vloss = avg_vloss

            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)
        else:
            print('{}: Training Loss:{}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), avg_loss))

        print()
        loss_record.append(float(avg_loss))
        x_axis_loss.append(epoch + 1)
    return model, x_axis_loss, x_axis_vloss, loss_record, vloss_record


def train_one_epoch_backward(training_loader, optimizer, backward_model, forward_model, loss_fn):
    """

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

        outputs, _ = backward_model(inputs)
        outputs_forward, _ = forward_model(outputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs_forward, inputs)
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


def train_epochs_backward(training_loader, test_loader, backward_model, loss_fn, optimizer, scheduler, attention,
                          epochs=EPOCHS):
    """
    Train backward model for epochs
    :param backward_model: torch.nn.Module
    :param attention: double
    :param scheduler: torch.optim.lr_scheduler.StepLR
    :param epochs: int
    :param test_loader: torch.utils.data.DataLoader
    :param training_loader: torch.utils.data.DataLoader
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
    timestamp = datetime.now().strftime('%Y%m%d')
    model_save_path = os.path.join(RESULTS_PATH, timestamp, MODEL_PATH)

    # Load forward model
    forward_model_name = 'Forward_mse_vloss_best_attn_{}.pth'.format(attention)
    forward_model = torch.load(os.path.join(model_save_path, forward_model_name))
    forward_model.train(True)
    forward_model.to(device)
    # Freeze parameters
    for paras in forward_model.parameters():
        paras.requires_grad = False

    # Train
    for epoch in range(epochs):
        print('{}: Forward EPOCH {}:'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), epoch + 1))
        backward_model.train(True)
        avg_loss = train_one_epoch_backward(training_loader=training_loader, backward_model=backward_model,
                                            forward_model=forward_model, loss_fn=loss_fn, optimizer=optimizer)
        scheduler.step()

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

                voutputs, _ = backward_model(vinputs)
                voutputs_forward, _ = backward_model(voutputs)

                vloss = loss_fn(voutputs_forward, vinputs).item()
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('{}: LOSS train: {}, valid: {}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()),
                                                         avg_loss, avg_vloss))
            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)

            if epoch == 0:
                best_vloss = avg_vloss

            elif avg_vloss < best_vloss:
                # Save model
                model_name = 'Backward_mse_vloss_best_attn_{}.pth'.format(attention)
                if os.path.exists(os.path.join(model_save_path, model_name)):
                    os.remove(os.path.join(model_save_path, model_name))
                torch.save(backward_model, os.path.join(model_save_path, model_name))
                best_vloss = avg_vloss

            vloss_record.append(float(avg_vloss))
            x_axis_vloss.append(epoch + 1)
        else:
            print('{}: Training Loss:{}'.format(time.strftime("%Y%m%d  %H:%M:%S", time.localtime()), avg_loss))

        print()
        loss_record.append(float(avg_loss))
        x_axis_loss.append(epoch + 1)
    return backward_model, x_axis_loss, x_axis_vloss, loss_record, vloss_record
