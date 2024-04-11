import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
author: Dong
Instructions:
    Data Requirements:
        Input images should be .jpg images. They do not need to have a fixed size and will be automatically resized before training.
        Grayscale images will be automatically converted to RGB images for training, so no manual modification is required.
        If the input images have a suffix other than .jpg, you need to batch convert them to .jpg before starting training.
        Labels should be .png images. They do not need to have a fixed size and will be automatically resized before training.
    Note: If the input images in the dataset fall into two categories, where the background pixel value is 0 and the target pixel value is 255, the dataset can run normally, but predictions will not be effective! It needs to be changed so that the background pixel value is 0 and the target pixel value is 1.
    
    Loss value:
        The magnitude of the loss value is used to determine convergence. It is important to observe a trend of convergence, i.e., the validation loss continuously decreases. If the validation loss remains basically unchanged, the model has essentially converged.
        The specific magnitude of the loss value does not have much meaning; whether it is large or small depends on the loss calculation method, not on approaching zero. If you want the loss to look better, you can directly divide it by 10000 in the corresponding loss function.
        The loss values during training will be saved in the "logs" folder under the "loss" folder.

    The trained weight files are saved in the "logs" folder. Each training epoch contains several training steps, and each training step performs one gradient descent step.
'''

if __name__ == "__main__":

    Cuda = True
    # -----------------------------------------------------#
    # Seed is used to fix the random seed, ensuring that each independent training session produces the same results every time.
    # -----------------------------------------------------#
    seed            = 11
    distributed     = True
    sync_bn         = False
    # -----------------------------------------------------#
    # fp16    Whether to use mixed precision training. It can reduce memory usage by approximately half but requires PyTorch 1.7.1 or above.
    # -----------------------------------------------------#
    fp16            = False
    # -----------------------------------------------------#
    #   num_classes     number of classes +1
    # -----------------------------------------------------#
    num_classes = 3
    # -----------------------------------------------------#
    #   Backbone network : vgg, resnet50
    # -----------------------------------------------------#
    backbone    = "vgg"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      Whether to use the pre-trained weights of the backbone network. The weights of the backbone are used here, so they are loaded when the model is built.
    #                   If the model path is set, the weight of the backbone does not need to be loaded, and the pretrained value is meaningless.
    #                   If model_path is not set, pretrained = True, only the backbone is loaded to start training.
    #                   If model_path is not set, pretrained = False, and Freeze_Train = Fasle, training starts from 0 and there is no process of freezing the trunk.
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    # The pre-trained weights of the model are common to different data sets because the features are common.
    # The more important part of the model's pre-training weights is the weight part of the backbone feature extraction network, which is used for feature extraction.
    # Pre-training weights must be used in 99% of cases. If not used, the weights of the backbone part will be too random, the feature extraction effect will not be obvious, and the results of network training will not be good.
    # When training your own data set, it prompts that the dimension mismatch is normal, and the predicted things are different. The natural dimension does not match.
    #
    # If there is an operation to interrupt the training during the training process, you can set the model_path to the weight file in the logs folder and load the partially trained weights again.
    # At the same time, modify the parameters of the freezing phase or thawing phase below to ensure the continuity of the model epoch.
    #
    # When model_path = '', the weights of the entire model are not loaded.
    #
    # The weights of the entire model are used here, so they are loaded in train.py. Pretrain does not affect the weight loading here.
    # If you want the model to start training from the pre-training weights of the backbone, set model_path = '', pretrain = True, and only the backbone will be loaded at this time.
    # If you want the model to start training from 0, set model_path = '', pretrain = Fasle, Freeze_Train = Fasle. At this time, training will start from 0, and there is no process of freezing the trunk.
    #
    # Generally speaking, the training effect of the network starting from 0 will be very poor, because the weights are too random and the feature extraction effect is not obvious. Therefore, it is very, very, very not recommended that you start training from 0!
    # If you must start from 0, you can learn about the imagenet data set. First, train the classification model to obtain the weight of the backbone part of the network. The backbone part of the classification model is common to the model, and training is performed based on this.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = "logs/white_white/last_run/best_epoch_weights.pth"
    # -----------------------------------------------------#
    #   input_shape     Enter the size of the image, must be a multiple of 32
    # -----------------------------------------------------#
    input_shape = [1152, 768]

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Training is divided into two phases, the freezing phase and the defrosting phase. The freezing stage is set up to meet the training needs of students with insufficient machine performance.
    # Freeze training requires less video memory and the graphics card is very poor. You can set Freeze_Epoch equal to UnFreeze_Epoch. At this time, only freeze training is performed.
    #
    # Here are some parameter setting suggestions for trainers to flexibly adjust according to their own needs:
    # (1) Start training from the pre-trained weights of the entire model:
    # Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0. (not frozen)
    # SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4. (not frozen)
    # Among them: UnFreeze_Epoch can be adjusted between 100-300.
    # (2) Start training from the pre-trained weights of the backbone network:
    # Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-4, weight_decay = 0. (not frozen)
    # SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 120, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 120, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 1e-4. (not frozen)
    # Among them: Since training starts from the pre-trained weights of the backbone network, the weights of the backbone are not necessarily suitable for semantic segmentation, and more training is required to jump out of the local optimal solution.
    # UnFreeze_Epoch can be adjusted between 120-300.
    # Adam converges faster than SGD. Therefore, UnFreeze_Epoch can theoretically be smaller, but more Epochs are still recommended.
    # (3) Batch_size setting:
    # Within the range that the graphics card can accept, the larger is better. Insufficient video memory has nothing to do with the size of the data set. If it prompts insufficient video memory (OOM or CUDA out of memory), please adjust the batch_size smaller.
    # Since there is a BatchNormalization layer in resnet50
    # When the backbone is resnet50, batch_size cannot be 1
    # Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size. It is not recommended that the setting gap is too large, because it is related to the automatic adjustment of the learning rate.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    # Freeze phase training parameters
    # At this time, the backbone of the model is frozen, and the feature extraction network does not change.
    # It occupies a small amount of video memory and only fine-tunes the network.
    # Init_Epoch is the current training generation of the model. Its value can be greater than Freeze_Epoch, such as setting:
    # Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    # Will skip the freezing stage, start directly from generation 60, and adjust the corresponding learning rate.
    # (used when resuming practice from a breakpoint)
    # Freeze_Epoch model freezes the trained Freeze_Epoch
    # (Invalid when Freeze_Train=False)
    # Freeze_batch_size model freezes the training batch_size
    # (Invalid when Freeze_Train=False)
    # ------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 2
    # ------------------------------------------------------------------#
    # Training parameters during the unfreezing phase
    # At this time, the backbone of the model is no longer frozen, and the feature extraction network will change.
    # It occupies a large amount of video memory and all network parameters will change.
    # UnFreeze_Epoch The total number of epochs trained by the model
    # Unfreeze_batch_size batch_size of the model after unfreezing
    # ------------------------------------------------------------------#
    UnFreeze_Epoch      = 1000
    Unfreeze_batch_size = 2
    # ------------------------------------------------------------------#
    #   Freeze_Train    Whether to perform freeze training
    #                   By default, trunk training is frozen first and then unfrozen.
    # ------------------------------------------------------------------#
    Freeze_Train        = False

    # ------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decrease related
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    # Init_lr The maximum learning rate of the model
    #   It is recommended to set Init_lr=1e-4 when using the Adam optimizer
    #   It is recommended to set Init_lr=1e-2 when using the SGD optimizer
    #
    # Min_lr The minimum learning rate of the model, the default is 0.01 of the maximum learning rate
    # ------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  The type of optimizer used, optional options include adam and sgd
    #                   Recommended setting when using adam optimizer  Init_lr=1e-4
    #                   Recommended setting when using sgd optimizer   Init_lr=1e-2
    #   momentum        Momentum parameter used internally by the optimizer
    #   weight_decay    Weight decay prevents overfitting
    #                   Adam will cause weight decay errors, and it is recommended to set it to 0 when using adam.
    # ------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   The learning rate reduction method used, the options are 'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    # ------------------------------------------------------------------#
    #   save_period     How many epochs are used to save the weight?
    # ------------------------------------------------------------------#
    save_period         = 20
    # ------------------------------------------------------------------#
    #   save_dir        The folder where weights and log files are saved
    # ------------------------------------------------------------------#
    save_dir            = 'logs/white_white/latest_run/'
    # ------------------------------------------------------------------#
    #   eval_flag       Whether to perform evaluation during training, the evaluation object is the verification set
    #   eval_period     Represents how many epochs are evaluated once. Frequent evaluation is not recommended.
    #                   Evaluation takes a lot of time, and frequent evaluation will cause training to be very slow.
    #
    #   The map obtained here will be different from that obtained by get map.py for two reasons:
    #   (1) The map obtained here is the map of the verification set.
    #   (2) The evaluation parameters set here are conservative in order to speed up the evaluation.
    # ------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5

    # ------------------------------#
    #   Dataset path
    # ------------------------------#
    VOCdevkit_path  = '/home/abhi/Thailand_Project/data/white_white/'
    # ------------------------------------------------------------------#
    #   Suggested options:
    # When there are few categories (several categories), set to True
    # When there are many categories (more than a dozen categories), if batch_size is relatively large (more than 10), then set it to True
    # When there are many categories (more than a dozen categories), if batch_size is relatively small (less than 10), then set it to False
    # ------------------------------------------------------------------#
    dice_loss       = True
    # ------------------------------------------------------------------#
    #   Whether to use focal loss to prevent imbalance of positive and negative samples
    # ------------------------------------------------------------------#
    focal_loss      = False
    # ------------------------------------------------------------------#
    #   Whether to assign different loss weights to different categories, the default is balanced.
    #   If setting, pay attention to set it in numpy format, the length is the same as num classes.
    #   like:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multi-threading to read data, 1 means turning off multi-threading
    #                   Turning it on will speed up data reading, but will take up more memory.
    #                   Sometimes enabling multi-threading in Keras is much slower.
    #                   Turn on multi-threading when IO is the bottleneck, that is, the GPU computing speed is much greater than the speed of reading images.
    # ------------------------------------------------------------------#
    num_workers     = 4

    seed_everything(seed)
    # ------------------------------------------------------#
    #   Set the graphics card used
    # ------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    # ----------------------------------------------------#
    #   Download pre-trained weights
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   For the weight file, please see the readme and download it from Baidu Netdisk.
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   Load according to the key of the pre-trained weights and the key of the model
        # ------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   Shows no matching key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m Warm reminder: It is normal that the head part is not loaded, and it is an error that the backbone part is not loaded.\033[0m")

    # ----------------------#
    #   Record loss
    # ----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    # ------------------------------------------------------------------#
    #   Torch 1.2 does not support amp. It is recommended to use torch 1.7.1 and above to use fp16 correctly.
    #   Therefore, torch1.2 shows "could not be resolve" here
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    # ----------------------------#
    #   Multi-SIM synchronization bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   Multi-GPU parallel operation
            #   run with: $python torch.distributed.launch train.py
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   Read the txt corresponding to the data set
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "Segment_config/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "Segment_config/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    # ------------------------------------------------------#
    # Backbone feature extraction network features are universal, and freezing training can speed up training.
    # It can also prevent the weights from being destroyed in the early stages of training.
    # Init_Epoch is the starting generation
    # Interval_Epoch is the generation of frozen training
    # Epoch total training generation
    # If it prompts OOM or insufficient video memory, please adjust the Batch_size smaller.
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   Freeze certain portions of training
        # ------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        # -------------------------------------------------------------------#
        #   If you do not freeze training, directly set the batch size to unfreeze batch size.
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Determine the current batch size and adaptively adjust the learning rate
        # -------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   Select optimizer based on optimizer_type
        # ---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   Obtain the formula for learning rate decrease
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   Determine the length of each generation
        # ---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small to continue training. Please expand the data set.")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   Record the map curve of eval
        # ----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        # ---------------------------------------#
        #   Start model training
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   If the model has a frozen learning part
            #   Then unfreeze and set parameters
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   Determine the current batch size and adaptively adjust the learning rate
                # -------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   Obtain the formula for learning rate decrease
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The data set is too small to continue training. Please expand the data set.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
