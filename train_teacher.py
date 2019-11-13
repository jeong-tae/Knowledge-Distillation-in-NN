import os, time
import argparse
from utils import Logger, Checkpointer
from tqdm import tqdm
from models import build_model
from engine import build_optimizer, build_lr_scheduler, inference
from data import CIFAR10_loader
from utils import ClassWiseAverageMeter, AverageMeter


def run_train(args):

    device = torch.device(args.device)

    model = build_model(args.model_name)
    model = model.to(device)
    # build checkpointer, optimizer, scheduler, logger
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    checkpointer = build_checkpoint(student, optimizer, scheduler, args.experiment, args.checkpoint_period)
    logger = Logger(os.path.join(args.experiment, 'tf_log'))

    # objective function to train student
    loss_fn = loss_fn_kd


    # data_load
    train_loader = CIFAR_loader(args, is_train=True)

    for epoch in tqdm(range(0, args.max_epoch)):
        train_epoch(model, train_loader, optimizer, checkpointer, device, logger)
        inference(model, test_loader, logger, device)
    
    checkpointer.save("model_last")

def train_epoch(model, train_loader, optimizer, current_iter, checkpointer, device, logger):
    model.train()

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        t1 = time.time()
        output = model(images)
        loss = F.cross_entropy(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        if current_iter % 20 == 0:
            print(" [*] Iter %d || Loss: %.4f || Timer: %.4f"%(current_iter, loss.item(), t2 - t1))

        if current_iter % checkpointer.checkpoint_period == 0:
            checkpointer.save("model_{:07}".format(current_iter))

        logger.scalar_summary("train_loss", loss.item(), current_iter)
        current_iter += 1

    return current_iter

def main():
    parser = argparse.ArgumentParser(description = "Training arguments for KD")
    parser.add_argument('--lr', default=1e-3, help="Initial learning rate")
    parser.add_argument('--batch-size', default=128, help="# of dataset you forward at once")
    parser.add_argument('--num-workers', default=4, help="# of worker to queue your dataset")
    parser.add_argument('--momentum', default=0.9, help="Rate to accumulates the gradient of the past steps")
    parser.add_argument('--max-epoch', default=30, type=int, help="Maximum epoch to train")
    parser.add_argument('--lr-decaysteps', default=[15, 25], nargs='+', type=int, help="Decay the learning rate at given steps")
    parser.add_argument('--lr-anneal', default=0.1, help="Multiplicative factor of lr decay")
    parser.add_argument('--model_name', default='resnet101', help="Model name to train")
    parser.add_argument('--cuda', default=True, type=bool, help="Set True: avaiable cuda, GPU")
    parser.add_argument('--num_classes', default=10, type=int, help="# of classes in the dataset")
    parser.add_argument('--experiment', default='cifar10_teacher_resnet101_1', type=str, help="Path to save your experiment")
    parser.add_argument('--checkpoint-period', default=5000, type=int, help="Frequency of model save based on # of iterations")

    args = parser.parse_args()

    if os.path.exists(args.experiment):
        print(" [*] %s already exists. Process may overwrite existing experiemnt"%args.experiment)
    else:
        print(" [*] New experiment is set. Create directory at %s"%args.experiment)
        os.makedirs(args.experiment)

    run_train(args)

if __name__ == "__main__":
    main()
