import os
import argparse
from engine import do_train, inference
from models import loss_fn_kd
from utils import Logger, Checkpointer
from tqdm import tqdm
from models import build_model
from engine import build_optimizer, build_lr_scheduler, do_train
from data import CIFAR10_loader

def build_teachers(args):
    if len(args.teachers) != len(args.teachers_weights):
        raise AssertionError(" [!] Number doesn't match! %d teachers are given, but %d weights are given"%(len(args.teachers), len(args.teachers_weigts))
    device = torch.device(args.device)

    teachers = []
    for teacher_name, teacher_weight_path in zip(args.teachers, args.teachers_weights):
        teacher = build_model(teacher_name)
        teacher.load_state_dict(torch.load(teacher_weight_path))
        teacher = teacher.to(device)
        teacher.eval()
        teachers.append(teacher)
    return teachers

def run_train(args):

    # build student
    student = build_model(args.student)
    # build teachers
    teachers = build_teachers(args)
    # build checkpointer, optimizer, scheduler, logger
    optimizer = build_optimizer(args, student)
    scheduler = build_scheduler(args, optimizer)
    checkpointer = build_checkpoint(student, optimizer, scheduler, args.experiment, args.checkpoint_period)
    logger = Logger(os.path.join(args.experiment, 'tf_log'))

    # objective function to train student
    loss_fn = loss_fn_kd


    # data_load
    train_loader = CIFAR_loader(args, is_train=True)

    for epoch in tqdm(range(0, args.max_epoch)):
        do_train(student, teachers, loss_fn, train_loader, optimizer, scheduler, checkpointer, device, logger, epoch)
        inference(student, test_loader, logger, device)
    
    checkpointer.save("model_last")

def main():
    parser = argparse.ArgumentParser(description = "Training arguments for KD")
    parser.add_argument('--lr', default=1e-3, help="Initial learning rate")
    parser.add_argument('--batch-size', default=128, help="# of dataset you forward at once")
    parser.add_argument('--num-workers', default=4, help="# of worker to queue your dataset")
    parser.add_argument('--momentum', default=0.9, help="Rate to accumulates the gradient of the past steps")
    parser.add_argument('--lr-decaysteps', default=None, nargs='+', type=int, help="Decay the learning rate at given steps")
    parser.add_argument('--lr-anneal', default=0.1, help="Multiplicative factor of lr decay")
    parser.add_argument('--student', default='resnet18', help="student model")
    parser.add_argument('--device', default='cuda', type=str, help="To enable GPU, set 'cuda', otherwise 'cpu'")

    args = parser.parse_args()

    if os.path.exists(args.experiment):
        print(" [*] %s already exists. Process may overwrite existing experiment"%args.experiment)
    else:
        print(" [*] New experiment is set. Create directory at %s"%args.experiment)
        os.makedirs(args.experiment)

    run_train(args)

if __name__ == "__main__":
    main()
