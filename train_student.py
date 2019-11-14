import os
import argparse
import torch
from engine import do_train, inference
from models import loss_fn_kd
from utils import Logger, Checkpointer
from tqdm import tqdm
from models import build_model
from engine import build_optimizer, build_lr_scheduler, do_train
from data import CIFAR10_loader
import shutil

def build_teachers(args, device):
    if len(args.teachers) != len(args.teachers_weights):
        raise AssertionError(" [!] Number doesn't match! %d teachers are given, but %d weights are given"%(len(args.teachers), len(args.teachers_weigts)))

    teachers = []
    for teacher_name, teacher_weight_path in zip(args.teachers, args.teachers_weights):
        print(" [*] Load teacher: %s"%teacher_weight_path)
        teacher = build_model(teacher_name, args.num_classes, args.pretrained)
        teacher.load_state_dict(torch.load(teacher_weight_path)['model'])
        teacher = teacher.to(device)
        teacher.eval()
        teachers.append(teacher)
    return teachers

def run_train(args):

    device = torch.device(args.device)
    # build student
    student = build_model(args.student, args.num_classes, args.pretrained)
    student = student.to(device)
    # build teachers
    teachers = build_teachers(args, device)
    # build checkpointer, optimizer, scheduler, logger
    optimizer = build_optimizer(args, student)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = Checkpointer(student, optimizer, scheduler, args.experiment, args.checkpoint_period)
    logger = Logger(os.path.join(args.experiment, 'tf_log'))

    # objective function to train student
    loss_fn = loss_fn_kd

    # data_load
    train_loader = CIFAR10_loader(args, is_train=True)
    test_loader = CIFAR10_loader(args, is_train=False)

    acc1, m_acc1 = inference(student, test_loader, logger, device, 0, args)
    checkpointer.best_acc = acc1
    for epoch in tqdm(range(0, args.max_epoch)):
        do_train(student, teachers, loss_fn, train_loader, optimizer, checkpointer, device, logger, epoch)
        acc1, m_acc1 = inference(student, test_loader, logger, device, epoch+1, args)
        if acc1 > checkpointer.best_acc:
            checkpointer.save("model_best")
            checkpointer.best_acc = acc1
        scheduler.step()
    
    checkpointer.save("model_last")

def main():
    parser = argparse.ArgumentParser(description = "Training arguments for KD")
    parser.add_argument('--lr', default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument('--batch-size', default=128, help="# of dataset you forward at once")
    parser.add_argument('--num-workers', default=4, help="# of worker to queue your dataset")
    parser.add_argument('--momentum', default=0.9, help="Rate to accumulates the gradient of the past steps")
    parser.add_argument('--max-epoch', default=30, type=int, help="Maximum epoch to train")
    parser.add_argument('--lr-decaysteps', default=[15, 25], nargs='+', type=int, help="Decay the learning rate at given steps")
    parser.add_argument('--lr-anneal', default=0.1, help="Multiplicative factor of lr decay")
    parser.add_argument('--weight-decay', default=5e-4, help="L2 regularization coefficient")
    parser.add_argument('--scheduler-type', default='multistep', help="")
    parser.add_argument('--student', default='resnet18', help="student model")
    parser.add_argument('--teachers', default=['resnet101'], nargs='+', type=str, help="Name of teacher models")
    parser.add_argument('--teachers_weights', default=['cifar10_teacher_resnet101_1'], nargs='+', type=str, help="Name of teacher models")
    parser.add_argument('--device', default='cuda', type=str, help="To enable GPU, set 'cuda', otherwise 'cpu'")
    parser.add_argument('--pretrained', default=False, action='store_true', help="Set:True starts from imagenet pretrained model")
    parser.add_argument('--num_classes', default=10, type=int, help="# of classes in the dataset")
    parser.add_argument('--experiment', default='cifar10_student_resnet18_1', type=str, help="Path to save your experiment")
    parser.add_argument('--checkpoint-period', default=5000, type=int, help="Frequency of model save based on # of iterations")
    args = parser.parse_args()
    args.experiment = 'experiment/' + args.experiment

    if os.path.exists(args.experiment):
        print(" [*] %s already exists. Process may overwrite existing experiment"%args.experiment)
    else:
        print(" [*] New experiment is set. Create directory at %s"%args.experiment)
        os.makedirs(args.experiment)
    shutil.copy2('./train_student.sh', args.experiment+'/train_student.sh')

    run_train(args)
    print(" [*] Done! Results are saved in %s"%args.experiment)

if __name__ == "__main__":
    main()
