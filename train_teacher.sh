CUDA_VISIBLE_DEVICES=0 python3 train_teacher.py --experiment 'cifar10_teacher_resnet18_3' --lr 1e-1 --max-epoch 200 --lr-decaysteps 100 --model_name 'resnet18'
