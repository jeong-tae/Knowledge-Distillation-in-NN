import time
import torch
from tqdm import tqdm

def do_train(student, teachers, loss_fn, data_loader, optimizer, checkpointer, device, logger, current_epoch = 0):
    """
      args:
        student: 
        teachers: list of pretrained model
        data_loader:
        optimizer:
        checkpointer:
        device:
        logger:
    """
    if not isinstance(teachers, list):
        teachers = [teachers]

    global_iteration = current_epoch*len(data_loader)

    for iteration, (images, labels) in tqdm(enumerate(data_loader), desc="Train:":
        images = images.to(device)
        labels = labels.to(device)

        t1 = time.time()
        s_output = student(images)
        t_outputs = [teacher(images) for teacher in teachers]

        optimizer.zero_grad()
        loss = loss_fn(s_output, t_outputs, labels, T=1, alpha=0.5)
        loss.backward()
        optimizer.step()
        t2 = time.time()

        global_iteration += iteration

        #==================== Logging =====================#
        logger.scalar_summary("train_loss", loss.item(), global_iteration)

        if global_iteration % 20 == 0:
            print(" [*] Iter %d || Loss: %.4f || Timer: %.4f"%(global_iteration, loss.item(), t2 - t1))

        if global_iteration % checkpointer.checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(global_iteration))
            # Need evaluation?

