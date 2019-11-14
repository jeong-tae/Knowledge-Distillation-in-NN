import torch.nn.functional as F

def loss_fn_kd(s_output, t_outputs, labels, T=1, alpha=0, weights=None):
    """
      args:
        s_output: student's output
        t_outputs: list of specialists' output, including generalist
        T: Temperature
        alpha: Control value to adjust contribution of teacher
        weights: consider weight of each class to get cross entropy, 
                 normally used for imbalanced dataset

      return:
        loss: KL(p^q, q) + sum(KL(p^m, q)) + cross_entropy(q, y)
    """
    if not isinstance(t_outputs, list):
        t_outputs = [t_outputs]

    s_output = F.log_softmax(s_output/T, 1)
    kd_loss = sum([F.kl_div(s_output, F.softmax(t_output/T, 1), reduction='batchmean') for t_output in t_outputs])
    loss = (1 - alpha)*F.cross_entropy(s_output, labels) + alpha*kd_loss
    return loss

