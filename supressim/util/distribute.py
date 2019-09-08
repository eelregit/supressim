import torch.distributed as dist

def average_gradients(model):
    """ Computes the average gradient of a model from all ranks. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
