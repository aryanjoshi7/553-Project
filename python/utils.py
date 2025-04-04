import torch

def stop(err):
    raise ValueError(err)

def is_unique_tensor(t):
    '''
    Returns true if all elements of input tensor are unique
    Params:
    t (torch.tensor): tensor to check for unique values
    '''
    return t.numel() == torch.unique(t).numel()


def is_unsorted_tensor(t):
    '''
    Return true if tensor is not sorted, false if it is
    Params:
    t (torch.tensor): tensor to check for sortedness
    '''
    if t.numel() == 0:
        return False
    val = t[0]
    for num in t:
        if num < val:
            return True
        val = num
    return False

def tensor_IQR(t, q1=0.25, q2=0.75):
    '''
    Returns the interquartile range (IQR) of t.
    You can specify a custom quantile range by passing values to q1, q1
    Params:
    t (torch.tensor): tensor to calculate IQR on
    q1 (optional, float, default 0.25): lower quantile. takes value in [0, 1]
    q1 (optional, float, default 0.75): upper quantile. takes value in [0, 1]
    '''
    if q1 >= q2:
        return 0.0
    quant1 = torch.quantile(t, q1)
    quant2 = torch.quantile(t, q2)
    return quant2 - quant1