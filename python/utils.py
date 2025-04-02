import torch

def stop(err):
    print(err)
    exit(1)

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