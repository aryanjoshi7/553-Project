import torch


def stop(err):
    raise ValueError(err)


def is_unique_tensor(t):
    """
    Returns true if all elements of input tensor are unique
    Params:
    t (torch.tensor): tensor to check for unique values
    """
    return t.numel() == torch.unique(t).numel()


def is_unsorted_tensor(t):
    """
    Return true if tensor is not sorted, false if it is
    Params:
    t (torch.tensor): tensor to check for sortedness
    """
    if t.numel() == 0:
        return False
    val = t[0]
    for num in t:
        if num < val:
            return True
        val = num
    return False


def tensor_IQR(t, q1=0.25, q2=0.75):
    """
    Returns the interquartile range (IQR) of t.
    You can specify a custom quantile range by passing values to q1, q2
    Params:
    t (torch.tensor): tensor to calculate IQR on
    q1 (optional, float, default 0.25): lower quantile. takes value in [0, 1]
    q2 (optional, float, default 0.75): upper quantile. takes value in [0, 1]
    """
    if q1 >= q2:
        return 0.0
    quant1 = torch.quantile(t, q1)
    quant2 = torch.quantile(t, q2)
    return quant2 - quant1


def scale(y, c=True, sc=True):
    """
    (I think) A Python port of the R function "scale." Found this on SO
    https://stackoverflow.com/questions/18005305/implementing-r-scale-function-in-pandas-in-python
    """
    # x = y.copy() I don't think we need to make a copy for our purposes

    if c:
        y -= y.mean()
    if sc and c:
        y /= y.std()
    elif sc:
        y /= torch.sqrt(y.pow(2).sum().div(y.count() - 1))
    return y
