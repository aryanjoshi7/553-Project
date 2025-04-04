import torch

def getA(a, penalty):
    if penalty in ["aLASSO", "gAdLASSO"]:
        if a is None:
            a = 1
        elif (torch.tensor(a) < 0).sum().item() > 0:
            raise ValueError('For adaptive lasso, the tuning parameter "a" must be positive')
    else:
        if a is None:
            if penalty in ["SCAD", "gSCAD"]:
                a = 3.7
                penalty = "SCAD"
            if penalty in ["MCP", "gMCP"]:
                a = 3
                penalty = "MCP"
        else:
            a_tensor = torch.tensor(a)
            if penalty == "SCAD" and (a_tensor <= 2).sum().item() > 0:
                raise ValueError('Tuning parameter "a" must be larger than 2 for SCAD')
            if penalty == "MCP" and (a_tensor <= 1).sum().item() > 0:
                raise ValueError('Tuning parameter "a" must be larger than 1 for MCP')
    return a