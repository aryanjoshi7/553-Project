class Parameters:
  def __init__(self,lambda_=None, scad_alpha=None, scad_lambda=None, splam_alpha=None, quantile_tau=None, huber_alpha=None, huber_delta=None, quantile_tau_list=None):
        self.lambda_ = lambda_
        
        self.scad_alpha = scad_alpha
        self.scad_lambda = scad_lambda
        self.splam_alpha = splam_alpha
        self.quantile_tau = quantile_tau
        self.quantile_tau_list = quantile_tau_list
        self.huber_alpha = huber_alpha
        self.huber_delta = huber_delta