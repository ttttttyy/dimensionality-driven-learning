import torch
import torch.nn.functional as F

class lass(object):
    def __init__(self, model, device, a=0.25/255., b=0.2/255., r=0.3/255., iter_max=100, clip_min=-1.0e8, clip_max=1.0e8):
        # x and y_target are tensorflow placeholders, y_pred is the model output tensorflow tensor
        # SEARCH PARAMETERS: a- gradient sign coefficient; b- noise coefficient; r- search radius per pixel; iter- max number of iters
        self.a = a
        self.b = b
        self.r = r
        self.model = model
        self.device = device
        self.iter_max = iter_max
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def find(self, X):
        # elements of X in [0,1] for using default params a,b,r; otherwise scale accordingly
        # generate max output label
        X.requires_grad_(True)
        pred, _ = self.model(X)
        pred = F.softmax(pred, dim=1)
        Y_pred_vec = torch.argmax(pred, dim=1)
        Y_pred = F.one_hot(Y_pred_vec, pred.shape[1]).float()
        
        X_adv = 1.*X
        adv_ind = torch.zeros(X.shape[0],dtype=torch.bool,device=self.device)
        converged = False
        converged_label_thres = 3
        adv_num_old = 0 
        i = 0
        Y_pred_adv = pred
        while i < self.iter_max and converged == False:
            # I would recommend annealing the noise coefficient b gradually in this while loop
            #print('on iter %s' % i)
            i += 1
            #X_adv.requires_grad_(True)
            loss = F.cross_entropy(Y_pred_adv, Y_pred_vec)
            if i == 1:
                grad = torch.autograd.grad(loss, X)[0]
            else:
                grad = torch.autograd.grad(loss, X_adv)[0]
            X_adv = X_adv.detach()
            
  
            step = self.a * torch.sign(grad) + self.b * torch.randn(*grad.shape, device=self.device)
            X_adv += step
            diff = X_adv - X
            abs_diff = torch.abs(diff)
            ind = abs_diff > self.r
            X_adv[ind] = X[ind] + self.r * torch.sign(diff[ind])  
            X_adv = torch.clamp(X_adv, self.clip_min , self.clip_max )
            
            X_adv.requires_grad_(True)
            Y_pred_adv, _ = self.model(X_adv)
            Y_pred_adv = F.softmax(Y_pred_adv, dim=1)
            Y_pred_adv_vec = torch.argmax(Y_pred_adv, dim=1)
            # if we ever identify a sample as critical sample, record it
            adv_ind = adv_ind | ~torch.eq(Y_pred_vec, Y_pred_adv_vec).to(self.device)
            adv_num_new = torch.sum(adv_ind)
            #print('number of adv samples: %s' % adv_num_new)
            
            if adv_num_new - adv_num_old < converged_label_thres:
                converged = True
                
            adv_num_old = adv_num_new
            
        return X_adv, adv_ind