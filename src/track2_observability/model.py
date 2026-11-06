"""Position is anchored exactly; velocity is the derivative of the same trajectory."""
import torch
from torch import nn
from .common import C

class Dynamics(nn.Module):
    def __init__(self,family):
        super().__init__();self.family=family;d=C['latent_dimensions'];self.d=d
        self.encoder=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,d),nn.GELU())
        self.velocity=nn.Linear(d,2);self.acceleration=nn.Linear(d,2,bias=False)
        nn.init.zeros_(self.velocity.weight);nn.init.zeros_(self.velocity.bias);nn.init.zeros_(self.acceleration.weight)
        if family=='anchored_polynomial':self.jerk=nn.Linear(d,2,bias=False);nn.init.zeros_(self.jerk.weight)
        elif family in ('anchored_linear','anchored_stable'):
            self.A=nn.Parameter(torch.zeros(d,d))
            if family=='anchored_stable':self.damping=nn.Parameter(torch.full((d,),-3.))
        else:raise ValueError('Unknown dynamics family')
    def generator(self):
        if self.family=='anchored_linear':return self.A
        return (self.A-self.A.T)/2-torch.diag(torch.nn.functional.softplus(self.damping))
    def forward(self,x,horizons=None):
        h=x.new_tensor(C['horizons_seconds']) if horizons is None else horizons.to(x)
        z=self.encoder(x.flatten(1));v=self.velocity(z)
        if self.family=='anchored_polynomial':
            a=self.acceleration(z);j=self.jerk(z);t=h[None,:,None]
            return torch.cat([t*v[:,None]+t*t*a[:,None]/2+t**3*j[:,None]/6,v[:,None]+t*a[:,None]+t*t*j[:,None]/2],-1)
        # Block exponential evolves [p, v, z], including singular A without inversion.
        size=self.d+4;top=torch.cat([x.new_zeros(2,2),torch.eye(2,device=x.device,dtype=x.dtype),x.new_zeros(2,self.d)],1)
        middle=torch.cat([x.new_zeros(2,4),self.acceleration.weight],1)
        bottom=torch.cat([x.new_zeros(self.d,4),self.generator()],1);g=torch.cat([top,middle,bottom],0)
        initial=torch.cat([x.new_zeros(len(x),2),v,z],1);prop=torch.matrix_exp(h[:,None,None]*g[None])
        return torch.einsum('hij,bj->bhi',prop,initial)[:,:,:4]
