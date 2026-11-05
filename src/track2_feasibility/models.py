"""Four fixed model families; CPU float32 training and cached inference propagators."""
import torch
from torch import nn
from .data import CFG,INPUT_DIM,OUTPUT_DIM

class Dynamics(nn.Module):
 def __init__(self,family):
  super().__init__();self.family=family;d=CFG['latent_dimension'];width=CFG['encoder_width'];self.register_buffer('horizons',torch.tensor(CFG['horizons_us'],dtype=torch.float32)/1e6);self.register_buffer('_cache',torch.empty(0),persistent=False)
  if family in ['koopman','stable_koopman']:
   # A fixed four-state observation encoder provides one compact latent history state.
   self.encoder=nn.Sequential(nn.Linear(4*INPUT_DIM,width),nn.GELU(),nn.Linear(width,d));self.decoder=nn.Sequential(nn.Linear(d,width),nn.GELU(),nn.Linear(width,OUTPUT_DIM))
   if family=='koopman':self.A=nn.Parameter(torch.randn(d,d)*.01)
   else:self.K=nn.Parameter(torch.randn(d,d)*.02);self.L=nn.Parameter(torch.eye(d)*.31622777);self.B=nn.Parameter(torch.eye(d))
  elif family=='gru':self.gru=nn.GRU(INPUT_DIM,d,batch_first=True);self.head=nn.Linear(d,3*OUTPUT_DIM)
  elif family=='mlp':self.net=nn.Sequential(nn.Linear(4*INPUT_DIM,width),nn.GELU(),nn.Linear(width,d),nn.GELU(),nn.Linear(d,3*OUTPUT_DIM))
  else:raise ValueError('Unknown locked model family')
 def generator(self):
  if self.family=='koopman':return self.A
  if self.family!='stable_koopman':raise ValueError('No linear generator')
  eye=torch.eye(self.K.shape[0],device=self.K.device,dtype=self.K.dtype);j=(self.K-self.K.T)/2;r=self.L@self.L.T+CFG['generator_epsilon']*eye;h=self.B@self.B.T+CFG['generator_epsilon']*eye;return (j-r)@h
 def train(self,mode=True):
  self._cache=self._cache.new_empty(0);return super().train(mode)
 def compile_inference(self):
  self.eval()
  if self.family in ['koopman','stable_koopman']:
   with torch.no_grad():self._cache=torch.matrix_exp(self.horizons[:,None,None]*self.generator()[None]).detach()
 def forward(self,x):
  if x.ndim!=3 or x.shape[1:]!=(4,INPUT_DIM):raise ValueError('Expected four normalized frozen M16 states')
  if self.family in ['koopman','stable_koopman']:
   z=self.encoder(x.flatten(1));a=self._cache if not self.training and self._cache.numel() else torch.matrix_exp(self.horizons[:,None,None]*self.generator()[None]);future=torch.einsum('hij,bj->bhi',a,z);return self.decoder(future),z.square().mean()
  if self.family=='gru':_,z=self.gru(x);return self.head(z[-1]).reshape(-1,3,OUTPUT_DIM),z.square().mean()
  return self.net(x.flatten(1)).reshape(-1,3,OUTPUT_DIM),x.new_tensor(0.)

def loss(prediction,target,mask,latent_penalty,weights=None):
 import torch.nn.functional as F
 if mask.shape!=prediction.shape[:2]:raise ValueError('Explicit target mask required')
 if not mask.any():raise ValueError('Batch has no supported future targets')
 weights=torch.ones(len(prediction),device=prediction.device) if weights is None else weights
 w=mask.float()*weights[:,None];den=w.sum().clamp_min(1e-12);pos=F.smooth_l1_loss(prediction[:,:,:2],target[:,:,:2],reduction='none').mean(-1);vel=F.smooth_l1_loss(prediction[:,:,2:4],target[:,:,2:4],reduction='none').mean(-1);coarse=(prediction[:,:,4:]-target[:,:,4:]).square().mean(-1);terms=[(pos*w).sum()/den,(vel*w).sum()/den,(coarse*w).sum()/den,latent_penalty];return sum(a*b for a,b in zip(CFG['loss_weights'],terms))
