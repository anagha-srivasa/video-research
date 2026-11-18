"""Fixed-association v3 replay: detector innovations only; LK transports with diffusion."""
import numpy as np
from .belief_filter import Belief,advance,condition


def replay_v3(archives,q,floor,output=False):
    score=0.;count=0;nis_total=0.;states={}
    for episode,values in archives:
        if not len(values):continue
        if values.shape[1]!=32 or np.any(np.diff(values[:,0])<0):raise ValueError('Chronological v3 rows required')
        size=int(values[:,1].max())+1
        if not 0<size<=100000:raise ValueError('Bounded nonnegative IDs required')
        means=np.zeros((size,4));covs=np.zeros((size,4,4));last=np.full(size,np.nan)
        cuts=np.r_[0,np.flatnonzero(np.diff(values[:,0]))+1,len(values)]
        for begin,end in zip(cuts[:-1],cuts[1:]):
            rows=values[begin:end];ids=rows[:,1].astype(int);t=rows[0,0]
            if len(set(ids))!=len(ids) or np.any(ids<0):raise ValueError('Unique nonnegative IDs per timestamp required')
            seen=np.isfinite(last[ids]);fresh=ids[~seen]
            if len(fresh):
                assert np.all(rows[~seen,28]==1)
                means[fresh]=np.c_[rows[~seen,2:4],rows[~seen,10:12]]
                covs[fresh,:2,:2]=rows[~seen,4:8].reshape(-1,2,2)+np.eye(2)*(floor**2+1e-4);covs[fresh,2:,2:]=np.eye(2)*.01
            old=ids[seen]
            if len(old):
                batch=rows[seen];dt=t-last[old]
                if np.any(dt<=0):raise ValueError('Increasing per-ID time required')
                f=np.broadcast_to(np.eye(4),(len(old),4,4)).copy();f[:,0,2]=dt;f[:,1,3]=dt
                process=np.zeros_like(f)
                for axis in range(2):
                    process[:,axis,axis]=q*dt**3/3;process[:,axis,axis+2]=process[:,axis+2,axis]=q*dt**2/2;process[:,axis+2,axis+2]=q*dt
                mean=(f@means[old,:,None])[...,0];cov=f@covs[old]@f.transpose(0,2,1)+process
                lk=batch[:,28]==2;det=~lk
                if lk.any():
                    assert np.all(batch[lk,31]>0)
                    mean[lk,:2]=means[old[lk],:2]+batch[lk,29:31];mean[lk,2:]=batch[lk,29:31]/batch[lk,31,None]
                    cov[lk,:2,:2]+=batch[lk,4:8].reshape(-1,2,2)+np.eye(2)*(.015**2*batch[lk,31,None,None])
                if det.any():
                    r=batch[det,4:8].reshape(-1,2,2)+np.eye(2)*floor**2;c=cov[det];m=mean[det]
                    innovation=batch[det,2:4]-m[:,:2];s=c[:,:2,:2]+r;sign,logdet=np.linalg.slogdet(s)
                    if np.any(sign<=0):raise ValueError('Nonpositive innovation covariance')
                    nis=np.einsum('bi,bi->b',innovation,np.linalg.solve(s,innovation[...,None])[...,0]);score+=float((logdet+nis).sum());nis_total+=float(nis.sum());count+=int(det.sum())
                    gain=np.linalg.solve(s,c[:,:2,:]).transpose(0,2,1);m+=(gain@innovation[...,None])[...,0]
                    a=np.broadcast_to(np.eye(4),c.shape).copy();a[:,:,:2]-=gain
                    mean[det]=m;cov[det]=a@c@a.transpose(0,2,1)+gain@r@gain.transpose(0,2,1)
                means[old]=mean;covs[old]=cov
            last[ids]=t
            if output:
                for tid in ids:states[(episode,int(round(t*1e6)),int(tid))]=(means[tid].copy(),covs[tid].copy())
    replay_v3.last_nis=nis_total/max(count,1)
    return score/max(count,1),count,states


def replay_v3_scalar(archives,q,floor,output=False):
    """Independent scalar equations used to verify the batched implementation."""
    score=0.;nis_total=0.;count=0;states={}
    for episode,values in archives:
        tracks={}
        for row in values:
            t,tid=row[:2];r=row[4:8].reshape(2,2);z=row[2:4];source=int(row[28])
            if tid not in tracks:
                assert source==1
                cov=np.zeros((4,4));cov[:2,:2]=r+np.eye(2)*(floor**2+1e-4);cov[2:,2:]=np.eye(2)*.01
                state=Belief(t,np.r_[z,row[10:12]],cov,np.eye(2),'OBSERVED')
            else:
                old=tracks[tid];pred=advance(old,t,q)
                if source==2:
                    cov=pred.covariance.copy();cov[:2,:2]+=r+np.eye(2)*.015**2*row[31]
                    state=Belief(t,np.r_[old.mean[:2]+row[29:31],row[29:31]/row[31]],cov,np.eye(2),'OBSERVED')
                else:
                    r=r+np.eye(2)*floor**2;delta=z-pred.mean[:2];s=pred.covariance[:2,:2]+r
                    nis=float(delta@np.linalg.solve(s,delta));nis_total+=nis;score+=np.linalg.slogdet(s)[1]+nis;count+=1
                    state=condition(pred,z,r)
            tracks[tid]=state
            if output:states[(episode,int(round(t*1e6)),int(tid))]=(state.mean.copy(),state.covariance.copy())
    replay_v3_scalar.last_nis=nis_total/max(count,1)
    return score/max(count,1),count,states
