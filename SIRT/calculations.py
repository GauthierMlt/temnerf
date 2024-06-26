#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:05:24 2019

@author: leuliet

Here we draw up a list of annex functions that are often used to compute our algorithms of reconstruction.
These are mostly operations on data ( divergence, division component-wise, projections... etc)
"""

import numpy as np
import astra
import cupy as cp

def pos(x):
    new=x.copy()
    new[x<=0]=0
    return new

def div_zer(a,b):
    assert a.shape==b.shape, "both matrix should be the same size"
    new = a/b
    new[b==0] = 0
#    size=a.shape
#    new=cp.zeros(size)
#    dom=b>0
#    new[dom]=a[dom]/b[dom]
    return new

def div_eps(a,b):
    eps = 10**(-8)
    assert a.shape==b.shape, "both matrix should be the same size"
    new = a/b
    new[b==0] = eps
    new[new<eps] = eps
#    size=a.shape
#    new=cp.zeros(size)
#    dom=b>0
#    new[dom]=a[dom]/b[dom]
#    dom = new<eps
#    new[dom] = eps
    return new


def log_zer(a):
    size=a.shape
    new=cp.zeros(size)
    dom=a>0
    new[dom]=cp.log(a[dom])
    return new

    
def gradient(u) :
    '''
    Compute the gradient of an array u
    
    Parameters
    ----------
    u : cupy array (dim 2 or 3)
        matrix we want the gradient of
    
    Return
    ------
    g : cupy array
        gradient of u
    '''
    assert len(u.shape)==2 or len(u.shape)==3, "function gradient adapted for dim 2 or 3" 
    if len(u.shape)==2:
        grad = cp.zeros((2,u.shape[0],u.shape[1]))
        grad[0] = cp.diff(u,1,axis=0,append=u[-1,:].reshape(1,u.shape[1]))
        grad[1] = cp.diff(u,1,axis=1,append=u[:,-1].reshape(u.shape[0],1))
    elif len(u.shape)==3:
        grad = cp.zeros((3,u.shape[0],u.shape[1],u.shape[2]), dtype = 'float32')
        grad[0] = cp.diff(u,1,axis=0,append=u[-1,:,:].reshape(1,u.shape[1],u.shape[2]))
        grad[1] = cp.diff(u,1,axis=1,append=u[:,-1,:].reshape(u.shape[0],1,u.shape[2]))
        grad[2] = cp.diff(u,1,axis=2,append=u[:,:,-1].reshape(u.shape[0],u.shape[1],1))
    return grad    

def grad_for_div(u) :
    assert len(u.shape)==2 or len(u.shape)==3, "function gradient adapted for dim 2 or 3" 
    if len(u.shape)==2:
        grad = cp.zeros((2,u.shape[0],u.shape[1]))
        grad[0] = cp.diff(u,1,axis=0,prepend=u[0,:].reshape(1,u.shape[1]))
        grad[1] = cp.diff(u,1,axis=1,prepend=u[:,0].reshape(u.shape[0],1))
    elif len(u.shape)==3:
        grad = cp.zeros((3,u.shape[0],u.shape[1],u.shape[2]), dtype = 'float32')
        grad[0] = cp.diff(u,1,axis=0,prepend=u[0,:,:].reshape(1,u.shape[1],u.shape[2]))
        grad[1] = cp.diff(u,1,axis=1,prepend=u[:,0,:].reshape(u.shape[0],1,u.shape[2]))
        grad[2] = cp.diff(u,1,axis=2,prepend=u[:,:,0].reshape(u.shape[0],u.shape[1],1))
    return grad    

def divergence(u) :
    '''
    Compute the divergence of the array p
    
    Parameters
    ----------
     u : cupy array
    
    Return
    ------
    div : cupy array
          array of one dimension 
    '''
    assert u.shape[0]==2 or u.shape[0]==3, "divergence adapted for dim 2 or 3 only"
    if u.shape[0] == 2:
        div = grad_for_div(u[0])[0] + grad_for_div(u[1])[1]
    elif u.shape[0] == 3:
        div = grad_for_div(u[0])[0] + grad_for_div(u[1])[1] + grad_for_div(u[2])[2]

    return div   


def add_gaussian_noise(f,mean,sigma) :
    '''
    Add a gaussian noise to an image f
    '''
    f += cp.random.normal(mean,sigma,f.shape)
    return f



def TV(z):
    grad=gradient(z)
    norm=cp.sqrt(cp.sum(grad**2,axis=0))
    return cp.sum(norm)

def make_projection(data,vol_geom,proj_geom):
    '''
    Compute the projection of the image data
    data is a cupy array which needs to be transformed into a numpy array for Astra use
    
    Parameters
    ----------
     data :    cupy array
               image to project
    vol_geom : Astra object
               geometry of the volume reconstructed
    proj_geom: Astra object
               geometry of the projections
    
    Return
    ------
    projections : cupy array
                  projections of the image 
    '''    
    npdata = cp.asnumpy(data)
    if len(npdata.shape)==3:
        data_id=astra.data3d.create('-vol', vol_geom, data=npdata)
        projections_id, projections = astra.creators.create_sino3d_gpu(data_id, proj_geom, vol_geom)
        astra.data3d.delete(projections_id)
        astra.data3d.delete(data_id)    
    elif len(npdata.shape)==2:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        projections_id, projections = astra.creators.create_sino(npdata,projector_id)
        astra.data2d.delete(projector_id)
        astra.data2d.delete(projections_id)
    projections = cp.asarray(projections)
    return projections

def backprojection(proj,vol_geom,proj_geom):
    '''
    Compute the backprojection of a sinogram
    proj is a cupy array which needs to be transformed into a numpy array for Astra use
    
    Parameters
    ----------
     proj :    cupy array
               sinogram to backproject
    vol_geom : Astra object
               geometry of the volume reconstructed
    proj_geom: Astra object
               geometry of the projections
    
    Return
    ------
    bp : cupy array
         backprojection of the sinogram 
    '''        
    np_proj = cp.asnumpy(proj)
    if len(np_proj.shape)==3:
        bp_id , bp = astra.creators.create_backprojection3d_gpu(np_proj, proj_geom, vol_geom)
        astra.data3d.delete(bp_id)
    elif len(np_proj.shape)==2:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        bp_id, bp = astra.creators.create_backprojection(np_proj,projector_id)
        astra.data2d.delete(projector_id)
        astra.data2d.delete(bp_id)
    bp = cp.asarray(bp)
    return bp
#####################################################


def norm_A_grad(imshape, nbiter,vol_geom,proj_geom): #for a Radon transform
    x=cp.ones(imshape) #initialize x0 to a non-zero image
    for i in range(nbiter):
        proj = make_projection(x,vol_geom,proj_geom)
        bp = backprojection(proj,vol_geom,proj_geom)
        bp=bp-divergence(gradient(x))
        bp=bp/cp.linalg.norm(bp)
        new_proj = make_projection(bp,vol_geom,proj_geom)
        s=cp.linalg.norm(new_proj)**2+cp.linalg.norm(gradient(bp))**2
        s=cp.sqrt(s)
    return s

def KL(x,y,vol_geom,proj_geom):
    proj = make_projection(x,vol_geom,proj_geom)
    val=cp.sum(proj-y+y*log_zer(pos(y))-y*log_zer(proj))
    return val    



def cost_function(x,p,alpha,vol_geom, proj_geom, noise) :
    '''
    Compute the value of the cost function
    
    Parameters
    ----------
    x : cp.array
        current reconstructed volume
    p : cp.array
        projection observed
    alpha : float
            TV parameter
    proj_geom : Astra object
    vol_geom : Astra object
    noise : string
            'poisson' or 'gaussian'
    Return
    ------
    cost : float
           value of cost function
    '''
    proj = make_projection(x,vol_geom,proj_geom)
    if noise == 'poisson' :
        #cost = proj - p + p*np.log(p) - p*np.log(pos(proj))
        cost = proj - p*np.log(pos(proj))
        cost = cp.sum(cost) + alpha*TV(x)
    elif noise == 'gaussian' :
        cost = 0.5*cp.linalg.norm(proj-p,'fro')**2 + alpha*TV(x)
    else :
        raise IOError('Noise should be gaussian or poisson')
    return float(cost)    

def cPD(u,p,g,alpha,vol_geom,proj_geom):
    cpd=cost_function(u,g,alpha,vol_geom,proj_geom)-cp.sum(g*log_zer(pos(1-p)))
    return cpd    




def div_zer_replace_first(a,b):
    assert a.shape==b.shape, "both matrix should be the same size"
    a = a/b
    a[b==0] = 0


def TV_denoising_Poisson(f, size_image, s, alpha, nbiter2, max_vol_size) :
    # compute the gradient descent step
    num = np.min(s)-2*len(size_image)*alpha
    if (num <=0) :
       num = 1
    Lh=4*len(size_image)*alpha**2*np.max(s*f)/(num**2)
    tau=alpha/Lh
    # auxiliary variable
    size = list(size_image)
    size.insert(0,len(size_image))
    phi=np.zeros(tuple(size), dtype = 'float32') #6n^3
    # optimization on the dual variable
    for k in range(nbiter2):
        z=gradient_numpy(div_zer(s*f,s+alpha*divergence_numpy(phi))) #9n^3 (up to 10n^3 during computation)
        denom=np.sum(z**2,axis=0) #10n^3
        denom=1+tau*np.sqrt(denom)
        phi=(phi-tau*z)/denom
    f=div_eps(s*f,s+alpha*divergence_numpy(phi)) #up to 9n^3 momentaneously


def make_projection_numpy(data,vol_geom,proj_geom):
    '''
    Compute the projection of the image data
    data is a cupy array which needs to be transformed into a numpy array for Astra use
    
    Parameters
    ----------
     data :    cupy array
               image to project
    vol_geom : Astra object
               geometry of the volume reconstructed
    proj_geom: Astra object
               geometry of the projections
    
    Return
    ------
    projections : numpy array
                  projections of the image 
    '''    
 #   npdata = cp.asnumpy(data)
    if len(data.shape)==3:
        data_id=astra.data3d.create('-vol', vol_geom, data=data)
        projections_id, projections = astra.creators.create_sino3d_gpu(data_id, proj_geom, vol_geom)
        astra.data3d.delete(projections_id)
        astra.data3d.delete(data_id)    
    elif len(data.shape)==2:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        projections_id, projections = astra.creators.create_sino(data,projector_id)
        astra.data2d.delete(projector_id)
        astra.data2d.delete(projections_id)
#    projections = np.asarray(projections)
    return projections


def backprojection_numpy(proj,vol_geom,proj_geom):
    '''
    Compute the backprojection of a sinogram
    proj is a cupy array which needs to be transformed into a numpy array for Astra use
    
    Parameters
    ----------
     proj :    numpy array
               sinogram to backproject
    vol_geom : Astra object
               geometry of the volume reconstructed
    proj_geom: Astra object
               geometry of the projections
    
    Return
    ------
    bp : numpy array
         backprojection of the sinogram 
    '''        
#    np_proj = cp.asnumpy(proj)
    if len(proj.shape)==3:
        bp_id , bp = astra.creators.create_backprojection3d_gpu(proj, proj_geom, vol_geom)
        astra.data3d.delete(bp_id)
    elif len(proj.shape)==2:
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        bp_id, bp = astra.creators.create_backprojection(proj,projector_id)
        astra.data2d.delete(projector_id)
        astra.data2d.delete(bp_id)
#    bp = np.asarray(bp)
    return bp
#####################################################

def gradient_numpy(u) :
    '''
    Compute the gradient of an array u
    
    Parameters
    ----------
    u : numpy array (dim 2 or 3)
        matrix we want the gradient of
    
    Return
    ------
    g : numpy array
        gradient of u
    '''
    assert len(u.shape)==2 or len(u.shape)==3, "function gradient adapted for dim 2 or 3" 
    if len(u.shape)==2:
        grad = np.zeros((2,u.shape[0],u.shape[1]))
        grad[0] = np.diff(u,1,axis=0,append=u[-1,:].reshape(1,u.shape[1]))
        grad[1] = np.diff(u,1,axis=1,append=u[:,-1].reshape(u.shape[0],1))
    elif len(u.shape)==3:
        grad = np.zeros((3,u.shape[0],u.shape[1],u.shape[2]), dtype = 'float32')
        grad[0] = np.diff(u,1,axis=0,append=u[-1,:,:].reshape(1,u.shape[1],u.shape[2]))
        grad[1] = np.diff(u,1,axis=1,append=u[:,-1,:].reshape(u.shape[0],1,u.shape[2]))
        grad[2] = np.diff(u,1,axis=2,append=u[:,:,-1].reshape(u.shape[0],u.shape[1],1))
    return grad    

def grad_for_div_numpy(u) :
    assert len(u.shape)==2 or len(u.shape)==3, "function gradient adapted for dim 2 or 3" 
    if len(u.shape)==2:
        grad = np.zeros((2,u.shape[0],u.shape[1]))
        grad[0] = np.diff(u,1,axis=0,prepend=u[0,:].reshape(1,u.shape[1]))
        grad[1] = np.diff(u,1,axis=1,prepend=u[:,0].reshape(u.shape[0],1))
    elif len(u.shape)==3:
        grad = np.zeros((3,u.shape[0],u.shape[1],u.shape[2]), dtype = 'float32')
        grad[0] = np.diff(u,1,axis=0,prepend=u[0,:,:].reshape(1,u.shape[1],u.shape[2]))
        grad[1] = np.diff(u,1,axis=1,prepend=u[:,0,:].reshape(u.shape[0],1,u.shape[2]))
        grad[2] = np.diff(u,1,axis=2,prepend=u[:,:,0].reshape(u.shape[0],u.shape[1],1))
    return grad    

def divergence_numpy(u) :
    '''
    Compute the divergence of the array p
    
    Parameters
    ----------
     u : numpy array
    
    Return
    ------
    div : numpy array
          array of one dimension 
    '''
    assert u.shape[0]==2 or u.shape[0]==3, "divergence adapted for dim 2 or 3 only"
    if u.shape[0] == 2:
        div = grad_for_div_numpy(u[0])[0] + grad_for_div_numpy(u[1])[1]
    elif u.shape[0] == 3:
        div = grad_for_div_numpy(u[0])[0] + grad_for_div_numpy(u[1])[1] + grad_for_div_numpy(u[2])[2]

    return div   
