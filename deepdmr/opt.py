'''
Optimizers.

Adrian Benton
'''

import json
import numpy as np

import theano
import theano.tensor as T

class Optimizer:
  def getUpdates(self, params, grads):
    '''
    :type params: [ theano_shared_variable ]
    :param params: model weights
    
    :type grads: [ theano_tensor ]
    :param grads: symbolic gradient w.r.t each set of weights
    
    :returns: [ (theano_shared_variable, theano_tensor) ] -- updated weights/history
              after a single gradient step
    '''
    raise NotImplementedError
  
  def reset(self):
    '''
    To reset history, etc.
    '''
    raise NotImplementedError
  
  def toJson(self):
    raise NotImplementedError

class SGDOptimizer(Optimizer):
  ''' Vanilla SGD with decay. '''
  
  def __init__(self, learningRate=0.01, decay=1.0):
    '''
    :type learningRate: float
    :param learningRate: how big a step to take each epoch
    '''
    
    self.learningRate = theano.shared(np.array(learningRate).astype(theano.config.floatX))
    self.decay        = theano.shared(np.array(decay).astype(theano.config.floatX))
  
  def getUpdates(self, params, grads):
    updates = []
    
    for p, g in zip(params, grads):
      step = self.learningRate * g
      updates.append( (p, p + step) )
    
    updates.append( ( self.learningRate, self.learningRate*self.decay ) )
    
    return updates
  
  def reset(self):
    pass
  
  def toJson(self):
    return json.dumps({'type':'sgd',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'decay':self.decay.get_value()}})

class MomentumSGDOptimizer(Optimizer):
  ''' SGD with momentum term. '''
  
  def __init__(self, learningRate=0.01, momentum=0.0, decay=1.0):
    '''
    :type learningRate: float
    :param learningRate: how big a step to take each epoch
    
    :type momentum: float
    :param momentum: how badly to go in the same direction
    '''
    self.learningRate = theano.shared( np.array(learningRate).astype(theano.config.floatX) )
    self.momentum     = theano.shared( np.array(momentum).astype(theano.config.floatX) )
    self.decay        = theano.shared( np.array(decay).astype(theano.config.floatX) )
    
    self.__prevSteps = None # will initialize these to zero once we know the shape of weights
  
  def getUpdates(self, params, grads):
    if self.__prevSteps is None:
      self.__prevSteps = []
      for p in params:
        self.__prevSteps.append( theano.shared(p.get_value()*np.array(0.).
                                               astype(theano.config.floatX),
                                             allow_downcast=True) )
    
    updates = []
    
    for prevStep, p, g in zip(self.__prevSteps, params, grads):
      momentumPrev = self.momentum*prevStep
      sgdStep = self.learningRate * g
      step = momentumPrev + sgdStep
      
      updates.append( (prevStep, step) )
      updates.append( (p, p + step) )
    
    updates.append( ( self.learningRate, self.learningRate*self.decay ) )
    
    return updates
  
  def toJson(self):
    return json.dumps({'type':'sgd_momentum',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'momentum':self.momentum.get_value(),
                                 'decay':self.decay.get_value()}})

class AdamOptimizer(Optimizer):
  '''
  Adam, adaptive learning rate optimization: https://arxiv.org/pdf/1412.6980v8.pdf
  Implementation based on code in https://gist.github.com/Newmu/acb738767acb4788bac3
  '''
  
  def __init__(self, learningRate=0.01, adam_b1=0.1, adam_b2=0.001):
    '''
    :type  learningRate: float
    :param learningRate: how big a step to take each epoch
    
    :type  adam_b1: float
    :param adam_b1: 1 - decay rate for first moment estimate
    
    :type  adam_b2: float
    :param adam_b2: 1 - decay rate for second moment estimate
    '''
    self.learningRate = theano.shared(np.array(learningRate).astype(theano.config.floatX))
    
    self.adam_b1 = theano.shared( np.array(adam_b1).astype(theano.config.floatX) )
    self.adam_b2 = theano.shared( np.array(adam_b2).astype(theano.config.floatX) )
    
    self.__adam_i   = theano.shared( np.array(0.0).astype(theano.config.floatX) )
    self.__adam_i_t = self.__adam_i + np.array(1.0).astype(theano.config.floatX)
    
    self.__moments_m = None
    self.__moments_v = None
  
  def getUpdates(self, params, grads):
    updates = []
    
    # Moment bias correction
    fix1 = 1. - (1. - self.adam_b1)**self.__adam_i_t
    fix2 = 1. - (1. - self.adam_b2)**self.__adam_i_t
    lr_t = self.learningRate * (T.sqrt(fix2)/fix1)
    
    updates = []
    
    if (self.__moments_m is None) or (self.__moments_v is None):
      self.__moments_m = []
      self.__moments_v = []
      
      for p, g in zip(params, grads):
        self.__moments_m.append( theano.shared(p.get_value() *
                                               np.array(0.).astype(theano.config.floatX),
                                 allow_downcast=True) )
        self.__moments_v.append( theano.shared(p.get_value() *
                                               np.array(0.).astype(theano.config.floatX),
                                 allow_downcast=True) )
      
      for p, g, adam_m, adam_v in zip(params, grads, self.__moments_m, self.__moments_v):
        adam_m_t = (self.adam_b1 * g) + ((1. - self.adam_b1) * adam_m)
        adam_v_t = (self.adam_b2 * T.sqr(g)) + ((1. - self.adam_b2) * adam_v)
        step = lr_t * adam_m_t / (T.sqrt(adam_v_t) + np.float32(1.e-8))
        
        updates.append((adam_m, adam_m_t))
        updates.append((adam_v, adam_v_t))
        updates.append((p,  p + step))
      
    updates.append((self.__adam_i, self.__adam_i_t))
    
    return updates
  
  def reset(self):
    self.__adam_i.set_value(np.float32(0.0))
    
    for moments in [self.__moments_m, self.__moments_v]:
      if moments is not None:
        for moment in moments:
          moment.set_value( moment.get_value()*0.0 )
  
  def toJson(self):
    return json.dumps({'type':'adam',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'adam_b1':self.adam_b1.get_value(),
                                 'adam_b2':self.adam_b2.get_value()}})

class AdadeltaOptimizer(Optimizer):
  '''
  Adadelta optimization: https://arxiv.org/abs/1212.5701
  Based on code from: https://blog.wtf.sg/2014/08/28/implementing-adadelta/
  '''
  
  def __init__(self, learningRate=1.0, rho=0.95, eps=1.e-6):
    '''
    :type  learningRate: float
    :param learningRate: how big a step to take each epoch
    
    :type  rho: float
    :param rho: interpolate between previous and current gradient
    
    :type  eps: float
    :param eps: small value to avoid divide by zero
    '''
    self.learningRate = theano.shared(np.array(learningRate).astype(theano.config.floatX))
    
    self.rho = theano.shared( np.array(rho).astype(theano.config.floatX) )
    self.eps = theano.shared( np.array(eps).astype(theano.config.floatX) )
    
    self.__gradsq_i   = None
    self.__gradsq_i_t = None
    
    self.__deltas_i    = None
    self.__deltas_sq   = None
    self.__deltas_sq_t = None
  
  def getUpdates(self, params, grads):
    updates = []
    
    #import pdb; pdb.set_trace()
    self.__gradsq_i = [theano.shared(np.zeros(
      p.get_value().shape).astype(theano.config.floatX),
                                     broadcastable=p.broadcastable)
                       for p in params]
    self.__gradsq_i_t = [self.rho*g_sq + (1.0-self.rho)*(g**2.)
                         for g_sq, g in zip(self.__gradsq_i, grads)]
    
    self.__deltas_sq = [theano.shared(np.zeros(p.get_value().shape).
                                      astype(theano.config.floatX),
                                      broadcastable=p.broadcastable) for p in params]
    self.__deltas_i = [self.learningRate *
                       (T.sqrt(d_sq+self.eps)/T.sqrt(g_sq+self.eps))*grad
                       for d_sq, g_sq, grad
                       in zip(self.__deltas_sq, self.__gradsq_i_t, grads)] # Step we take
    self.__deltas_sq_t = [self.rho*d_sq + (1.-self.rho)*(d**2.)
                          for d_sq, d in zip(self.__deltas_sq, self.__deltas_i)]
    
    updates += zip(self.__gradsq_i, self.__gradsq_i_t)
    updates += zip(self.__deltas_sq, self.__deltas_sq_t)
    updates += [(p, p + d) for p, d in zip(params, self.__deltas_i)]
    
    return updates
  
  def reset(self):
    self.__gradsq_i  = [0. * g for g in self.__gradsq_i]
    self.__deltas_sq = [0. * d for d in self.__deltas_sq]
  
  def toJson(self):
    return json.dumps({'type':'adadelta',
                       'params':{'learningRate':self.learningRate.get_value(),
                                 'rho':self.rho.get_value(),
                                 'eps':self.eps.get_value()}})

def jsonToOpt(jsonStr):
  ''' Build optimizer from JSON string. '''
  
  optObj = json.loads(jsonStr)
  
  optType   = optObj['type']
  optParams = optObj['params']
  
  TYPE_TO_CONS = {'sgd':SGDOptimizer,
                  'sgd_momentum':MomentumSGDOptimizer,
                  'adam':AdamOptimizer,
                  'adadelta':AdadeltaOptimizer}
  
  if optType in TYPE_TO_CONS:
    return TYPE_TO_CONS[optType](**optParams)
  else:
    raise Exception('Optimizer type "%s" not in {%s}' % (optType,
                                                        ' '.join(list(TYPE_TO_CONS.keys()))))

