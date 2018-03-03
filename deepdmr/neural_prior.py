'''
Keep track of document and topic priors.  Gradient updates performed by Theano.

Adrian Benton
'''

from functools import reduce

from scipy.special import digamma

import theano
from theano import config
import theano.tensor as T

import numpy as np

import mlp, opt

from _sample import _loglikelihood_marginalizeTopics

EPS=np.array(1.e-8).astype(np.float32)

def compLL(Ds, Ws, nzw, ndz, nz, nd, priorDZ, priorZW, thetaDenom, phiDenom):
  return _loglikelihood_marginalizeTopics(Ds, Ws, nzw.astype(np.int32),
                                          ndz.astype(np.int32), nz[:,0].astype(np.int32),
                                          nd[:,0].astype(np.int32), priorDZ,
                                          priorZW, 1./(nd[:,0]+thetaDenom[:,0]),
                                          1./(nz[:,0]+phiDenom[:,0]))

def buildSpriteParams(C, W, Z, betaMean, betaL1, betaL2, deltaMean, deltaL1, deltaL2, omegaMean, omegaL1, omegaL2, deltaBMean, deltaBL1, deltaBL2, omegaBMean, omegaBL1, omegaBL2, tieBetaAndDelta):
  betaInit = RegularizedWeights(np.random.normal(betaMean, 0.01, (Z, C)).
                                astype(np.float32), betaMean, betaL1, betaL2, 'beta')
  if tieBetaAndDelta:
    deltaTInit = betaInit
  else:
    deltaTInit = RegularizedWeights(np.random.normal(deltaMean, 0.01, (Z, C)).
                                    astype(np.float32), deltaMean, deltaL1, deltaL2, 'deltaT')
  
  omegaInit = RegularizedWeights(np.random.normal(omegaMean, 0.01, (C, W)).
                                 astype(np.float32), omegaMean, omegaL1, omegaL2, 'omega')
  
  omegaBInit = RegularizedWeights(np.random.normal(omegaMean, 0.01, (1,W)).
                                  astype(np.float32), omegaBMean, omegaBL1, omegaBL2, 'omegaB')
  
  deltaBInit = RegularizedWeights(np.random.normal(deltaBMean, 0.01, (1,Z)).
                                  astype(np.float32), deltaBMean, deltaBL1, deltaBL2, 'deltaB')
  
  return SpriteParams(betaInit, deltaTInit, omegaInit, omegaBInit, deltaBInit)

class RegularizedWeights:
  def __init__(self, initWeights, meanWeight, l1=0.0, l2=0.0, name=''):
    bcast = (True, False) if initWeights.shape[0]==1 else (False, False)
    
    self.wts  = theano.shared(initWeights.astype(np.float32), borrow=True,
                              broadcastable=bcast, name=name)
    self._mean = (meanWeight *
                  np.ones(initWeights.shape, dtype=np.float32)).astype(np.float32)
    self._l1   = np.array(l1).astype(np.float32)
    self._l2   = np.array(l2).astype(np.float32)
    
    self.reg_cost = self._l1 * T.sum(abs(self.wts-self._mean)) + \
                    self._l2 * T.sum((self.wts-self._mean)**2.)
    self.reg_grad = T.grad(self.reg_cost, self.wts)

class SpriteParams:
  '''
  Keep track of Sprite weights and regularization terms -- excludes the neural network for
  alpha.
  '''
  
  def __init__(self, beta, deltaT, omega, omegaB, deltaB):
    self.beta   = beta
    self.deltaT = deltaT
    self.omega  = omega
    self.omegaB = omegaB
    self.deltaB = deltaB
    
    # collect all weights together
    self.wts = [self.beta.wts, self.deltaT.wts,
                self.omega.wts, self.omegaB.wts,
                self.deltaB.wts]
    self.reg_cost = sum([self.beta.reg_cost, self.deltaT.reg_cost,
                         self.omega.reg_cost, self.omegaB.reg_cost,
                         self.deltaB.reg_cost])

class LdaCounts:
  def __init__(self, nd, nz, ndz, nzw):
    # borrowed because gibbs sampling step updates the counts
    
    self._nd  = theano.shared(nd,  borrow=True, name='Nd', broadcastable=(False, True))
    self._nz  = theano.shared(nz,  borrow=True, name='Nz', broadcastable=(False, True))
    self._ndz = theano.shared(ndz, borrow=True, name='Ndz')
    self._nzw = theano.shared(nzw, borrow=True, name='Nzw')

class NeuralPrior:
  '''
  Factor where document labels influence priors through a neural network.  Computes document
  and topic priors.
  '''
  
  def __init__(self, params, alphaGraph, optimizer=opt.SGDOptimizer(0.01), onlyBias=False, mbatchsize=1000000):
    self._params = params
    self.optimizer = optimizer
    
    self._alpha    = alphaGraph.output
    self._alphaObj = alphaGraph
    self._nlayers  = alphaGraph.nlayers
    self.__docLabels      = alphaGraph.input
    self.numComponents    = alphaGraph.outWidth
    self.inNumComponents  = alphaGraph.inWidth
    self.__alpha_reg_cost = alphaGraph.getRegCost()
    self.__alpha_params   = alphaGraph.getParams()
    
    self.mbatchsize = mbatchsize
    self._onlyBias = onlyBias
    
    # zero out weights that are not bias terms
    if self._onlyBias:
      for w in self._params.wts:
        if w.name not in set(['omegaB', 'deltaB']):
          w.set_value(np.array(0.).astype(np.float32) * w.get_value())
  
  def setCorpus(self, alphaValues, counts):
    '''
    alphaValues: this factor is observed, set document labels
    '''
    # shared variables with topic/word counts
    self._nd  = counts._nd
    self._nz  = counts._nz
    self._ndz = counts._ndz
    self._nzw = counts._nzw
    
    self._alphaValues = alphaValues
    M, K, V = (alphaValues.shape[0],
               self._params.deltaB.wts.get_value().shape[1],
               self._params.omegaB.wts.get_value().shape[1])
    
    self._batch_index = theano.shared(0)
    
    self.__thetaTilde = theano.shared(np.zeros((M, K)).astype(np.float32),
                                      borrow='True')
    self.__phiTilde   = theano.shared(np.zeros((K, V)).astype(np.float32),
                                      borrow='True')
    self.__thetaNorm  = theano.shared(np.zeros((M,1)).astype(np.float32),
                                      borrow='True', broadcastable=(False, True))
    self.__phiNorm    = theano.shared(np.zeros((K,1)).astype(np.float32), borrow='True',
                                      broadcastable=(False, True))
    
    self._tt = self.__thetaTilde
    self._pt = self.__phiTilde
    self._tn = self.__thetaNorm
    self._pn = self.__phiNorm
    
    self._buildGraph()
  
  def _buildGraph(self):
    # Document and topic priors
    self.thetaTilde_next = T.exp(self._alpha.dot(self._params.deltaT.wts.T) +
                                 self._params.deltaB.wts)
    self.phiTilde_next   = T.exp(self._params.beta.wts.dot(self._params.omega.wts) +
                       self._params.omegaB.wts)
    self.thetaNorm_next = T.sum(self.thetaTilde_next, axis=1, keepdims=True)
    self.phiNorm_next   = T.sum(self.phiTilde_next, axis=1, keepdims=True)
    
    self.__priorUpdates = [(self.__thetaTilde,
                            T.set_subtensor(self.__thetaTilde[self._batch_index*self.mbatchsize:
                                                              ((self._batch_index+1)*self.mbatchsize)],
                                            self.thetaTilde_next)),
                           (self.__phiTilde, self.phiTilde_next),
                           (self.__thetaNorm,
                            T.set_subtensor(self.__thetaNorm[self._batch_index*self.mbatchsize:
                                                             ((self._batch_index+1)*self.mbatchsize)],
                                            self.thetaNorm_next)),
                           (self.__phiNorm, self.phiNorm_next)]
    
    #self._updatePriorFns = [theano.function(inputs=inputs,
    #                                         outputs=[], updates=[update])
    #                         for i, (inputs, update)
    #                         in enumerate(zip([[self.__docLabels], [], [self.__docLabels], []],
    #                                          self.__priorUpdates))]
    self.__updatePriors = theano.function(inputs=[self.__docLabels, self._alphaObj._dummy],
                                          outputs=[], updates=self.__priorUpdates)
    
    self.__getPriors = theano.function(inputs=[self.__docLabels, self._alphaObj._dummy],
                                       outputs=[self.thetaTilde_next,
                                                self.thetaNorm_next,
                                                self.phiTilde_next,
                                                self.phiNorm_next])
    
    # Calculate gradient w.r.t. prior
    self.buildExtGradFn()
    
    # Function to update model weights
    self.buildStepFn()
    
    # Calculate current value of priors
    self.__updatePriors(self._alphaValues, 0)
  
  def getPriors(self):
    return self.__thetaTilde.get_value(), self.__thetaNorm.get_value(), self.__phiTilde.get_value(), self.__phiNorm.get_value()
  
  def step(self):
    ''' Take a gradient step, updating all parameters. '''
    
    gradTerm_phi, gradTerm_theta = self.calcExternalGrad()
    
    if self._onlyBias:
      self.take_step_bias(self._alphaValues, gradTerm_phi, gradTerm_theta, 0)
    else:
      self.take_step(self._alphaValues, gradTerm_phi, gradTerm_theta, 0)
  
  def stepSprite(self):
    ''' Take a gradient step, updating only SPRITE parameters, not alpha '''
    gradTerm_phi, gradTerm_theta = self.calcExternalGrad()
    if self.mbatchsize <= 0:
      self.take_step_sprite(self._alphaValues, gradTerm_phi, gradTerm_theta, 0)
    else:
      # update prior in minibatches
      N = self._alphaValues.shape[0]
      batches = (N // self.mbatchsize) + 1
      for i in range(batches):
        if i*self.mbatchsize >= N:
          continue
        else:
          self._batch_index.set_value(i)
          self.take_step_sprite(self._alphaValues,
                                gradTerm_phi,
                                gradTerm_theta, 0)
  
  def stepAlpha(self):
    ''' Take a gradient step, updating only alpha network weights '''
    gradTerm_phi, gradTerm_theta = self.calcExternalGrad()
    if self.mbatchsize <= 0:
      self.take_step_alpha(self._alphaValues, gradTerm_phi, gradTerm_theta, 0)
    else:
      # update prior in minibatches
      N = self._alphaValues.shape[0]
      batches = (N // self.mbatchsize) + 1
      for i in range(batches):
        if i*self.mbatchsize >= N:
          continue
        else:
          self._batch_index.set_value(i)
          self.take_step_alpha(self._alphaValues,
                                gradTerm_phi,
                                gradTerm_theta, 0)
  
  def buildExtGradFn(self):
    tn, tt, pn, pt = self.__thetaNorm, self.__thetaTilde, self.__phiNorm, self.__phiTilde
    
    dg1  = T.psi(self.__thetaNorm + EPS)
    dg2  = T.psi(self.__thetaNorm + T.cast(self._nd, 'float32') + EPS)
    dgW1 = T.psi(self.__thetaTilde + T.cast(self._ndz, 'float32') + EPS)
    dgW2 = T.psi(self.__thetaTilde + EPS)
    gradTerm_theta = dg1 - dg2 + dgW1 - dgW2
    
    dg1  = T.psi(self.__phiNorm + EPS)
    dg2  = T.psi(self.__phiNorm + T.cast(self._nz, 'float32') + EPS)
    dgW1 = T.psi(self.__phiTilde + T.cast(self._nzw, 'float32') + EPS)
    dgW2 = T.psi(self.__phiTilde + EPS)
    gradTerm_phi = dg1 - dg2 + dgW1 - dgW2
    
    self.calcExternalGrad_phi   = theano.function(inputs=[],
                                                  outputs=[gradTerm_phi])
    self.calcExternalGrad_theta = theano.function(inputs=[],
                                                  outputs=[gradTerm_theta])
    self.calcExternalGrad = theano.function(inputs=[],
                                            outputs=[gradTerm_phi, gradTerm_theta])
  
  def buildStepFn(self):
    # the gradient of likelihood w.r.t. topic and document priors
    gradTerm_phi   = T.matrix('gradientTerm_phi')
    gradTerm_theta = T.matrix('gradientTerm_theta')
    
    # Total cost
    extGradCost_phi   = - T.sum( self.phiTilde_next   * gradTerm_phi )
    extGradCost_theta = - T.sum( self.thetaTilde_next * gradTerm_theta )
    extGradCost = extGradCost_phi + extGradCost_theta
    reg_extGradCost = extGradCost + self.__alpha_reg_cost + self._params.reg_cost
    
    #reg_extGradCost = - reg_extGradCost
    
    # Collect SPRITE parameters together with the MLP weights
    __params_bias   = [w for w in self._params.wts if w.name=='deltaB' or w.name=='omegaB']
    __params_sprite = self._params.wts
    __params_alpha  = self.__alpha_params
    __params = __params_sprite + __params_alpha
    
    # Calculate gradient with respect to this cost function
    __grads_bias = [T.grad(reg_extGradCost, p) for p in __params_bias]
    __grads_sprite = [T.grad(reg_extGradCost, p) for p in __params_sprite]
    __grads_alpha  = [T.grad(reg_extGradCost, p) for p in __params_alpha]
    __grads = [T.grad(reg_extGradCost, p) for p in __params]
    
    __updates_bias   = self.optimizer.getUpdates(__params_bias, __grads_bias)
    __updates_sprite = self.optimizer.getUpdates(__params_sprite, __grads_sprite)
    __updates_alpha  = self.optimizer.getUpdates(__params_alpha, __grads_alpha)
    __updates = self.optimizer.getUpdates(__params, __grads)
    
    # update all model weights, and recalculate doc/topic priors
    def buildFn(updates):
      if updates:
        # To remove duplicates when beta&delta are tied
        updatesWoDups = list({k:v for (k,v) in updates}.items())
        
        return theano.function([ self.__docLabels, gradTerm_phi,
                                 gradTerm_theta, self._alphaObj._dummy],
                               outputs=[],
                               updates=updatesWoDups + self.__priorUpdates)
      else:
        return theano.function([ ], outputs=[])
    
    # update weights
    self.take_step_bias   = buildFn(__updates_bias)   # only bias terms
    
    self.take_step_sprite = buildFn(__updates_sprite) # only SPRITE parameters
    if self._nlayers > 0:
      self.take_step_alpha  = buildFn(__updates_alpha)  # only alpha network
    else:
      self.take_step_alpha = lambda x,y,z: []
    
    self.take_step = buildFn(__updates) # update all parameters
    
    self.__get_reg_cost = theano.function([], outputs=[self.__alpha_reg_cost +
                                                     self._params.reg_cost])
    self.__get_theta_cost = theano.function([self.__docLabels, gradTerm_theta,
                                             self._alphaObj._dummy],
                                            outputs=[extGradCost_theta])
    self.__get_phi_cost = theano.function([gradTerm_phi],
                                          outputs=[extGradCost_phi])
    self.__get_total_cost = theano.function([self.__docLabels, gradTerm_phi,
                                             gradTerm_theta, self._alphaObj._dummy],
                                            outputs=[reg_extGradCost])
    self.__get_all_costs = theano.function([self.__docLabels, gradTerm_phi,
                                            gradTerm_theta, self._alphaObj._dummy],
                                            outputs=[self._params.reg_cost,
                                                     extGradCost_theta,
                                                     extGradCost_phi, reg_extGradCost])
  
  def getCosts(self):
    gradTerm_phi, gradTerm_theta = self.calcExternalGrad()
    return self.__get_all_costs(self._alphaValues, gradTerm_phi, gradTerm_theta)
  
  def getSupertopics(self, itow, topn=20):
    omegaBTopic = []
    supertopics = []
    
    omegab = self._params.wts[3].get_value()[0]
    
    omegaBTopic = [(itow[i], weight) for i, weight in
                   sorted(enumerate(omegab), key=lambda x:x[1], reverse=True)[:topn]]
    
    omega = self._params.wts[2].get_value()
    for omegaRow in omega:
      topicPos = [(itow[i], weight) for i, weight in
                  sorted(enumerate(omegaRow), key=lambda x:x[1], reverse=True)[:topn]]
      topicNeg = [(itow[i], weight) for i, weight in
                  sorted(enumerate(omegaRow), key=lambda x:x[1])[:topn]]
      supertopics.append(topicPos + topicNeg)
    
    return omegaBTopic, supertopics
  
  def printSupertopics(self, itow, topn=20):
    omegaBTopic, supertopics = self.getSupertopics(itow, topn)
    
    print('==== Omega Bias ====')
    for word, weight in omegaBTopic:
      print('%s: %s' % (word, weight))
    print('')
    
    for topicIdx, topic in enumerate(supertopics):
      print('=== (Omega) Supertopic %d ===' % (topicIdx))
      for word, weight in topic:
        print('%s: %s' % (word, weight))
      print('')

def test():
  Cin, Cout, W, Z, D, Dn = 20, 4, 50, 2, 1000, 10
  
  L2 = 1.0
  
  betaMean, betaL1, betaL2 = 0.0, 0.0, L2
  deltaMean, deltaL1, deltaL2 = 0.0, 0.0, L2
  omegaMean, omegaL1, omegaL2 = 0.0, 0.0, L2
  deltaBMean, deltaBL1, deltaBL2 = -2.0, -2.0, L2
  omegaBMean, omegaBL1, omegaBL2 = -4.0, -4.0, L2
  
  params = buildSpriteParams(Cout, W, Z, betaMean, betaL1, betaL2, deltaMean, deltaL1,
                             deltaL2, omegaMean, omegaL1, omegaL2, deltaBMean, deltaBL1,
                             deltaBL2, omegaBMean, omegaBL1, omegaBL2, tieBetaAndDelta=False)
  
  docLabels = np.asarray([[1.0*(j==(i%(Cin//2)) and (j < (Cin//2))) for j in range(Cin)]
                          for i in range(D//2)] +
                         [[1.0*((j-(Cin//2))==(i%(Cin//2)) and (j >= (Cin//2)))
                           for j in range(Cin)]
                          for i in range(D//2,D)], dtype=np.float32)
  trueWts   = np.zeros((Cin, Cout))
  
  for c in range(Cout):
    trueWts[(c*(Cin//Cout)):((c+1)*(Cin//Cout)),c] = 10.0
  
  trueAlpha = docLabels.dot(trueWts).astype(np.float32)
  
  ndz = np.zeros((D, Z)).astype(np.float32)
  
  Ds  = np.asarray([d for d in range(D) for dn in range(Dn)]).astype(np.int32)
  Ws  = np.asarray([(idx % 25) for idx in range((D*Dn)//2)] +
                   [((idx % 25) + 25) for idx in range((D*Dn)//2, D*Dn)]).astype(np.int32)
  
  for z in range(Z):
    ndz[(z*(D//Z)):((z+1)*(D//Z)),z] = Dn
  ndz = ndz.astype(np.float32)
  
  nzw = np.zeros((Z, W))
  nzw[0,:(W//2)] = (D * Dn) // W
  nzw[1,(W//2):] = (D * Dn) // W
  nzw = nzw.astype(np.float32)
  
  nd = (Dn * np.ones((D, 1))).astype(np.float32)
  nz = ((D//Z) * Dn * np.ones((Z, 1))).astype(np.float32)
  
  optimizer = opt.AdadeltaOptimizer(-1.0)
  
  alphaGraph = mlp.MLP(12345, [Cin, Cout], None, None,
                       optimizer, 0.0, L2, 'alphaGraph')
  
  prior = NeuralPrior(params, alphaGraph,
                      optimizer=optimizer)
  prior.setCorpus(docLabels, LdaCounts(nd, nz, ndz, nzw))
  
  thetaTilde, thetaNorm, phiTilde, phiNorm = prior.getPriors()
  
  ll = compLL(Ds, Ws, nzw, ndz, nz, nd, thetaTilde, phiTilde, thetaNorm, phiNorm)
  costStr = ', '.join(map(str, prior.getCosts()))
  print('LL: %.3f, DeltaB: %s' % (ll, prior._params.wts[4].get_value()))
  print('Costs: ' + costStr)
  
  for epoch in range(2000):
    prior.step()
    thetaTilde, thetaNorm, phiTilde, phiNorm = prior.getPriors()
    
    ll = compLL(Ds, Ws, nzw, ndz, nz, nd, thetaTilde, phiTilde, thetaNorm, phiNorm)
    
    if not epoch % 1:
      print('Finished epoch %d, LL %.3f, DeltaB: %s' % (epoch, ll,
                                                        prior._params.wts[4].get_value()))
      costStr = ', '.join(map(str, prior.getCosts()))
      print('Costs: ' + costStr)

if __name__ == '__main__':
  test()
