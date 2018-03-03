'''
LDA where priors are defined by Theano function of alpha.

Adrian Benton
'''

import argparse, logging, os, sys, time

from logging import info, warn, error

import numpy as np

import utils
from utils import timeit, EPS, getLeakyRelu

import opt, mlp
from neural_prior import LdaCounts, NeuralPrior, buildSpriteParams

from _sample import _sample_topics, _loglikelihood_marginalizeTopics, _agg_samples

import theano.tensor as T
import h5py

from functools import reduce

NUM_RANDS = 1048583

def initLogger(logPath):
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
  rootLogger = logging.getLogger()
  
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setFormatter(logFormatter)
  rootLogger.addHandler(consoleHandler)
  
  fileHandler = logging.FileHandler(logPath)
  fileHandler.setFormatter(logFormatter)
  rootLogger.addHandler(fileHandler)

class NeuralLDA:
  def __init__(self, K, V, prior, heldoutIndex=2, seed=12345, numThreads=1):
    '''
    
    Parameters
    ----------
    K : int
      Number of topics to learn
    V : int
      Vocabulary size
    prior : NeuralPrior
      Handles updating of theta and phi priors
    heldoutIndex : int
      Index of tokens that will be heldout to evaluate log-likelihood.  If None,
      then we don't evaluate heldout LL.  If 2, then half of tokens are heldout.
    seed : int
      Random seed.
    numThreads : int
      How many threads for Gibbs sampling.
    '''
    
    self.K = K
    self.V = V
    
    self.prior = prior
    
    self.heldoutIndex = heldoutIndex
    
    if seed:
      rng = np.random.RandomState(seed)
    else:
      rng = np.random.RandomState()
    
    self._rands = rng.rand(NUM_RANDS).astype(np.float32)
    
    self.numThreads = numThreads
  
  def setData(self, Ws, Ds, docLabels):
    '''
    Set corpus to fit model to.
    
    Parameters
    ----------
    Ws : [ int ]
      List of word indexes
    Ds : [ int ]
      List of document indexes
    docLabels : numpy array
      annotations for each document
    '''
    
    self.Ws = Ws.astype(np.int32)
    self.Ds = Ds.astype(np.int32)
    self.Zs = np.array(np.random.randint(0, self.K, self.Ws.shape), dtype=np.intc)
    
    if self.heldoutIndex is None:
      self.Ws_heldout = np.array([], dtype=np.intc)
      self.Ds_heldout = np.array([], dtype=np.intc)
      self.Zs_heldout = np.array([], dtype=np.intc)
    else:
      # Move elements from train to heldout set
      self.Ws_heldout = np.array([w for i,w in enumerate(self.Ws)
                                     if ((i+1)%self.heldoutIndex)==0], dtype=np.intc)
      self.Ds_heldout = np.array([d for i,d in enumerate(self.Ds)
                                     if ((i+1)%self.heldoutIndex)==0], dtype=np.intc)
      self.Zs_heldout = np.array([z for i,z in enumerate(self.Zs)
                                     if ((i+1)%self.heldoutIndex)==0], dtype=np.intc)
      
      self.Ws = np.array([w for i,w in enumerate(self.Ws)
                                     if ((i+1)%self.heldoutIndex)!=0], dtype=np.intc)
      self.Ds = np.array([d for i,d in enumerate(self.Ds)
                                     if ((i+1)%self.heldoutIndex)!=0], dtype=np.intc)
      self.Zs = np.array([z for i,z in enumerate(self.Zs)
                                     if ((i+1)%self.heldoutIndex)!=0], dtype=np.intc)
    
    self.N = self.Ws.shape[0]  # total number of tokens
    self.M = self.Ds.max() + 1 # number of documents
    
    # Initialize counts
    self._nd  = np.zeros( (self.M, 1), dtype=np.intc) # Words per doc
    self._nz  = np.zeros( (self.K, 1), dtype=np.intc) # Number topics
    self._ndz = np.zeros( (self.M, self.K), dtype=np.intc) # Topic counts per doc
    self._nzw = np.zeros( (self.K, self.V), dtype=np.intc) # Topic counts per word
    
    for d, w, z in zip(self.Ds, self.Ws, self.Zs):
      self._nd[d] += 1
      self._nz[z] += 1
      self._ndz[d,z] += 1
      self._nzw[z,w] += 1
    
    self.docLabels = docLabels
    
    counts = LdaCounts(self._nd, self._nz, self._ndz, self._nzw)
    self.prior.setCorpus(self.docLabels, counts)
    self._thetaTilde, self._thetaNorm, self._phiTilde, self._phiNorm = self.prior.getPriors()
  
  def computePerplexity(self, isHeldout):
    return self.getPerplexity(self.computeLL_cython(isHeldout))
  
  def getPerplexity(self, ll):
    return 2.0**(-ll/self.N)
  
  def computeLL_cython(self, isHeldout):
    '''
    Calculate log-likelihood over train/heldout set.  
    '''
    
    thetaDenom = (1./(self._nd[:,0] + self._thetaNorm[:,0])).astype(np.float32)
    phiDenom   = (1./(self._nz[:,0] + self._phiNorm[:,0])).astype(np.float32)
    
    if isHeldout:
      Ds, Ws = (self.Ds_heldout, self.Ws_heldout)
    else:
      Ds, Ws = (self.Ds, self.Ws)
    
    return _loglikelihood_marginalizeTopics(Ds, Ws, self._nzw, self._ndz,
                                            self._nz[:,0], self._nd[:,0],
                                            self._thetaTilde, self._phiTilde, thetaDenom,
                                            phiDenom, np.asarray(self.numThreads, dtype=np.int32))
  
  def _sample_cython(self, iteration):
    '''
    Resample topics.  Quick: samples couple million tokens each second.
    '''
    
    randOffset = np.array((self.N*iteration) % (NUM_RANDS), dtype=np.int32)
    
    if self.numThreads > 1:
      _sample_topics_omp(self.Ws, self.Ds, self.Zs, self._nzw, self._ndz,
                         self._nz[:,0], self._thetaTilde, self._phiTilde, self._phiNorm[:,0],
                         self._rands, randOffset, self.numThreads)
    else:
      _sample_topics(self.Ws, self.Ds, self.Zs, self._nzw, self._ndz,
                     self._nz[:,0], self._thetaTilde, self._phiTilde, self._phiNorm[:,0],
                     self._rands, randOffset)
  
  def gradientStep(self):
    '''
    Update neural hyperparameters.
    '''
    
    self.prior.step()
    
    self._thetaTilde, self._thetaNorm, self._phiTilde, self._phiNorm = self.prior.getPriors()
  
  def learn(self, iters=1000, burnin=-1, numSamples=100, llFreq=10, stepFreq=1):
    '''
    Update model parameters by Gibbs sampling.  Update hyperparameters via gradient descent
    after burnin period.
    
    Parameters
    ----------
    iters : int
      How many Gibbs sampling iterations to learn parameters for.
    burnin : int
      Burn-in iterations before updating hyperparameters.  If negative, then does not update
      hyperparameters.
    numSamples: int
      Number of samples to collect to estimate topic distribution per document
    llFreq: int
      How often to evaluate log-likelihood
    stepFreq: int
      How often to update hyperparameters
    '''
    
    samples = np.zeros((self.M, self.K), dtype=np.int32)
    
    self.printLL(-1)
    
    for i in range(iters):
      i = np.array(i, dtype=np.int32) # cython function expects C int
      
      self._sample_cython(i)
      
      # Collect final set of samples...
      if i >= (iters-numSamples):
        _agg_samples(self.Ds, self.Zs, self.Ds.shape[0], samples)
        #for d, z in zip(self.Ds, self.Zs):
        #  samples[d,z] += 1.0
      
      if 0 <= burnin <= i:
        if (i % stepFreq) == 0:
          self.gradientStep()
      
      if (i % llFreq)==0:
        self.printLL(i)
    self.finalSamples = samples.astype(np.float32)
  
  def getTopics(self, includePrior=False):
    ''' Get word distribution per topic '''
    
    topics = np.zeros((self.K, self.V))
    
    if includePrior:
      topics += model._priorZW  # pseudocounts for each word
    
    for w, z in zip(self.Ws, self.Zs):
      topics[z,w] += 1.0
    
    return topics
  
  def getAnnotationIdxes(self):
    ''' Get index and values of document annotations. '''
    idxes = []
    
    for row in self.docLabels:
      if type(row.tolist()[0]) == float:
        idxVals = sorted(list(enumerate(row.tolist())), key=lambda x:x[1], reverse=True)
      else:
        idxVals = sorted(list(enumerate(row.tolist()[0])), key=lambda x:x[1], reverse=True)
      
      idxes.append([(i,v) for i,v in idxVals if abs(v)>0.])
    
    return idxes
  
  def getTopAnnotations(self, indexToAnnName, topn=5):
    ''' Get top N largest annotations for each document. '''
    
    annIdxes = self.getAnnotationIdxes()
    
    topN = []
    
    for doc in annIdxes:
      topN.append([(indexToAnnName[idx], value) for idx, value in doc[:topn]])
    
    return topN
  
  def getDocRepresentations(self):
    ''' Get transformed versions of alpha. '''
    
    docRepresentations = self.prior._alphaObj.get_output(self.docLabels, 0)
    return docRepresentations
  
  def getTopWords(self, indexToWord, topn=20, includePrior=False):
    ''' Get representative words for each topic. '''
    
    topics = self.getTopics(includePrior)
    
    topNWords = []
    for z in range(self.K):
      topic = [(wLogProb, w) for w, wLogProb in enumerate(topics[z,:])]
      topic.sort(reverse=True)
      topic = topic[:topn]
      topNWords.append([(indexToWord[w], p) for p, w in topic])
    
    return topNWords
  
  def printTopWords(self, indexToWord, topn=20, includePrior=False):
    topNWords = self.getTopWords(indexToWord, topn, includePrior)
    
    for tidx, topic in enumerate(topNWords):
      print('Topic %d:' % (tidx), ' '.join([w for w,p in topic]))
  
  def printLL(self, i):
    if i <= 0:
      self.__startTime = time.time()
      self.ll_history  = []
    
    if self.heldoutIndex is not None:
      train_ll    = self.computeLL_cython(False)
      train_ppl   = self.getPerplexity(train_ll)
      heldout_ll  = self.computeLL_cython(True)
      heldout_ppl = self.getPerplexity(heldout_ll)
      print('Iter %d, %ds, Train: (LL %.3f, Perplexity %.3f), Heldout: (LL %.3f, Perplexity %.3f)' % (i, time.time() - self.__startTime, train_ll, train_ppl, heldout_ll, heldout_ppl))
      self.ll_history.append({'train_ll':train_ll, 'train_ppl':train_ppl,
                              'heldout_ll':heldout_ll, 'heldout_ppl':heldout_ppl,
                              'iteration':i})
    else:
      ll  = self.computeLL_cython(False)
      ppl = self.getPerplexity(ll)
      print('Iter %d, %ds, Train: (LL %.3f, Perplexity %.3f)' %
            (i, time.time() - self.__startTime, ll, ppl))
      self.ll_history.append({'train_ll':ll, 'train_ppl':ppl, 'iteration':i})
    
    for handler in logging.getLogger().handlers:
      handler.flush()
  
  def serialize(self, saveCorpus=False):
    payload = {}
    
    payload['K'] = self.K
    payload['V'] = self.V
    payload['heldout_index'] = self.heldoutIndex
    payload['topics'] = self.getTopics()
    
    for wt in self.prior._params.wts:
      payload[wt.name + '_wts'] = wt.get_value()
    
    payload['alpha_wts'] = self.prior._alphaObj.getWeights()
    
    if saveCorpus:
      #payload['Ds'] = self.Ds
      #payload['Ds_heldout'] = self.Ds_heldout
      #payload['Ws'] = self.Ws
      #payload['Ws_heldout'] = self.Ws_heldout
      #payload['Zs'] = self.Zs
      #payload['Zs_heldout'] = self.Zs_heldout
      payload['doc_labels'] = self.docLabels
      payload['final_samples'] = self.finalSamples
      payload['doc_representations'] = self.getDocRepresentations()
    
    return payload

def strToActFn(keyword):
  kwToFn = {'linear':None, 'leakyrelu_0.1':getLeakyRelu(0.1),
            'leakyrelu_0.5':getLeakyRelu(0.5), 'relu':T.nnet.relu,
            'sigmoid':T.nnet.sigmoid,
            'tanh':T.tanh, 'softmax':T.nnet.softmax}
  
  if keyword in kwToFn:
    return kwToFn[keyword]
  else:
    print('I do not recognize "%s" activation... using linear activation function.' % (keyword))
    return None

def test(inPath, outPath, Z=5, iters=1000, llFreq=20, annName='descriptor', arch=[], tieBetaDelta=False, hiddenActivation='linear', outputActivation='linear', L1=0.0, L2=0.1, numThreads=1, input_params={}, netType='mlp', stepFreq=1, minibatch=-1, annotationPath=None, maxExamples=1000000000, stepSize=0.5):
  initLogger(outPath+'.log')
  
  d = np.load(inPath)
  
  if annotationPath is None:
    if annName is None:
      annDicts = d['annotation_dicts'].item()
      arbitraryKey = list(annDicts.keys())[0]
      annotations    = d['annotations'].item()[arbitraryKey].astype(np.float32)
      annotationDict = d['annotation_dicts'].item()[arbitraryKey]
    else:
      annotations    = d['annotations'].item()[annName].astype(np.float32)
      annotationDict = d['annotation_dicts'].item()[annName]
  else:
    f = h5py.File(annotationPath) # load annotations from hdf5-formatted file,
                                  # too big for numpy array
    if annName is None:
      annotations    = f['final-layer_image'].value.astype(np.float32)
      annotationDict = d['annotation_dicts'].item()['final-layer_image']
    else:
      annotations    = f[annName].value.astype(np.float32)
      annotationDict = d['annotation_dicts'].item()[annName]
    
    f.close()
  
  annotations = annotations[:maxExamples] # only keep this many documents
  
  tokenDict   = d['token_dict'].item()
  
  print('Loaded data')
  
  Cin  = reduce(lambda x,y:x*y, [annotations.shape[i] for i in range(1, len(annotations.shape))])
  if netType == 'mlp' and (annName is None or len(arch) == 0):
    Cout = Cin
  else:
    Cout = arch[-1]
  
  Ds, Ws = d['Ds_body'], d['Ws_body']
  D = Ds.max()+1
  W = Ws.max()+1
  
  # drop some documents, restrict to subset
  if D >= maxExamples:
    isMaxDoc  = Ds==(maxExamples-1)
    maxDocIdx = Ds.shape[0] - (isMaxDoc[::-1]).argmax()
    Ds, Ws = Ds[:maxDocIdx], Ws[:maxDocIdx]
  
  D = Ds.max()+1
  W = Ws.max()+1
  
  # Initalize network parameters
  
  betaMean, betaL1, betaL2 = 0.0, L1, L2
  deltaMean, deltaL1, deltaL2 = 0.0, L1, L2
  omegaMean, omegaL1, omegaL2 = 0.0, L1, L2
  
  # Bias weights have small, fixed L2 regularization.
  deltaBMean, deltaBL1, deltaBL2 = -1.0, 0.0, 0.1
  omegaBMean, omegaBL1, omegaBL2 = -2.0, 0.0, 0.1
  
  params = buildSpriteParams(Cout, W, Z, betaMean, betaL1, betaL2, deltaMean, deltaL1,
                             deltaL2, omegaMean, omegaL1, omegaL2, deltaBMean, deltaBL1,
                             deltaBL2, omegaBMean, omegaBL1, omegaBL2,
                             tieBetaAndDelta=tieBetaDelta)
  
  optimizer = opt.AdadeltaOptimizer(-stepSize, rho=0.95)
  #optimizer = opt.MomentumSGDOptimizer(-stepSize, momentum=0.5)
  
  if netType == 'mlp':
    alphaGraph = mlp.MLP(12345, [Cin] + arch,
                         strToActFn(hiddenActivation), strToActFn(outputActivation),
                         optimizer, L1, L2, 'alphaGraph-' + str(annName))
  #elif netType == 'squeezenet':
  #  alphaGraph = Squeezenet(8) # keep first 8 fire modules fixed, tune last 2
  #elif netType == 'vgg16':
  #  alphaGraph = VGG() # Final convolution and dense layers for VGG model
  else:
    raise Exception('Don\'t recognize network type "%s"' % (netType))
  
  if annName is None:
    prior = NeuralPrior(params, alphaGraph, optimizer=optimizer, onlyBias=True)
  else:
    prior = NeuralPrior(params, alphaGraph, optimizer=optimizer, onlyBias=False,
                        mbatchsize=minibatch)
  
  model = NeuralLDA(Z, W, prior, heldoutIndex=2, seed=12345,
                    numThreads=np.asarray(numThreads, dtype=np.int32))
  model.setData(Ws, Ds, annotations)
  
  print('Built model')
  
  model.learn(iters=iters, burnin=100, numSamples=100, llFreq=llFreq, stepFreq=stepFreq)
  
  # Print out topics
  model.printTopWords(tokenDict)
  
  # Save things we may care about later...
  payload = model.serialize(saveCorpus=True)
  
  payload['token_dict']  = tokenDict
  payload['topics']      = model.getTopWords(tokenDict)
  #payload['annotations'] = model.getTopAnnotations(annotationDict)
  
  omegaBTopic, supertopics = model.prior.getSupertopics(tokenDict, topn=20)
  payload['omegaBias_supertopic'] = omegaBTopic
  payload['omega_supertopics']    = supertopics
  payload['likelihood_history']   = model.ll_history
  payload['input_params']         = input_params
  
  if 'preprocess_params' in d:
    payload['preprocess_params']  = d['preprocess_params'].item()
  elif 'params' in d:
    payload['preprocess_params']  = d['params'].item()
    
  #payload['all_annotations']      = d['annotations'].item()
  #payload['all_annotation_dicts'] = d['annotation_dicts'].item()
  
  if 'fold' in d:
    payload['fold'] = d['fold']
  
  np.savez_compressed(outPath, **payload)

  # Save individual topic samples to file
  outFile = open(outPath + '.samples.txt', 'wt')
  lastD = 0
  for d, w, z in zip(model.Ds, model.Ws, model.Zs):
    if d > lastD:
      outFile.write('\n')
      lastD += 1
    else:
      outFile.write('%s:%d ' % (tokenDict[w], z))
  outFile.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='learn neural Sprite model')
  parser.add_argument('-i', '--input', required=True, metavar='INPUT',
                      help='path to input npz file')
  parser.add_argument('-o', '--output', required=True, metavar='OUTPUT',
                      help='base path to output weights and log file')
  parser.add_argument('-a', '--annotation', default=None,
                      help='which set of annotations to use -- if not set, then just trains LDA model')
  parser.add_argument('--Z', default=5, type=int, help='number of topics to learn')
  parser.add_argument('--nettype', default='mlp', choices=['mlp', 'squeezenet', 'vgg16'],
                      help='network type  to use as a prior')
  parser.add_argument('--arch', default=[], nargs='+', type=int,
                      help='width of each hidden layer (input layer is implicit)')
  parser.add_argument('--tie_betadelta', default=False, action='store_true',
                      help='beta and delta weight vectors should be tied')
  parser.add_argument('--hidden_activation', default='linear', choices=['linear', 'leakyrelu_0.1',
                                                                        'leakyrelu_0.5', 'relu',
                                                                        'sigmoid', 'tanh'],
                      help='activation function for hidden layer')
  parser.add_argument('--output_activation', default='linear', choices=['linear', 'leakyrelu_0.1',
                                                                        'leakyrelu_0.5', 'relu',
                                                                        'sigmoid', 'tanh', 'softmax'],
                      help='activation function on the output layer')
  
  ################### Learning Params ###########################
  parser.add_argument('--maxexamples', default=1000000000, type=int,
                      help='maximum number of documents to load')
  parser.add_argument('--iters', default=1000, type=int,
                      help='number of iterations to train for')
  parser.add_argument('--nthreads', default=1, type=int,
                      help='number of Gibbs sampling threads')
  parser.add_argument('--llfreq', default=20, type=int,
                      help='how often to check log-likelihood')
  parser.add_argument('--L1', default=0.0, type=float,
                      help='l1 regularization for all hyperparameter weights')
  parser.add_argument('--L2', default=0.1, type=float,
                      help='l2 regularization for all hyperparameter weights')
  parser.add_argument('--stepfreq', default=1, type=int,
                      help='how often to update hyperparameters')
  parser.add_argument('--stepsize', default=0.5, type=float,
                      help='gradient step size')
  ###############################################################
  
  ################ CNN-specific Params ##########################
  parser.add_argument('--annotationpath', default=None,
                      help='path to annotations saved in hdf5 format')
  parser.add_argument('--mbatchsize', default=1000000, type=int,
          help='size of minibatches when updating network, for compute & space efficiency')
  ###############################################################
  
  ########################## TODO ###############################
  parser.add_argument('--configpath', default=None,
                      help='defines model architecture, etc.')
  parser.add_argument('--saved_modelpath', default=None,
                      help='rebuild and train model with same parameters as this serialized one')
  parser.add_argument('--pretrain_alpha', default=False, action='store_true',
                      help='pretrain alpha autoencoder style')
  ###############################################################
  
  args = parser.parse_args()
  input_params = args.__dict__
  
  nettype = args.nettype
  
  test(args.input, args.output, Z=args.Z, iters=args.iters, llFreq=args.llfreq,
       annName=args.annotation, arch=args.arch, tieBetaDelta=args.tie_betadelta,
       hiddenActivation=args.hidden_activation, outputActivation=args.output_activation,
       L1=args.L1, L2=args.L2, numThreads=args.nthreads, input_params=input_params,
       netType=args.nettype, stepFreq=args.stepfreq, minibatch=args.mbatchsize,
       annotationPath=args.annotationpath, maxExamples=args.maxexamples,
       stepSize=args.stepsize)
  
