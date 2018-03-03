'''
Validate deep DMR implementation on synthetic data.

Adrian Benton
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
sns.set(style="whitegrid")
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

import argparse, os, pickle, time
from copy import deepcopy
import numpy as np
import scipy.sparse
import theano
import theano.tensor as T

import mlp, opt
from neural_prior import buildSpriteParams, NeuralPrior
from lda_neural_prior import NeuralLDA

import pandas as pd

synth_output_dir = '../test_output/'

dataPathFmt  = os.path.join(synth_output_dir, 'data/synth_arch-%s_D-%s_Z-%d_Nd-%d_V-%d_noise-%f_activation-%s_outActivation-%s_run-%d.npz')
modelPathFmt = os.path.join(synth_output_dir, 'runs/synth_arch-%s_D-%s_Z-%d_Nd-%d_V-%d_noise-%f_activation-%s_outActivations-%s_model-%s_run-%d.npz')

def actToStr(fn):
  if fn is None:
    return 'None'
  else:
    try:
      return fn.name
    except:
      return fn.__name__

def genDataFromNetwork(architecture, D, Z, Nd, V, noise, activation=None, outActivation=None, learningRate=1.0, run=0):
  '''
  Generate corpus and observed supervision from a particular network architecture,
  sampling one-hot supervision for each document prior.
  '''
  
  print('Generating corpus from dDMR model...')
  print('Num Documents: %d\nNum Topics: %d\nNum Words per Document: %d\nVocabulary Size: %d\nSupervision Dimensionality: %d\nNeural Prior Architecture: %s\nActivation Function: %s\nStdDev of Supervision Noise: %.3f' % (D, Z, Nd, V, architecture[0], architecture[1:], actToStr(activation), noise))
  
  outWidth = architecture[-1]
  
  latentVals = np.random.randint(0, outWidth, D) # which prior to use for each document 
  trueSupervision = scipy.sparse.csr_matrix(([1.0 for v in latentVals],
                                             ([i for i, v in enumerate(latentVals)],
                                              latentVals))).todense().astype('float32')
  
  netWeights = []
  inWidth  = architecture[0]
  observedSupervision = theano.shared(0.01*np.random.randn(D, inWidth).astype('float32'),
                                      'input')
  net = observedSupervision
  allLayers = []
  
  for idx, (i, j) in enumerate(zip(architecture, architecture[1:])):
    
    netWeights.append(np.random.randn(j, i).astype('float32'))
    
    if idx >= len(architecture) - 2:
      if outActivation:
        net = outActivation(net.dot(netWeights[-1].T))
      else:
        net = net.dot(netWeights[-1].T)
    else:
      if activation:
        net = activation(net.dot(netWeights[-1].T))
      else:
        net = net.dot(netWeights[-1].T)
    allLayers.append(net)
  y = T.matrix('y')
  
  if outActivation == T.nnet.softmax:
    xent = T.nnet.categorical_crossentropy(T.clip(net, 1.e-12, 1.0 - 1.e-12), y)
    crossent_cost = xent.sum()
    cost = crossent_cost + 1.e-4 * (observedSupervision**2.).sum()/D # don't want input values to explode wildly
  else:
    cost = (T.sum((y - net)**2.0) + 1.e-4 * (observedSupervision**2.).sum())/D # MSE
  
  inGrad = T.grad(cost, [observedSupervision])
  
  optimizer = opt.AdadeltaOptimizer(learningRate=-learningRate, rho=0.95)
  
  updates = optimizer.getUpdates([observedSupervision], [inGrad[0]])
  
  updateObserved = theano.function(inputs=[y],
                                   outputs=net,
                                   updates=updates,
                                   allow_input_downcast=True)
  getCost = theano.function(inputs=[y], outputs=cost,
                                   allow_input_downcast=True)
  getActivations = theano.function(inputs=[], outputs=net,
                                   allow_input_downcast=True)
  
  for i in range(200):
    activations = updateObserved(trueSupervision)
    if not i % 10:
      print('Gen Data Iter %d: Cost %f' % (i, getCost(trueSupervision)))
  
  obsSup = observedSupervision.get_value() + noise*np.random.randn(D, inWidth)
  deltaVecs = np.exp(5.0 * np.random.choice(2, (outWidth, Z), p=[0.8, 0.2]) - 2.5) # dirichlet prior for topic preference for each label
  phiDists  = [np.random.dirichlet([0.1 for v in range(V)]) for z in range(Z)] # words preferred by each topic
  
  annotations = {'descriptor':obsSup.astype(np.float32),
                 'architecture':architecture,
                 'true_labels':activations,
                 'net_wts':netWeights,
                 'activation':activation,
                 'outActivation':outActivation,
                 'phi':phiDists,
                 'delta':deltaVecs,
                 'Z':Z,
                 'V':V,
                 'D':D,
                 'Nd':Nd}
  annNames = ['descriptor']
  annDicts = {'descriptor':{i:'features_%d' % (i) for i in range(inWidth)}}
  tokenDict = {wIdx:'word_%d' % (wIdx) for wIdx in range(V)}
  
  payload = {}
  payload['annotations']      = annotations
  payload['annotation_dicts'] = annDicts
  payload['annotation_names'] = annNames
  payload['token_dict']       = tokenDict
  
  # sample distributions, topics, and words for each document
  Ds = []
  Ws = []
  
  docPriors = activations.dot(deltaVecs) + 1.e-8
  for d in range(D):
    topicDist = np.random.dirichlet(docPriors[d,:])
    
    Ds.extend([d for i in range(Nd)])
    
    # how many times each topic was sampled
    Zs = np.random.choice(Z, Nd, replace=True, p=topicDist)
    
    # sample each word
    for z in Zs:
      phi = phiDists[z]
      Ws.append(np.random.choice(V, p=phi))
  
  payload['Ds_body'] = np.asarray(Ds, dtype=np.int32)
  payload['Ws_body'] = np.asarray(Ws, dtype=np.int32)
  
  actName    = actToStr(activation)
  outActName = actToStr(outActivation)
  
  dataPath = dataPathFmt % ('-'.join([str(width) for width in architecture]), D, Z, Nd, V, noise, actName, outActName, run)
  
  np.savez_compressed(dataPath, **payload)

def test(inPath, outPath, architecture, activation, outActivation, onlyBias, annName='descriptor', initOracleWeights=False):
  d = np.load(inPath)
  
  annotations = d['annotations'].item()[annName].astype(np.float32)
  annotationDict = d['annotation_dicts'].item()[annName]
  tokenDict = d['token_dict'].item()
  
  print('Loaded data')
  
  Ds, Ws = d['Ds_body'], d['Ws_body']
  D = Ds.max()+1
  Z = d['annotations'].item()['Z']
  W = Ws.max()+1
  
  # Initalize network parameters
  L2 = 1.0
  
  betaMean, betaL1, betaL2 = 0.0, 0.0, L2
  deltaMean, deltaL1, deltaL2 = 0.0, 0.0, L2
  omegaMean, omegaL1, omegaL2 = 0.0, 0.0, L2
  deltaBMean, deltaBL1, deltaBL2 = 0.0, 0.0, L2
  omegaBMean, omegaBL1, omegaBL2 = 0.0, 0.0, L2
  
  params = buildSpriteParams(architecture[-1], W, Z, betaMean, betaL1,
                             betaL2, deltaMean, deltaL1,
                             deltaL2, omegaMean, omegaL1, omegaL2, deltaBMean, deltaBL1,
                             deltaBL2, omegaBMean, omegaBL1, omegaBL2, tieBetaAndDelta=True)
  
  optimizer = opt.AdadeltaOptimizer(-0.25)
  
  alphaGraph = mlp.MLP(12345, architecture, activation, outActivation,
                       optimizer, 0.0, L2, 'alphaGraph')
  alphaWts = alphaGraph.getWeights()
  origWts = deepcopy(alphaWts)
  if initOracleWeights and (len(architecture) > 1):
    wts = d['annotations'].item()['net_wts']
    idealWts = alphaWts
    for i, w in enumerate(wts):
      idealWts[i*2] = w.T
    
    alphaGraph.setWeights(idealWts)
  
  prior = NeuralPrior(params, alphaGraph, optimizer=optimizer, onlyBias=onlyBias)
  
  model = NeuralLDA(Z, W, prior, heldoutIndex=2, seed=12345)
  
  model.setData(Ws, Ds, annotations)
  
  print('Built model')
  
  model.learn(iters=1000, burnin=100, numSamples=10, llFreq=50)
  
  # Print out topics
  model.printTopWords(tokenDict)
  
  payload = model.serialize(saveCorpus=True)
  
  payload['topics'] = model.getTopWords(tokenDict)
  payload['annotations'] = model.getTopAnnotations(annotationDict)
  
  omegaBTopic, supertopics = model.prior.getSupertopics(tokenDict, topn=20)
  payload['omegaBias_supertopic'] = omegaBTopic
  payload['omega_supertopics']    = supertopics
  payload['likelihood_history']   = model.ll_history
  
  np.savez_compressed(outPath, **payload)

def genSynthAndTest(architecture, D, Z, Nd, V, noise, activation=None, outActivation=None, annName='descriptor', run=0):
  learningRate = 1.0
  
  dataPath  = dataPathFmt  % ('-'.join([str(width) for width in architecture]), D, Z, Nd, V, noise, str(activation), str(outActivation), run)
  if not os.path.exists(dataPath):
    genDataFromNetwork(architecture, D, Z, Nd, V, noise, activation, outActivation, learningRate, run)
  else:
    print('Skipping building data...')
  
  print('Fitting LDA')
  
  actName = actToStr(activation)
  outActName = actToStr(outActivation)
  
  modelType = 'lda'
  modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
                              Z, Nd, V, noise, actName, outActName, modelType, run)
  test(dataPath, modelPath, [architecture[0]], None, None, True, annName, False)
  
  print('Fitting DMR')
  
  actName = actToStr(activation)
  outActName = actToStr(outActivation)
  
  modelType = 'dmr'
  modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
                              Z, Nd, V, noise, actName, outActName, modelType, run)
  test(dataPath, modelPath, [architecture[0]], None, None, False, annName, False)
  
  print('Fitting dDMR')
  
  actName = actToStr(activation)
  outActName = actToStr(None)
  
  modelType = 'neural'
  modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
                              Z, Nd, V, noise, actName, outActName, modelType, run)
  test(dataPath, modelPath, architecture, activation, None, False, annName, False)

def plotSynthResults(runDir='../test_output/runs/', plotDir='../test_output/plots/'):
  ''' Collect and print performance for different models. '''
  
  if not os.path.exists(plotDir):
    os.mkdir(plotDir)
  
  def getCorpusParams(path):
    flds = path.replace('.npz', '').split('_')[1:]
    return tuple(['-'.join(f.split('-')[1:]) for f in flds])
  
  modelPaths = [p for p in os.listdir(runDir) if p.endswith('.npz') and p.find('_run-') > -1]
  trainPPLs   = {}
  heldoutPPLs = {}
  iterations  = {}
  
  allParams = []
  
  for p in modelPaths:
    try:
      model = np.load(os.path.join(runDir, p))
    except Exception as ex:
      import pdb; pdb.set_trace()
    
    params = getCorpusParams(p)
    actFn = params[6] if len(params) > 7 else 'sigmoid'
    key   = (int(params[0].split('-')[0]), float(params[5]), actFn, int(params[-1])) # dimensionality, noise, activation fn
    modelName = params[-2]
    
    if modelName == 'lda':
      modelName = 'LDA'
    elif modelName == 'dmr':
      modelName = 'DMR'
    elif modelName == 'neural':
      modelName = 'dDMR'
    
    history = [o for i, o in enumerate(model['likelihood_history']) if (i % 5) == 0]
    
    if key not in trainPPLs:
      trainPPLs[key] = {};
      heldoutPPLs[key] = {};
      iterations[key] = {};
    
    trainPPLs[key][modelName]   = [o['train_ppl'] for o in history]
    heldoutPPLs[key][modelName] = [o['heldout_ppl'] for o in history]
    iterations[key][modelName]  = [o['iteration'].item() for o in history]
    
    allParams.append(params)
  
  MODELS = ['LDA', 'DMR', 'dDMR']
  
  # For each corpus, plot train/heldout perplexity curves
  for key in trainPPLs:
    if not (('LDA' in trainPPLs[key]) and ('DMR' in trainPPLs[key]) and ('dDMR' in trainPPLs[key]) and ('LDA' in heldoutPPLs[key]) and ('DMR' in heldoutPPLs[key]) and ('dDMR' in heldoutPPLs[key])):
      print('Skipping', key)
      continue
    
    try:
      pp = PdfPages(os.path.join(plotDir,
                                 'supdim-%d_noise-%.3f_activation-%s_run-%d.pdf' % key))
      fig, ax = plt.subplots()
      
      ax.plot(iterations[key]['LDA'], trainPPLs[key]['LDA'],   'r^:', label='LDA-train')
      ax.plot(iterations[key]['LDA'], heldoutPPLs[key]['LDA'], 'r^--', label='LDA-dev')
      ax.plot(iterations[key]['DMR'], trainPPLs[key]['DMR'],   'gv:', label='DMR-train')
      ax.plot(iterations[key]['DMR'], heldoutPPLs[key]['DMR'], 'gv--',
              label='DMR-dev')
      ax.plot(iterations[key]['dDMR'], trainPPLs[key]['dDMR'], 'b<:', label='dDMR-train')
      ax.plot(iterations[key]['dDMR'], heldoutPPLs[key]['dDMR'], 'b<--',
              label='dDMR-dev')
      ax.set_xlabel('Iteration', fontsize=24)
      ax.set_ylabel('Heldout Perplexity', fontsize=24)
      legend = ax.legend(loc='best', fontsize=18)
      
      plt.gcf().subplots_adjust(bottom=0.15)
      plt.axvline(x=100, color='black', linestyle='dashed')
      
      pp.savefig()
      pp.close()
    except Exception as ex:
      print('Exception:', ex)
      import pdb; pdb.set_trace()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='train and evaluate models on synthetic data')
  #parser.add_argument('--arch', required=True, metavar='LAYER_WIDTH',
  #                    nargs='+', type=int,
  #                    help='hidden layer widths in the alpha network architecture')
  parser.add_argument('--arch', metavar='LAYER_WIDTH',
                      nargs='+', type=int,
                      help='hidden layer widths in the alpha network architecture')
  #parser.add_argument('--noise', required=True, metavar='NOISE', type=float,
  #  help='standard deviation of gaussian noise to perturb supervision')
  parser.add_argument('--noise', default=0.1, metavar='NOISE', type=float,
    help='standard deviation of gaussian noise to perturb supervision')
  parser.add_argument('--D', type=int, default=10000, help='number of documents to generate')
  parser.add_argument('--Z', type=int, default=5,
                      help='number of topics')
  parser.add_argument('--V', type=int, default=100,
                      help='vocabulary size')
  parser.add_argument('--Nd', type=int, default=10,
                      help='number of tokens per document')
  parser.add_argument('--nonlinear', action='store_true',
                      help='use sigmoid activations in hidden layer')
  parser.add_argument('--run', type=int, default=0,
                      help='which run')
  args = parser.parse_args()
  
  args.arch = [1000, 10] if not args.arch else args.arch
  
  if not args.nonlinear:
    act, outAct = None, None
    print('Only linear activation functions')
  else:
    act, outAct = T.nnet.sigmoid, None
    print('Using nonlinear activation functions')

  data_dir  = os.path.join(synth_output_dir, 'data')
  run_dir   = os.path.join(synth_output_dir, 'runs')
  plot_dir  = os.path.join(synth_output_dir, 'plots')
  
  if not os.path.exists(synth_output_dir):
    os.mkdir(synth_output_dir)
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  if not os.path.exists(run_dir):
    os.mkdir(run_dir)
  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
  
  genSynthAndTest(architecture=args.arch, D=args.D, Z=args.Z, Nd=args.Nd,
                  V=args.V, noise=args.noise, activation=act, outActivation=outAct,
                  run=args.run)
  plotSynthResults(run_dir, plot_dir)
