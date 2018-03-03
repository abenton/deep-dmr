import logging
import sys
import time

import numpy as np

import theano.tensor as T

EPS=1.e-6

logger = logging.getLogger('sprite')

def digamma(x):
  r = 0.0
  
  while (x <= 5.0):
    r -= 1.0 / x
    x += 1.0
    
  f = 1.0 / (x * x)
  t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617.0 / 8160.0)))))))
  return r + np.log(x) - 0.5 / x + t

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print ('%r (%r, %r) %2.5f sec' % (method.__name__, args, kw, te-ts))
        return result

    return timed

def getLeakyRelu(leakiness=0.0):
  return lambda x: T.nnet.relu(x, leakiness)
