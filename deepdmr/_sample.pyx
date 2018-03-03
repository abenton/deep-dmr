#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

from cython.parallel import prange
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
from libc.math cimport log

cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil

cdef double log2(double x) nogil:
  return log(x) * 1.4426950408889634

cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

cpdef int _agg_samples(int[:] DS, int[:] ZS, int N, int[:,:] samples) nogil:
  ''' Count up final topic samples '''
  for i in range(N):
    samples[DS[i],ZS[i]] += 1

  return 0;

cdef int searchsorted(float* arr, int length, float value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

# Sampling topics according to asymmetric prior, single-threaded
cpdef _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz, float[:, :] priorDZ, float[:, :] priorZW, float[:] phiNorm, float[:] rands, int randOffset):
  cdef int i, k, w, d, z, z_new
  cdef float r, dist_cum
  cdef int N = WS.shape[0]
  cdef int n_rand = rands.shape[0]
  cdef int n_topics = nz.shape[0]
  cdef float* dist_sum = <float*> malloc(n_topics * sizeof(float))
  
  if dist_sum is NULL:
    raise MemoryError("Could not allocate memory during sampling.")
  with nogil:
    for i in range(N):
      w = WS[i]
      d = DS[i]
      z = ZS[i]
      
      dec(nzw[z, w])
      dec(ndz[d, z])
      dec(nz[z])
      
      dist_cum = 0
      for k in range(n_topics):
        dist_cum += (nzw[k, w] + priorZW[k, w]) / (nz[k] + phiNorm[k]) * (ndz[d, k] + priorDZ[d, k])
        dist_sum[k] = dist_cum
      
      r = rands[(i+randOffset) % n_rand] * dist_cum
      z_new = searchsorted(dist_sum, n_topics, r)
      
      ZS[i] = z_new
      inc(nzw[z_new, w])
      inc(ndz[d, z_new])
      inc(nz[z_new])
  
  free(dist_sum)

cpdef float _loglikelihood_marginalizeTopics(int[:] Ds, int[:] Ws, int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, float[:, :] priorDZ, float[:, :] priorZW, float[:] thetaDenom, float[:] phiDenom, int numThreads):
    cdef int i, z, d, w
    cdef int D = ndz.shape[0]
    cdef int N = Ds.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]
    
    cdef float tokenLL = 0
    cdef float ll = 0
    
    for i in prange(N, num_threads=numThreads, nogil=True):
      d = Ds[i]
      w = Ws[i]
      tokenLL = 0.0
      
      for z in range(n_topics):
        tokenLL += (ndz[d,z] + priorDZ[d,z]) * (nzw[z,w] + priorZW[z,w]) * phiDenom[z]
      
      ll += log2(tokenLL * thetaDenom[d])
    
    return ll

