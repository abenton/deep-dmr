# deep-dmr
Implementation of Deep Dirichlet Multinomial Regression in Python 3 with Cython Gibbs sampler.

The Gibbs sampler was built from that the cython implementation here: <https://github.com/lda-project/lda>.
Current Gibbs sampler supports asymmetric priors.

## Build ##

  `sh build.sh`

## Run ##

After building, run the sample code to fit LDA, DMR, and dDMR on synthetic data (data are
generated from a deep DMR model):

```bash
cd deepdmr
python test_synth.py
```

Synthetic data can be found under *test/data/*, model runs can be found under *test/runs/*, and
plots under *test/plots/*.  The synthetic data is saved in compressed numpy format -- format
can be grokked by inspecting this file.  Relevant fields are:

- *Ds_body*: integer array for document indices
- *Ws_body*: integer array for token indices
- *token_dict*: dictionary mapping token type to index
- *annotation_names*: list of different annotation names associated with each document
- *annotation_dicts*: maps annotation name to an index-to-feature name dictionary
- *annotations*: dictionary of model parameters along with document supervision.
  In the synthetic data, the document-level supervision is in "descriptor".

## Contact Information ##

If you use this code and find it useful, please consider citing the accompanying paper:

> Adrian Benton and Mark Dredze. Deep Dirichlet Multinomial Regression. 2018. HLT-NAACL.

Feel free to email any questions or concerns to the first author:

first_name dot last_name at gmail dot com
