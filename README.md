imm
===

imm is a Python package for Bayesian MCMC inference in infinite mixture
models. It is partially implemented in Cython for speed.

imm's main purpose is the clustering of multidimensional data in cases in
which the number of clusters is not exactly known.

Currently, imm supports the following nonparametric models:

* the Dirichlet process (DP) mixture model and
* the mixture of finite mixtures (MFM) model.

The base measure can be either

* a multivariate normal distribution with a conjugate normal-Wishart prior or
* a multivariate normal with a conditionally conjugate normal-Wishart prior.

I will release proper documentation eventually. For now, have a look at the
tutorial section below.

Getting started
---------------

<!-- To install the latest version from PyPI, call `sudo pip install imm`. -->

To install imm manually from the GitHub repository, clone it and do
`python setup.py install --user`. Required are recent Cython and SciPy
installations.

First steps
-----------

Below I demonstrate how to address a simple inference problem in imm. Use
Google Chrome to view the WebM content.

.. code-block:: python

    import imm

    mm = imm.models.ConjugateGaussianMixture(
        xi=[0.,0.],
        rho=.1,
        beta=2.5,
        W=[[1.5,.5],[.5,1.5]])

    pm = imm.models.DP(mm, alpha=.75, seed=1)

    x_n, c_n = pm.draw(size=1000)

This will generate a data of the following form:

.. figure:: https://raw.githubusercontent.com/tscholak/imm/master/dpgmm.png
   :alt: DP example
   :align: center

Now we will try to infer the labels from the data:

.. code-block:: python

    s = imm.samplers.CollapsedSAMSSampler(pm, max_iter=100, warmup=0)

    c_n_sams, _ = pm.infer(x_n, sampler=s)

This is the result:

.. figure:: https://raw.githubusercontent.com/tscholak/imm/master/dpgmm_sams.png
   :alt: DP example
   :align: center

Or, as video:

<video controls>
    <source src="data:video/x-webm;base64,https://raw.githubusercontent.com/tscholak/imm/master/dpgmm_sams.webm" type="video/webm">
    Your browser does not support the video tag.
</video>

Acknowledgments, credits, and contact info
------------------------------------------

Credit goes to Hanna Wallach (UMass Amherst), whose
[DPMM project](https://github.com/hannawallach/dpmm) laid the foundation for
the present code. The unified approach, the application to (conditionally)
conjugate Gaussian mixture models, the mixture of finite mixtures model,
and the SAMS sampler is my doing.

Although the code underwent and continues to
undergo significant testing, please understand that I cannot give a guarantee
that it is correct or will produce correct results. If you find an error,
open an issue or drop me an email at <torsten.scholak@googlemail.com>.
