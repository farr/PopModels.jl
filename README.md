# PopModels.jl

Julia package for population modeling: you supply the parameterized population
model, we supply the rest.  

In the "lingo:" we fit an inhomogeneous, censored Poisson process to your
catalog of observations that come with measurement uncertainty.  Or: we fit a
population model to a catalog of uncertain parameter estimates for objects
subect to selection effects.  

You supply:

* The population model: the Poisson rate density in parameter space (you code
  this up in a [Turing](https://turing.ml) model, including with its parameters).
* The catalog of observations: parameter samples drawn from some posterior
  density for each object in the catalog.
* A set of mock detections of some known population of objects produced by your
  search/detection pipeline (this is used to estimate the selection function;
  see, e.g., [Farr
  (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract),
  [Mandel, Farr, & Gair
  (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.1086M/abstract)).

PopModels.jl will:

* Compute the poisson likelihood for you, with the correct normalization to
  account for selection effects.
* Draw population-weighted samples for the catalog objects from the set of
  posterior samples you provided.
* Draw population-weighted *detected* parameters from the set of injections you
  supplied.
* Estimate the normalized event rate.
* Provide you with various sampling diagnostics that will tell you when you need
  more samples from each event or more injections in order to accurately
  complete the simulation.

## Example

(This is just a more extensively commented version of one of the tests in
`test/runtests.jl` if you want to see it in action!)

We begin by generating a mock catalog of observations and a set of injections to
measure our selection function.  Normally this part of the process would be done
already by the time you are doing your population analysis (see e.g. the [GWTC-3
LIGO-Virgo-Kagra catalog parameter estimates](https://zenodo.org/record/5546663)
and [search sensitivity estimates](https://zenodo.org/record/5546676)), but we
do it here for the sake of the example.

Then we infer the population model parameters using this catalog and make some
plots that show we accurately recover the true parameters used to generate the
catalog.

### Mock Catalog

Suppose you have a population of objects with a single parameter `x` distributed
according to a unit normal: `x ~ Normal(0, 1)`.  You have an experimental setup
that "observes" these objects and measures `x` with some uncertainty; let us
suppose you observe `x_obs ~ Normal(x, 1)`.  There is a selection effect: if
`x_obs < 0`, your setup does not detect the object!  You want to estimate:

* The number of objects in the *true* population (i.e. the "rate").
* The mean of the population distribution.
* The s.d. of the population distribution.

Using your noisy observations of the selected subset of the full population.
Here is some Julia code that will generate such a population, observe it, and
produce posterior samples of the true `x` for each observed object:

```julia
using Distributions
using Random

Random.seed!(0x18b9cb8ac7cf4f1a) # For reproducability!
mu_true = 0.0
sigma_true = 1.0
sigma_obs = 1.0

Ntrue = 200
x_th = 0.0

x_true = rand(Normal(mu_true, sigma_true), Ntrue)
x_obs = x_true .+ rand(Normal(0, sigma_obs), Ntrue)
x_det = x_obs[x_obs .> x_th]
```

`x_det` is the observed values of `x` for the detected subset of the population.

If we put a flat prior on `x`, the posterior for each observation (which, with
this prior, will just be proportional to the likelihood function) is `x_post ~
Normal(x_obs, 1)`.  So we can draw posterior samples for each observation:

```julia
Npost = 100
xpost = map(x_det) do xd
    rand(Normal(xd, sigma_obs), Npost)
end
```

Because we used a flat prior, the prior weight for each sample is just a
constant (PopModels.jl does not need the normalized prior weight, which is good,
since our prior is improper!).

```julia
log_wts = map(xpost) do xp
    zeros(length(xp))
end
```

To measure our selection effects, we draw a synthetic population of true `x`
values uniformly between `-5` and `5`:

```julia
Ndraw = 20000
x_draw = rand(Uniform(-5, 5), Ndraw)
x_draw_obs = x_draw .+ rand(Normal(0, sigma_obs), Ndraw)
x_draw_det = x_draw[x_draw_obs .> x_th]
log_pdraw_det = log.(1/10*ones(length(x_draw_det)))
```

`log_pdraw_det` is the log of the (properly normalized!) population density from
which the synthetic population was drawn.  

At this point our synthetic catalog is complete (usually these data products
would already be provided to you before you begin your population analysis).

### Population Analysis

We want to determine the number of objects in the full population, the
population mean, and the population standard deviation.  We build a Turing model
for our analysis:

```julia
using PopModels
using Turing

# nn == Normal-Normal model: Normal population, Normal measurement uncertainty.
@model function nn_model_fn(xpost, log_wts, xdraw, log_pdraw, Ndraw)
    # These are priors on our population parameters: the mean and s.d. of the population
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    # Here is our population model: a (normalized) rate density in
    # parameter space, which in this case is just a Gaussian with mean
    # parameter mu and s.d. parameter sigma.  Since this is normalized
    # to integrate to 1 over all `x`, the rate parameter that is fitted
    # by PopModels.jl will be the total number of objects in the
    # population; another possible normalization could be 
    #
    # log_dN = x -> -1/2*((x-mu)/sigma)^2
    #
    # in which case the rate parameter `R` would be the rate density at
    # `x = mu`.  In all cases, the (properly normalized) poisson
    # intensity as a function of parameter is given by `x -> R *
    # exp(log_dN(x))`.
    log_dN = x -> logpdf(Normal(mu, sigma), x)

    # We pass the (log of the) population density, our array of
    # posterior samples for each event, the (log of the) prior weights,
    # our mock detections, the (log of the) mock population density, and
    # the total number of draws made to `pop_model_body`.  It returns
    # the log-likelihood, the log-normalization to account for
    # selection, and some generated quantities (more on this later).
    logl, lognorm, genq = pop_model_body(log_dN, xpost, log_wts, xdraw, log_pdraw, Ndraw)
    
    # We accumulate the log likelihood and log normalization into the Turing model.
    Turing.@addlogprob! logl
    Turing.@addlogprob! lognorm

    # And return the generated quantities from the model:
    genq
end
```

You can now get posterior samples over `mu` and `sigma` using Turing, e.g.

```julia
model = nn_model_fn(xpost, log_wts, x_draw_det, log_pdraw_det, Ndraw)

Nmcmc = 1000
Ntune = Nmcmc
trace = sample(model, NUTS(Ntune, 0.65), Nmcmc)
genq = generated_quantities(model, trace)
```

`trace` will contain `:mu` and `:sigma` entries that peak around `0` and `1`,
and measure the uncertainty given this catalog of around 100 observations.  

`genq` has various generated quantities:

* `R` is the rate normalization; the Poisson intensity in parameter space is
  given by `x -> R*exp(log_dN(x))` where `log_dN` is the population model you
  provided.
* `Neff_sel` gives the effective number of injections that contribute to the
  Monte-Carlo selection function integral; this should be *at least*
  `4*len(xpost)` (four times the size of the catalog) or else the selection
  function is not measured well enough to get a reliable fit.  (See [Farr
  (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract) for
  details.)
* `Neff_samps` gives the effective number of posterior samples for each
  observation that contribute to the Monte-Carlo likelihood integral.  This
  should be >> 1 (3 is marginal, 10 is great, 100 is overkill) or else the
  posterior samples do not cover the region of high population density well
  enough to estimate the likelihood accurately for this sample.  (This can be
  fixed by sampling from a prior that is better adapted to the population model,
  or just by brute-forcing more samples for the troublesome events in the
  catalog; note that PopModels.jl does not require the same number of posterior
  samples for each event, so you can adjust this on an event-by-event basis).
* `thetas_popwt` is an array with one parameter sample for each event drawn from
  a *population-weighted* prior.  The set of these samples can provide
  population-weighted parameter estimates for each event.
* `theta_draw` is a draw of parameters from the set of detected synthetic
  population samples weighted by the current population model; these samples can
  be used to form a "predicted detected distribution" for the population model,
  which can be compared to the population-weighted catalog samples for model
  checking.

#### Checking the Simulation
  
Here is a traceplot:

```julia
using LaTeXStrings
using StatsPlots
plot(trace)
savefig("static/traceplot.png")
```

![traceplot](static/traceplot.png)

We can check that the effective number of injections contributing to the
selection function estimate is large enough for accurate estimation:

```julia
using Printf

neff_min = minimum([x.Neff_sel for x in genq])
@printf("neff_min = %.1f\n", neff_min) # => 1769.7
println("neff_min > 4*len(xpost): $(neff_min > 4*length(xpost))") # => true
```

We can also check that the effective number of posterior samples contributing to
the likelihood integral is reasonable:

```julia
npost_min = minimum([minimum(x.Neff_samps) for x in genq])
@printf("npost_min = %.1f\n", npost_min) # => 1.2
```

That's a bit smaller than we would like in a production analysis, but we'll roll
with it for now!

#### Post-Processing the Population Posterior, Plots

We can see that the posteriors for both `mu` and `sigma` support the true values
of these parameters: 

```julia
p1 = @df trace density(:mu, label=nothing, xlabel=L"\mu")
p2 = @df trace density(:sigma, label=nothing, xlabel=L"\sigma")

p1 = vline!(p1, [mu_true], color=:black, label=nothing)
p2 = vline!(p2, [sigma_true], color=:black, label=nothing)

plot(p1, p2, layout=(1, 2))
savefig("static/mu-sigma-posterior.png")
```

![mu-sigma-posterior](static/mu-sigma-posterior.png)

We can also see that the recovered `R` parameter (due to the choice of
normalization in `log_dN`, this is the total number of objects in the
population) is well-recovered:

```julia
p = density([x.R for x in genq], label=nothing, xlabel=L"R")
p = vline!(p, [Ntrue], color=:black, label=nothing)
plot(p)
savefig("static/R-posterior.png")
```

![R-posterior](static/R-posterior.png)

TODO: show how to use `thetas_popwt` and `theta_draw` to check the model.

### Conclusion

This example, while simple, contains all the elements of a "real" population
analysis; hopefully you can now see how to adapt your population analysis to use
PopModels.jl!