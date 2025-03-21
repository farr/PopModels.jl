module PopModels

using Distributions
using StatsFuns

export pop_model_body
export estimate_Nposts


"""
  pop_model_body(log_dN_dtheta, thetas, log_theta_wts, thetas_sel, log_pdraw, Ndraw)

Implements a population model.

# Arguments

- `log_dN_dtheta``: log density function over parameters `theta`.
- `thetas`: nested interable of `n_observations` length, each element of which
  is an interable of `n_draws` from a posterior over parameters for each
  observation.
- `log_theta_wts`: log of the (un-normalized) prior weight assigned to each
  posterior sample.
- `thetas_sel`: iterable of parameters of artificial "injections" that have been
  detected by the same pipeline used to produce the catalog of `thetas`.  
- `log_pdraw`: log of the normalized probability density from which the
  injections are drawn at each detected sample.
- `Ndraw`: the total number of injections drawn to produce the detections in
  `thetas_sel`.

See [Farr
(2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract) for a
description of how selection effects are estimated in this model using
`thetas_sel`, `log_pdraw`, and `Ndraw`.

# Returns

`(log_likelihood_sum, log_normalization_sum, generated_quantities)`

After calling `model_body(...)`, you should add the log-likelihood and
log-normalization terms to the density via

```julia
Turing.@addlogprob! log_likelihood_sum 
Turing.@addlogprob! log_normalization_sum
```

The generated quantities include 

- `R`: The rate density scaling parameter (overall rate density in parameter
space is `R*exp(log_dN(...))`).
- `Neff_sel`: The number of effective samples in the importance weighted
integral of the selection function (the normalization); see [Farr
(2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract) for
definition.  This should be at absolute minimum ``4 N_\\mathrm{obs}`` (i.e. 4
times the catalog size), preferably much larger, or else the estimate of the
selection effects is not sufficiently accurate for reliable inference.
- `Neff_samps`: An array giving the number of effective samples in the
importance-weighted likelihood integral for each observation; these should all
be ``\\gg 1`` (3-4 is good, 10 is better) or else the likelihood integral is not
converged.
- `thetas_popwt`: An array giving a population-weighted draw from the parameter
samples for each event.
- `theta_draw`: A single draw of parameters from the detected injections
weighted by the population (these draws can be used to predict the *observed*
population implied by the model, and compared to the population of
`thetas_popwt` in model checking).

You should return the generated quantities from the turing model, so the full
use of this function looks like 

```julia
@model function my_model(thetas, log_theta_wts, thetas_sel, log_pdraw, Ndraw, ...) 
    # Set up parameter priors, etc. my_parameter_1 ~ Normal(...) and so on

    # Compute any derived quantities you need.
    derived_quantity = my_parameter_1^2 + ...

    # Obtain the log-rate-density from the model:
    log_dN = model_log_dN(my_parameter_1, derived_quantity, ...)

    logl, log_norm, genq = model_body(log_dN, thetas, log_theta_wts, thetas_sel, log_pdraw, Ndraw)
    Turing.@addlogprob! logl
    Turing.@addlogprob! log_norm

    # Compute anything else you need, maybe some more generated quantities
    additional_genq = (derived_quatity = derived_quantity, ...)

    # Return generated quantities 
    return merge(genq, additional_genq) # or just return genq if you have no additional generated quantities
```
"""
function pop_model_body(log_dN, thetas, log_theta_wts, thetas_sel, log_pdraw, Ndraw)
  Nobs = length(thetas)

  log_wts = map(thetas, log_theta_wts) do ths, log_wwt
      map(ths, log_wwt) do th, log_wt
          log_dN(th) - log_wt
      end
  end

  Neff_samps = map(log_wts) do lws
      mu_l = logsumexp(lws)
      wt = exp.(lws .- mu_l)
      1 / (length(wt) * var(wt))
  end

  log_likes = map(log_wts) do lw
      logsumexp(lw) - log(length(lw))
  end

  log_like_sum = sum(log_likes)

  log_wts_sel = map(thetas_sel, log_pdraw) do th, logp
      log_dN(th) - logp
  end
  log_mu = logsumexp(log_wts_sel) - log(Ndraw)
  log_s2 = logsubexp(logsumexp(2 .* log_wts_sel) - 2.0*log(Ndraw), 2*log_mu - log(Ndraw))
  Neff_sel_est = exp(2*log_mu - log_s2)
  Neff_sel = 1/(1/Neff_sel_est + 1/Ndraw)

  log_norm_sum = -Nobs*log_mu

  mu = exp(log_mu)
  R = rand(Normal(Nobs/mu, sqrt(Nobs)/mu))
  
  thetas_popwt = map(thetas, log_wts) do ths, log_wwt
      i = rand(Categorical(exp.(log_wwt .- logsumexp(log_wwt))))
      ths[i]
  end

  i_sel = rand(Categorical(exp.(log_wts_sel .- logsumexp(log_wts_sel))))
  theta_draw = thetas_sel[i_sel]

  return log_like_sum, log_norm_sum, (R = R, Neff_sel = Neff_sel, Neff_samps = Neff_samps, thetas_popwt=thetas_popwt, theta_draw=theta_draw)
end

"""
    estimate_Nposts(genq, Nposts, min_Neff_samps_desired)

Return an array giving an estimate of the number of posterior samples needed for
each event in a catalog to ensure that the minimum `Neff_samps` achieved in a
re-run of an MCMC is approximately `min_Neff_samps_desired`.

If the *actual* number of effective samples for an object is between
`min_Neff_samps_desired/2` and `2*min_Neff_samps_desired`, then the number of
samples for that object is not changed.  Otherwise, the number of samples is
scaled by the ratio of the actual number of effective samples to
`min_Neff_samps_desired`.  

# Arguments

- `genq`: Generated quantities from a previous MCMC run.
- `Nposts`: Array giving the number of posterior samples used for each member of
  the catalog in that run.
- `min_Neff_samps_desired`: The minimum number of effective samples desired for
  each event in a future MCMC run.

# Return

`Nposts_new`: an array estimating the number of posterior samples needed to
produce approximately `min_Neff_samps_desired` effective samples for each event
in the catalog in a future run that is similar to this one.

# Example

Suppose that `genq` is the generated quantities from an un-converged MCMC run
that was un-converged because `Neff_samps` for one or more events is not `>> 1`
when `Nposts` samples were used for the corresponding events. We want to re-run,
drawing more posterior samples for the troublesome events (and fewer for events
that have more effective samples than needed).  Suppose we desire a minimum of
10 effective samples for each event.  Then 

```julia
Nposts_new = estimate_Nposts(genq, Nposts, 10)
```

gives the number of posterior samples needed for each event in a future run.
"""
function estimate_Nposts(genq, Nposts, min_Neff_samps_desired)
  map(enumerate(Nposts)) do (i, Np)
    min_ne = minimum([x.Neff_samps[i] for x in genq])
    if min_ne < min_Neff_samps_desired/2 || min_ne > 2*min_Neff_samps_desired
      Int(round(Np * min_Neff_samps_desired / min_ne))
    else
      Np
    end
  end
end

end # module PopModel
