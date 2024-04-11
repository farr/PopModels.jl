using CairoMakie
using Distributions
using LaTeXStrings
using PairPlots
using PopModels
using Random
using Turing

mu_true = 0.0
sigma_true = 1.0

N = 1000
xs_true = rand(Normal(mu_true, sigma_true), N)

hist(xs_true; axis = (;xlabel=L"x_\mathrm{true}"))

sigma_obs_min = 1.0
sigma_obs_max = 2.0
sigma_obs = rand(Uniform(sigma_obs_min, sigma_obs_max), N)
xs_obs = [rand(Normal(xt, s)) for (xt, s) in zip(xs_true, sigma_obs)]


f = Figure()
a = Axis(f[1, 1]; xlabel=L"x")
hist!(a, xs_true; label=L"x_\mathrm{true}")
hist!(a, xs_obs; label=L"x_\mathrm{obs}")
axislegend(a)
f

sel_thresh = mu_true-sigma_true
sel = xs_obs .> sel_thresh
xs_obs_sel = xs_obs[sel]
sigma_obs_sel = sigma_obs[sel]

N_samp = 100
xs_samp_sel = [rand(Normal(xo, s), N_samp) for (xo, s) in zip(xs_obs_sel, sigma_obs_sel)]
xs_samp_sel_logwts = [ones(N_samp) for _ in xs_samp_sel]

Ndraw = 20*N
xs_draw = rand(Normal(mu_true, 2*sigma_true), Ndraw)
sigma_draw = rand(Uniform(sigma_obs_min, sigma_obs_max), Ndraw)
xs_draw_obs = [rand(Normal(x, s)) for (x, s) in zip(xs_draw, sigma_draw)]
sel_draw = xs_draw_obs .> sel_thresh
xs_draw_sel = xs_draw[sel_draw]
xs_draw_obs_sel = xs_draw_obs[sel_draw]
sigma_draw_sel = sigma_draw[sel_draw]
p_draw_sel = pdf(Normal(mu_true, 2*sigma_true), xs_draw_sel)

function make_log_dN_dtheta(mu, sigma)
    function log_dN_dtheta(theta)
        r = theta - mu
        -0.5*r*r/(sigma*sigma) - log(sigma)
    end
    log_dN_dtheta
end 

@model function musigma_model(xs_samp_sel, xs_draw_sel, p_draw_sel, Ndraw)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)

    log_dN = make_log_dN_dtheta(mu, sigma)

    logl, log_norm, genq = pop_model_body(log_dN, xs_samp_sel, xs_samp_sel_logwts, xs_draw_sel, log.(p_draw_sel), Ndraw)
    Turing.@addlogprob! logl
    Turing.@addlogprob! log_norm

    genq
end

model = musigma_model(xs_samp_sel, xs_draw_sel, p_draw_sel, Ndraw)
@code_warntype model.f(
    model,
    Turing.VarInfo(model),
    Turing.SamplingContext(
        Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    ),
    model.args...,
)

trace = sample(model, NUTS(1000, 0.65), 1000)
genq = generated_quantities(model, trace)

f = Figure()
a = Axis(f[1, 1]; xlabel=L"N_\mathrm{eff,sel}")
density!(a, vec([x.Neff_sel for x in genq]))
vlines!(a, [4*length(xs_obs_sel)], color=:black)
f

pairplot(
    trace,
    PairPlots.Truth((;mu=mu_true, sigma=sigma_true));
    labels=Dict(:mu=>L"\mu", :sigma=>L"\sigma")
)