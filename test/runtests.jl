using Distributions
using GaussianKDEs
using PopModels
using Random
using Test
using Turing

function p_value(x, xs)
    sum(xs .<= x)/length(xs)
end

@testset "PopModels.jl Tests" begin
    @testset "Normal-Normal model" begin
        Random.seed!(0x8d183f093277d67f)
        mu_true = 0.5
        sigma_true = 1.0
        sigma_obs = 1.0

        Ntrue = 1000
        x_th = 0.0

        x_true = rand(Normal(mu_true, sigma_true), Ntrue)
        x_obs = x_true .+ rand(Normal(0, sigma_obs), Ntrue)
        x_det = x_obs[x_obs .> x_th]

        # Started with Nposts = 16, and refined using `estimate_Nposts` until
        # there were at least 8 Neff for each event
        Nposts = [21, 24, 36, 34, 27, 20, 38, 27, 24, 41, 31, 24, 19, 20, 22, 20, 59, 21, 28, 18, 40, 25, 23, 24, 21, 28, 80, 19, 25, 28, 20, 20, 23, 39, 22, 47, 19, 29, 21, 22, 19, 18, 33, 27, 28, 23, 31, 18, 30, 136, 21, 75, 23, 18, 19, 29, 79, 56, 19, 21, 21, 21, 20, 23, 115, 24, 25, 25, 18, 22, 20, 22, 19, 25, 22, 25, 23, 21, 28, 20, 19, 26, 19, 30, 37, 19, 19, 20, 19, 49, 19, 21, 22, 24, 35, 86, 24, 22, 21, 28, 19, 23, 19, 23, 70, 32, 28, 19, 18, 34, 20, 32, 22, 50, 19, 32, 22, 20, 25, 22, 43, 22, 97, 23, 22, 52, 160, 55, 45, 340, 20, 32, 21, 18, 21, 44, 21, 27, 40, 27, 19, 18, 20, 20, 36, 19, 23, 26, 22, 22, 32, 21, 19, 92, 22, 19, 56, 22, 23, 21, 24, 30, 19, 38, 43, 51, 20, 23, 26, 19, 20, 23, 22, 21, 28, 20, 28, 28, 20, 122, 30, 21, 29, 25, 79, 25, 23, 19, 55, 22, 19, 43, 19, 42, 27, 17, 20, 32, 22, 19, 109, 19, 19, 58, 35, 128, 52, 25, 19, 19, 31, 27, 40, 94, 122, 22, 19, 20, 19, 36, 19, 19, 112, 29, 33, 22, 34, 20, 19, 38, 19, 41, 21, 20, 100, 20, 19, 20, 29, 22, 28, 21, 26, 27, 20, 32, 54, 24, 20, 37, 22, 29, 34, 21, 25, 19, 19, 20, 42, 35, 71, 21, 28, 51, 20, 38, 19, 21, 30, 49, 23, 24, 25, 19, 21, 25, 19, 99, 20, 32, 24, 20, 24, 19, 21, 24, 21, 33, 29, 25, 33, 21, 55, 129, 19, 21, 58, 21, 45, 20, 20, 22, 24, 27, 19, 27, 27, 19, 33, 20, 45, 19, 21, 19, 20, 21, 101, 23, 60, 24, 22, 134, 27, 23, 19, 22, 23, 21, 22, 24, 21, 20, 31, 24, 29, 20, 132, 32, 27, 21, 22, 21, 33, 21, 22, 20, 580, 24, 22, 33, 21, 21, 322, 19, 29, 20, 30, 31, 19, 34, 46, 35, 22, 71, 32, 111, 36, 20, 22, 23, 23, 32, 19, 19, 21, 22, 21, 37, 33, 40, 21, 19, 21, 51, 19, 28, 21, 33, 23, 20, 29, 30, 20, 21, 20, 26, 20, 18, 29, 26, 1523, 23, 36, 20, 22, 18, 24, 20, 49, 19, 19, 19, 29, 20, 26, 50, 113, 23, 23, 19, 20, 19, 25, 25, 21, 21, 26, 26, 19, 20, 114, 21, 21, 24, 20, 22, 21, 28, 19, 57, 23, 34, 26, 50, 26, 23, 48, 20, 25, 18, 27, 20, 28, 24, 20, 20, 36, 20, 19, 18, 21, 20, 27, 22, 24, 27, 22, 33, 47, 22, 20, 20, 20, 21, 20, 30, 20, 51, 23, 24, 19, 111, 19, 19, 35, 20, 20, 21, 20, 20, 107, 44, 19, 19, 20, 21, 20, 24, 19, 22, 19, 22, 24, 95, 101, 20, 27, 24, 19, 46, 21, 49, 193, 41, 21, 31, 31, 21, 20, 87, 18, 27, 43, 22, 27, 24, 25, 26, 82, 25, 27, 21, 36, 36, 229, 25, 35, 22, 20, 21, 55, 32, 36, 20, 19, 20, 21, 36, 23, 20, 22, 20, 19, 29, 34, 21, 32, 18, 21, 21, 20, 18, 23, 30, 28, 20, 28, 20, 21, 67, 19, 20, 24, 26, 20, 23, 19, 21, 37, 70, 19, 24, 19, 20, 26, 21, 28, 100, 135, 27, 21, 35, 44, 35, 47, 20, 128, 31, 20, 30, 22, 18, 25, 24, 30, 22, 25, 20, 19, 76, 27, 21, 19, 25, 21, 20, 711, 19, 33, 20, 25, 115, 22, 22, 23, 20, 26, 19, 25, 19, 27, 21, 26, 24, 20, 24, 25, 25, 33, 31, 31, 28, 19, 21, 19, 33]
        xpost = map(x_det, Nposts) do xd, Npost
            rand(Normal(xd, sigma_obs), Npost)
        end
        log_wts = map(xpost) do xp
            zeros(length(xp))
        end

        Ndraw = 26824 # Seems to work OK (found after some trial and error)
        x_draw = rand(Uniform(-5, 5), Ndraw)
        x_draw_obs = x_draw .+ rand(Normal(0, sigma_obs), Ndraw)
        x_draw_det = x_draw[x_draw_obs .> x_th]
        log_pdraw_det = log.(1/10*ones(length(x_draw_det)))

        @model function nn_model_fn(xpost, log_wts, xdraw, log_pdraw, Ndraw)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)

            log_dN = x -> logpdf(Normal(mu, sigma), x)

            logl, lognorm, genq = pop_model_body(log_dN, xpost, log_wts, xdraw, log_pdraw, Ndraw)
            Turing.@addlogprob! logl
            Turing.@addlogprob! lognorm

            genq
        end

        model = nn_model_fn(xpost, log_wts, x_draw_det, log_pdraw_det, Ndraw)

        Nmcmc = 1000
        Ntune = Nmcmc
        trace = sample(model, NUTS(Ntune, 0.65), Nmcmc)
        genq = generated_quantities(model, trace)

        @test minimum([x.Neff_sel for x in genq]) > 4*length(x_det)
        @test minimum([minimum([x.Neff_samps[i] for x in genq]) for i in eachindex(x_det)]) > 8

        Rq = p_value(Ntrue, [x.R for x in genq])
        @test (Rq > 0.05) && (Rq < 0.95)

        muq = p_value(mu_true, vec(trace[:mu]))
        @test (muq > 0.05) && (muq < 0.95)

        sigmaq = p_value(sigma_true, vec(trace[:sigma]))
        @test (sigmaq > 0.05) && (sigmaq < 0.95)

        # 2D p-value
        pts = vcat(vec(trace[:mu])', vec(trace[:sigma])')
        kde = KDE(pts)
        log_p_pts = [logpdf(kde, p) for p in eachcol(pts)]
        log_p_true = logpdf(kde, [mu_true, sigma_true])
        @test sum(log_p_true .> log_p_pts) > 0.1*length(log_p_pts)
    end
end