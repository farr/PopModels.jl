using Distributions
using PopModels
using Random
using Test
using Turing

function p_value(x, xs)
    sum(xs .<= x)/length(xs)
end

@testset "PopModels.jl Tests" begin
    @testset "Normal-Normal model" begin
        Random.seed!(1404601137169444883)
        mu_true = 0.0
        sigma_true = 1.0
        sigma_obs = 1.0

        Ntrue = 200
        Ndraw = 20000
        Npost = 100

        x_th = 0.0

        x_true = rand(Normal(mu_true, sigma_true), Ntrue)
        x_obs = x_true .+ rand(Normal(0, sigma_obs), Ntrue)
        x_det = x_obs[x_obs .> x_th]
        xpost = map(x_det) do xd
            rand(Normal(xd, sigma_obs), Npost)
        end
        log_wts = map(xpost) do xp
            zeros(length(xp))
        end

        x_draw = rand(Uniform(-5, 5), Ndraw)
        x_draw_obs = x_draw .+ rand(Normal(0, sigma_obs), Ndraw)
        x_draw_det = x_draw[x_draw_obs .> x_th]
        log_pdraw_det = log.(1/10*ones(length(x_draw_det)))

        @model function nn_model_fn(xpost, log_wts, xdraw, log_pdraw, Ndraw)
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)

            log_dN = x -> logpdf(Normal(mu, sigma), x)
            logl, lognorm, genq = model_body(log_dN, xpost, log_wts, xdraw, log_pdraw, Ndraw)
            Turing.@addlogprob! logl
            Turing.@addlogprob! lognorm

            genq
        end

        model = nn_model_fn(xpost, log_wts, x_draw_det, log_pdraw_det, Ndraw)
        trace = sample(model, NUTS(1000, 0.65), 1000)
        genq = generated_quantities(model, trace)

        @test minimum([x.Neff_sel for x in genq]) > 4*length(x_det)

        Rq = p_value(Ntrue, [x.R for x in genq])
        @test (Rq > 0.05) && (Rq < 0.95)

        muq = p_value(mu_true, vec(trace[:mu]))
        @test (muq > 0.05) && (muq < 0.95)

        sigmaq = p_value(sigma_true, vec(trace[:sigma]))
        @test (sigmaq > 0.05) && (sigmaq < 0.95)
    end
end