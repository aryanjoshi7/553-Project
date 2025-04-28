using Plots
using LaTeXStrings
L(r, τ) = (r >= 0) ? τ*r : (1-τ)*-r

x = range(-10, 10, 100)

L50 = L.(x, .50)
L90 = L.(x, .90)
L10= L.(x, .10)
plot(x, L50, xlabel=L"r", ylabel=L"\rho_{τ}(r)", label=L"τ=0.50", title="Quantile Loss")
plot!(x, L90, label=L"τ=0.90")
plot!(x, L10, label=L"τ=0.10")
savefig("quantloss.png")