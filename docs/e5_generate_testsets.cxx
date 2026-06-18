/**
 * E5 — Generate test sets at σ ∈ {0.10, 0.15, 0.20, 0.25}
 *
 * Each test set: 100 samples, fresh random geometries disjoint from training.
 * Tensors saved per file:
 *   [0] G       [100, 50]       geometry control points
 *   [1] u       [100, 100]      B-spline coefficients
 *   [2] u_val   [100, 100]      field values at Greville points
 *   [3] G_grid  [100, 64, 64, 2]
 *   [4] u_grid  [100, 64, 64]
 *
 * Output: e5_test_s010.pt  e5_test_s015.pt  e5_test_s020.pt  e5_test_s025.pt
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <cmath>

using namespace iganet;

void generate(double sigma, int64_t N, const std::string& out_file) {
    using Geo = S<UniformBSpline<double, 2, 2, 2>>;
    using Var = S<UniformBSpline<double, 1, 3, 3>>;

    constexpr int64_t Ngrid = 64;

    auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
    auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);
    utils::TensorArray<2> xi_grid;
    xi_grid[0] = xi_1d.repeat_interleave(Ngrid);
    xi_grid[1] = xi_1d.repeat(Ngrid);

    std::vector<torch::Tensor> G_ds, u_ds, uv_ds, Gg_ds, ug_ds;

    int64_t attempts = 0;
    while ((int64_t)G_ds.size() < N) {
        ++attempts;
        Geo G({5, 5}, init::linear);
        auto& sp = G.space<0>();
        for (short_t d = 0; d < sp.geoDim(); ++d) {
            auto c = sp.coeffs(d).view({5, 5});
            c.slice(0, 1, 4).slice(1, 1, 4) +=
                sigma * torch::randn({3, 3}, sp.coeffs(d).options());
        }

        auto rhs = [&G](const std::array<torch::Tensor, 2>& xi)
            -> std::array<torch::Tensor, 1> {
            auto phys = G.space<0>().eval(xi);
            return {torch::sin(M_PI * *phys[0]) * torch::sin(M_PI * *phys[1])};
        };

        Var u_template({10, 10});
        EZSolver solver(G, u_template, rhs);
        solver.init();
        solver.assemble();
        auto u_coeffs = solver.solve().clone();
        auto A = solver.lhs().to_dense();
        auto b = solver.rhs().clone();

        double rel_res = (torch::mm(A, u_coeffs.unsqueeze(1)).squeeze(1) - b)
                            .norm().item<double>() /
                        (b.norm().item<double>() + 1e-14);
        if (rel_res >= 0.01) continue;

        G_ds.push_back(G.as_tensor().clone());
        u_ds.push_back(u_coeffs);

        Var u_solved({10, 10});
        u_solved.space<0>().from_tensor(u_coeffs);
        auto gpts = u_solved.space<0>().greville(false);
        auto uv   = u_solved.space<0>().eval(gpts);
        uv_ds.push_back(uv[0]->clone());

        auto Gvals  = G.space<0>().eval(xi_grid);
        auto G_grid = torch::stack({Gvals[0]->reshape({Ngrid, Ngrid}),
                                    Gvals[1]->reshape({Ngrid, Ngrid})}, 2);
        auto uvals  = u_solved.space<0>().eval(xi_grid);
        auto u_grid = uvals[0]->reshape({Ngrid, Ngrid});
        Gg_ds.push_back(G_grid.clone());
        ug_ds.push_back(u_grid.clone());
    }

    double rej = 100.0 * (1.0 - double(N) / double(attempts));
    Log() << "  σ=" << sigma << "  accepted=" << N
          << "  attempts=" << attempts
          << "  rejection=" << std::fixed << std::setprecision(1) << rej << "%\n";

    torch::save({torch::stack(G_ds),
                 torch::stack(u_ds),
                 torch::stack(uv_ds),
                 torch::stack(Gg_ds),
                 torch::stack(ug_ds)},
                out_file);
    Log() << "  → " << out_file << "\n";
}

int main() {
    init();

    struct Cfg { double sigma; const char* tag; };
    for (auto [sigma, tag] : std::vector<Cfg>{
            {0.10, "s010"}, {0.15, "s015"},
            {0.20, "s020"}, {0.25, "s025"}}) {
        std::string fname = std::string("e5_test_") + tag + ".pt";
        Log() << "\nGenerating " << fname << " ...\n";
        generate(sigma, 100, fname);
    }

    finalize();
    return 0;
}
