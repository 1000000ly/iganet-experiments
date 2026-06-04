#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <splines/bspline.hpp>
#include <cmath>

using namespace iganet;

int main() {
  init();

  using Geo = S<UniformBSpline<double, 2, 2, 2>>;
  using Var = S<UniformBSpline<double, 1, 3, 3>>;

  const int64_t N     = 500;
  const int64_t Ngrid = 64;  // FNO grid resolution

  std::vector<torch::Tensor> G_dataset, u_dataset, u_val_dataset,
                              A_dataset, b_dataset,
                              G_grid_dataset, u_grid_dataset;

  // Precompute 64×64 parametric grid (same for every sample)
  auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
  auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);          // [64]
  utils::TensorArray<2> xi_grid;
  xi_grid[0] = xi_1d.repeat_interleave(Ngrid);  // [4096] row-coord (slow)
  xi_grid[1] = xi_1d.repeat(Ngrid);             // [4096] col-coord (fast)

  int64_t attempts = 0;
  for (int64_t i = 0; i < N; ++i) {

    // Retry until BiCGSTAB converges (relative residual < 1%)
    while (true) {
      ++attempts;
      Geo G({5, 5}, init::linear);

      auto& sp = G.space<0>();
      for (short_t d = 0; d < sp.geoDim(); ++d) {
        auto c = sp.coeffs(d).view({5, 5});
        c.slice(0, 1, 4).slice(1, 1, 4) +=
            0.10 * torch::randn({3, 3}, sp.coeffs(d).options());
      }

      auto rhs = [&G](const std::array<torch::Tensor, 2>& xi)
          -> std::array<torch::Tensor, 1> {
        auto phys = G.space<0>().eval(xi);
        return {torch::sin(M_PI * *phys[0]) *
                torch::sin(M_PI * *phys[1])};
      };

      Var u_template({10, 10});
      EZSolver solver(G, u_template, rhs);
      solver.init();
      solver.assemble();
      auto u_coeffs = solver.solve().clone();
      auto A        = solver.lhs().to_dense();
      auto b        = solver.rhs().clone();

      // Skip degenerate geometries (relative residual >= 1%)
      double rel_res = (torch::mm(A, u_coeffs.unsqueeze(1)).squeeze(1) - b)
                           .norm().item<double>() /
                       (b.norm().item<double>() + 1e-14);
      if (rel_res >= 0.01) continue;

      // ── IGA coefficient data (E1/E3) ──────────────────────────────────────
      G_dataset.push_back(G.as_tensor().clone());
      u_dataset.push_back(u_coeffs);
      A_dataset.push_back(A);
      b_dataset.push_back(b);

      // ── Field values at Greville points (E2) ──────────────────────────────
      Var u_solved({10, 10});
      u_solved.space<0>().from_tensor(u_coeffs);
      auto greville_pts = u_solved.space<0>().greville(false);
      auto u_val        = u_solved.space<0>().eval(greville_pts);
      u_val_dataset.push_back(u_val[0]->clone());

      // ── 64×64 grid data (E4 / FNO) ────────────────────────────────────────
    auto Gvals   = G.space<0>().eval(xi_grid);           // BlockTensor<T,1,2>
    auto G_grid  = torch::stack({Gvals[0]->reshape({Ngrid, Ngrid}),
                                  Gvals[1]->reshape({Ngrid, Ngrid})}, 2); // [64,64,2]

    auto uvals   = u_solved.space<0>().eval(xi_grid);    // BlockTensor<T,1,1>
    auto u_grid  = uvals[0]->reshape({Ngrid, Ngrid});    // [64, 64]

      G_grid_dataset.push_back(G_grid.clone());
      u_grid_dataset.push_back(u_grid.clone());
      break;  // sample accepted
    } // while retry
  } // for i

  auto G_tensor     = torch::stack(G_dataset);      // [N, 50]
  auto u_tensor     = torch::stack(u_dataset);      // [N, 100]
  auto u_val_tensor = torch::stack(u_val_dataset);  // [N, 100]
  auto A_tensor     = torch::stack(A_dataset);      // [N, 100, 100]
  auto b_tensor     = torch::stack(b_dataset);      // [N, 100]
  auto G_grid_tensor = torch::stack(G_grid_dataset); // [N, 64, 64, 2]
  auto u_grid_tensor = torch::stack(u_grid_dataset); // [N, 64, 64]

  // tensors[0]=G [1]=u [2]=u_val [3]=A [4]=b [5]=G_grid [6]=u_grid
  torch::save({G_tensor, u_tensor, u_val_tensor,
               A_tensor, b_tensor,
               G_grid_tensor, u_grid_tensor},
              "dataset.pt");

  Log() << "Dataset saved: " << N << " samples  (" << attempts << " geometry attempts)\n"
        << "G shape:       " << G_tensor.sizes()      << "\n"
        << "u shape:       " << u_tensor.sizes()      << "\n"
        << "u_val shape:   " << u_val_tensor.sizes()  << "\n"
        << "A shape:       " << A_tensor.sizes()      << "\n"
        << "b shape:       " << b_tensor.sizes()      << "\n"
        << "G_grid shape:  " << G_grid_tensor.sizes() << "\n"
        << "u_grid shape:  " << u_grid_tensor.sizes() << "\n";

  finalize();
  return 0;
}
