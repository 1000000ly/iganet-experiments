/**
 * Task 1 — Solver sanity check on identity geometry
 *
 * Solves  Δu = 1,  u = 1 on ∂[0,1]²
 * with the fixed EZSolver (physical-domain Laplacian + geometry copy).
 *
 * Identity geometry ⟹ J = I, g^{kl} = δ^{kl}, physical == parametric.
 * Exact solution via Fourier sine series:
 *   u_exact(x,y) = 1 - Σ_{m,n odd} 16/(mn π⁴ (m²+n²)) sin(mπx) sin(nπy)
 *
 * Reports relative L2 error on a 64×64 evaluation grid.
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <cmath>

using namespace iganet;

// Truncated Fourier series: Δu=1, u=1 on ∂[0,1]²
// u = 1 + v  where Δv = 1, v=0 on boundary
// v_{mn} = -16 / (mn π⁴ (m²+n²))  for m,n odd
static double u_exact(double x, double y, int M = 51) {
  double v = 0.0;
  for (int m = 1; m <= M; m += 2)
    for (int n = 1; n <= M; n += 2) {
      double c = 16.0 / (m * n * std::pow(M_PI, 4) * (m*m + n*n));
      v -= c * std::sin(m * M_PI * x) * std::sin(n * M_PI * y);
    }
  return 1.0 + v;
}

int main() {
  init();

  using Geo = S<UniformBSpline<double, 2, 2, 2>>;
  using Var = S<UniformBSpline<double, 1, 3, 3>>;

  // Identity geometry: linear init, no perturbation
  Geo G({5, 5}, init::linear);

  auto rhs = [](const std::array<torch::Tensor, 2>& xi)
      -> std::array<torch::Tensor, 1> {
    return {torch::ones_like(xi[0])};
  };

  // Solve
  Var u_template({10, 10});
  EZSolver solver(G, u_template, rhs);
  solver.init();
  solver.assemble();
  auto u_coeffs = solver.solve().clone();

  // Residual of the linear system
  auto A   = solver.lhs().to_dense();
  auto b   = solver.rhs().clone();
  double lin_res = (torch::mm(A, u_coeffs.unsqueeze(1)).squeeze(1) - b)
                       .norm().item<double>() /
                   b.norm().item<double>();
  Log() << "Linear system relative residual: " << lin_res << "\n";

  // Evaluate IGA solution and analytical solution on 64×64 grid
  const int64_t Ngrid = 64;
  auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
  auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);
  utils::TensorArray<2> xi_grid;
  xi_grid[0] = xi_1d.repeat_interleave(Ngrid); // [4096]
  xi_grid[1] = xi_1d.repeat(Ngrid);

  Var u_solved({10, 10});
  u_solved.space<0>().from_tensor(u_coeffs);
  auto u_vals = u_solved.space<0>().eval(xi_grid); // BlockTensor<T,1,1>
  auto u_iga  = u_vals[0]->reshape({Ngrid, Ngrid}); // [64, 64]

  // Build analytical solution tensor
  auto u_ref = torch::zeros({Ngrid, Ngrid}, xi_opts);
  auto x_pts = xi_grid[0].reshape({Ngrid, Ngrid}); // [64, 64]
  auto y_pts = xi_grid[1].reshape({Ngrid, Ngrid});
  auto x_acc = x_pts.accessor<double, 2>();
  auto y_acc = y_pts.accessor<double, 2>();
  auto r_acc = u_ref.accessor<double, 2>();
  for (int i = 0; i < Ngrid; ++i)
    for (int j = 0; j < Ngrid; ++j)
      r_acc[i][j] = u_exact(x_acc[i][j], y_acc[i][j]);

  // Relative L2 error
  auto diff     = u_iga - u_ref;
  double err_l2 = diff.norm().item<double>();
  double ref_l2 = u_ref.norm().item<double>();
  double rel_l2 = err_l2 / ref_l2;

  Log() << "\n[Task 1] Geometry: identity  |  Solver: EZSolver (physical Laplacian)\n";
  Log() << "[Task 1] Linear system residual : " << lin_res  << "\n";
  Log() << "[Task 1] Field relative L2 error: " << rel_l2 * 100 << " %\n";
  Log() << "[Task 1] (IGA max - exact max)  : "
        << (u_iga.max() - u_ref.max()).item<double>() << "\n";
  Log() << "[Task 1] IGA  u range: ["
        << u_iga.min().item<double>() << ", " << u_iga.max().item<double>() << "]\n";
  Log() << "[Task 1] Exact u range: ["
        << u_ref.min().item<double>() << ", " << u_ref.max().item<double>() << "]\n";

  finalize();
  return 0;
}
