#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <splines/bspline.hpp>

using namespace iganet;

int main() {
  init();

  using Geo = S<UniformBSpline<double, 2, 2, 2>>;
  using Var = S<UniformBSpline<double, 1, 3, 3>>;

  auto rhs = [](const std::array<torch::Tensor, 2>& xi)
      -> std::array<torch::Tensor, 1> {
    return {torch::ones_like(xi[0])};
  };

  const int N = 500;
  std::vector<torch::Tensor> G_dataset, u_dataset, u_val_dataset,
                              A_dataset, b_dataset;

  for (int i = 0; i < N; ++i) {
    Geo G({5, 5}, init::linear);

    auto& sp = G.space<0>();
    for (short_t d = 0; d < sp.geoDim(); ++d) {
      auto c = sp.coeffs(d).view({5, 5});
      c.slice(0, 1, 4).slice(1, 1, 4) +=
          0.08 * torch::randn({3, 3}, sp.coeffs(d).options());
    }

    // Use EZSolver directly so we can retrieve A and b after assembly
    Var u_template({10, 10});
    EZSolver solver(G, u_template, rhs);
    solver.init();
    solver.assemble();
    auto u_coeffs = solver.solve().clone();       // [100]
    auto A        = solver.lhs().to_dense();      // [100, 100]  sparse -> dense
    auto b        = solver.rhs().clone();         // [100]

    G_dataset.push_back(G.as_tensor().clone());
    u_dataset.push_back(u_coeffs);
    A_dataset.push_back(A);
    b_dataset.push_back(b);

    // Field values at Greville points (for E2)
    Var u_solved({10, 10});
    u_solved.space<0>().from_tensor(u_coeffs);
    auto greville_pts = u_solved.space<0>().greville(false);
    auto u_vals = u_solved.space<0>().eval(greville_pts);
    u_val_dataset.push_back(u_vals[0]->clone());
  }

  auto G_tensor     = torch::stack(G_dataset);      // [N, 50]
  auto u_tensor     = torch::stack(u_dataset);      // [N, 100]
  auto u_val_tensor = torch::stack(u_val_dataset);  // [N, 100]
  auto A_tensor     = torch::stack(A_dataset);      // [N, 100, 100]
  auto b_tensor     = torch::stack(b_dataset);      // [N, 100]

  // tensors[0]=G  [1]=u  [2]=u_val  [3]=A  [4]=b
  torch::save({G_tensor, u_tensor, u_val_tensor, A_tensor, b_tensor},
              "dataset.pt");

  Log() << "Dataset saved: " << N << " samples\n"
        << "G shape:     " << G_tensor.sizes()     << "\n"
        << "u shape:     " << u_tensor.sizes()     << "\n"
        << "u_val shape: " << u_val_tensor.sizes() << "\n"
        << "A shape:     " << A_tensor.sizes()     << "\n"
        << "b shape:     " << b_tensor.sizes()     << "\n";

  finalize();
  return 0;
}
