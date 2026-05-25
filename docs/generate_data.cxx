#include <iganet.h>
#include <solver/ezsolver.hpp>
#include <splines/bspline.hpp>

using namespace iganet;

int main() {
  init();

  // Function space types
  // GeometryMap: 2D -> 2D, degree (2,2), 5x5 control points
  using Geo = S<UniformBSpline<double, 2, 2, 2>>;
  // Variable: scalar field (geoDim=1), degree (3,3), 10x10 coefficients
  using Var = S<UniformBSpline<double, 1, 3, 3>>;

  // Source term: f = 1 (constant RHS, simplest Poisson case)
  auto rhs = [](const std::array<torch::Tensor, 2>& xi)
      -> std::array<torch::Tensor, 1> {
    return {torch::ones_like(xi[0])};
  };

  const int N = 500;
  std::vector<torch::Tensor> G_dataset, u_dataset, u_val_dataset;

  for (int i = 0; i < N; ++i) {
    // Initialize geometry as the identity map on [0,1]^2
    Geo G({5, 5}, init::linear);

    // Perturb interior control points only
    // Boundary control points are kept fixed to avoid geometry degeneracy
    auto& sp = G.space<0>();
    for (short_t d = 0; d < sp.geoDim(); ++d) {
      auto c = sp.coeffs(d).view({5, 5});
      c.slice(0, 1, 4).slice(1, 1, 4) +=
          0.08 * torch::randn({3, 3}, sp.coeffs(d).options());
    }

    // Solve Poisson on the perturbed geometry
    // Returns B-spline coefficients of u, shape [100]
    Var u_template({10, 10});
    auto u_coeffs = ezpoisson(G, u_template, rhs);

    // Store flattened control points of G and solution coefficients
    G_dataset.push_back(G.as_tensor().clone());  // shape [50] = 5x5x2
    u_dataset.push_back(u_coeffs);               // shape [100]

    // Evaluate u at the Greville abscissae of the variable space
    // to get pointwise field values (used by E2 baseline)
    Var u_solved({10, 10});
    u_solved.space<0>().from_tensor(u_coeffs);
    auto greville_pts = u_solved.space<0>().greville(false);
    auto u_vals = u_solved.space<0>().eval(greville_pts);
    u_val_dataset.push_back(u_vals[0]->clone());  // shape [100], field values
  }

  // Stack into training tensors
  auto G_tensor     = torch::stack(G_dataset);      // [N, 50]
  auto u_tensor     = torch::stack(u_dataset);      // [N, 100]
  auto u_val_tensor = torch::stack(u_val_dataset);  // [N, 100]

  torch::save({G_tensor, u_tensor, u_val_tensor}, "dataset.pt");

  Log() << "Dataset saved: " << N << " samples\n"
        << "G shape:     " << G_tensor.sizes()     << "\n"
        << "u shape:     " << u_tensor.sizes()     << "\n"
        << "u_val shape: " << u_val_tensor.sizes() << "\n";

  finalize();
  return 0;
}
