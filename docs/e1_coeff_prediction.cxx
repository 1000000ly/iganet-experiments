/**
 * E1 — Coefficient Prediction Network (Proposed Method)
 *
 * Neural network  G (50) -> [512, 512, 512] -> u_hat (100)
 *
 * Training:
 *   30 000 epochs total, split into 6 segments of 5 000 epochs.
 *   Learning rate halved at each segment boundary (cosine-style warmdown):
 *     seg 0: lr = 1e-3
 *     seg 1: lr = 5e-4
 *     ...
 *     seg 5: lr = 3.125e-5
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct PoissonNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;
  torch::Tensor u_ref;
  torch::Tensor current_u_ref;

  PoissonNet(torch::Tensor G_data_, torch::Tensor u_ref_)
      : Base(
            /* hidden layers */ {512, 512, 512},
            /* activations   */ {{{activation::tanh}},
                                 {{activation::tanh}},
                                 {{activation::tanh}},
                                 {{activation::none}}},
            /* input ncoeffs */ std::tuple{std::array<int64_t, 2>{5, 5}},
            /* output ncoefs */ std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_data_)),
        u_ref(std::move(u_ref_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    current_u_ref = u_ref[idx];
    return true;
  }

  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    return torch::mse_loss(outputs, current_u_ref);
  }
};

int main() {
  init();

  // ── Load dataset ──────────────────────────────────────────────────────────
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all = tensors[0].to(torch::kDouble);
  auto u_all = tensors[1].to(torch::kDouble);

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train = G_all.slice(0, 0, N_train).contiguous();
  auto u_train = u_all.slice(0, 0, N_train).contiguous();
  auto G_test  = G_all.slice(0, N_train).contiguous();
  auto u_test  = u_all.slice(0, N_train).contiguous();

  Log() << "[E1] Dataset: " << N << "  (train=" << N_train
        << ", test=" << N_test << ")\n";
  Log() << "[E1] Architecture: 50 -> 512 -> 512 -> 512 -> 100\n";

  // ── Build & train with LR decay ───────────────────────────────────────────
  PoissonNet net(G_train, u_train);

  // Disable built-in early stopping so each segment runs its full budget
  net.options()
      .min_loss(0.0)
      .min_loss_rel_change(0.0)
      .min_loss_change(0.0);

  constexpr int    N_SEGMENTS          = 6;
  constexpr int    EPOCHS_PER_SEGMENT  = 5000;
  double           lr                  = 1e-3;

  for (int seg = 0; seg < N_SEGMENTS; ++seg) {
    net.options().max_epoch(EPOCHS_PER_SEGMENT);
    net.optimizerReset(torch::optim::AdamOptions(lr));
    Log() << "[E1] Segment " << seg + 1 << "/" << N_SEGMENTS
          << "  lr=" << lr << "\n";
    net.train();
    lr *= 0.5;
  }

  // ── Evaluate on test set ──────────────────────────────────────────────────
  // Load u_grid_test for field-space comparison (tensors[6])
  auto u_grid_all  = tensors[6].to(torch::kDouble);           // [500, 64, 64]
  auto u_grid_test = u_grid_all.slice(0, N_train).contiguous();

  // Precompute 64×64 parametric grid (same as generate_data)
  constexpr int64_t Ngrid = 64;
  auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
  auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);
  utils::TensorArray<2> xi_grid;
  xi_grid[0] = xi_1d.repeat_interleave(Ngrid);
  xi_grid[1] = xi_1d.repeat(Ngrid);

  double sum_coeff_l2 = 0.0;   // Check 1a: coefficient-space (as before)
  double sum_field_l2 = 0.0;   // Check 1b: field-space on 64×64 grid
  double max_bc_viol  = 0.0;   // Check 2:  max boundary violation

  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();

    // ── Check 1a: coefficient-space L2 ──────────────────────────────────────
    auto u_pred_coeff = net.output<0>().as_tensor();
    auto u_true_coeff = u_test[i];
    double ec  = (u_pred_coeff - u_true_coeff).norm().item<double>();
    double rfc = u_true_coeff.norm().item<double>();
    sum_coeff_l2 += (rfc > 0.0 ? ec / rfc : ec);

    // ── Check 1b: field-space L2 (evaluate predicted spline on 64×64) ───────
    auto field_vals = net.output<0>().space<0>().eval(xi_grid);
    auto u_pred_grid = field_vals[0]->reshape({Ngrid, Ngrid});  // [64, 64]
    auto u_true_grid = u_grid_test[i];                          // [64, 64]
    double ef  = (u_pred_grid - u_true_grid).norm().item<double>();
    double rff = u_true_grid.norm().item<double>();
    sum_field_l2 += (rff > 0.0 ? ef / rff : ef);

    // ── Check 2: boundary violation ─────────────────────────────────────────
    // Boundary of 64×64 grid = edges at xi=0,1 → rows/cols [0] and [63]
    // For BCs u=0, predicted values at edges should be ≈ 0
    auto grid = u_pred_grid;
    double bc = std::max({
        grid[0].abs().max().item<double>(),        // xi2 = 0  (row 0)
        grid[Ngrid-1].abs().max().item<double>(),  // xi2 = 1  (row 63)
        grid.select(0,0).abs().max().item<double>(),       // xi1 = 0  (col 0)
        grid.select(0,Ngrid-1).abs().max().item<double>()  // xi1 = 1  (col 63)
    });
    max_bc_viol = std::max(max_bc_viol, bc);
  }

  Log() << "[E1] Mean coeff-space rel-L2 : " << sum_coeff_l2 / N_test << "\n";
  Log() << "[E1] Mean field-space rel-L2 : " << sum_field_l2 / N_test << "\n";
  Log() << "[E1] Max boundary violation  : " << max_bc_viol  << "\n";

  // ── Save ──────────────────────────────────────────────────────────────────
  net.save("e1_poissonnet.pt");
  Log() << "[E1] Model saved to e1_poissonnet.pt\n";

  finalize();
  return 0;
}
