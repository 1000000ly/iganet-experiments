/**
 * E1 — Coefficient Prediction Network (Proposed Method)
 *
 * Neural network  G (50) -> [256, 256] -> u_hat (100)
 * Maps B-spline geometry control points directly to IGA solution
 * coefficients for the Poisson problem on perturbed unit-square domains.
 *
 * Uses the current iganet::IgANet API (not the deprecated v1 variant):
 *   IgANet<Optimizer, std::tuple<Inputs...>, std::tuple<Outputs...>>
 *   input<0>()  — Geo function space (50 parameters)
 *   output<0>() — Var function space (100 parameters)
 *
 * Dataset: dataset.pt  (generate with generate_data.cxx)
 *   tensors[0]: G shape [N, 50]   — flattened 5x5x2 geometry control points
 *   tensors[1]: u shape [N, 100]  — 10x10 B-spline solution coefficients
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct PoissonNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;        // [N_train, 50]
  torch::Tensor u_ref;         // [N_train, 100]
  torch::Tensor current_u_ref; // reference coefficient for the current sample

  PoissonNet(torch::Tensor G_data_, torch::Tensor u_ref_)
      : Base(
            /* hidden layers */ {256, 256},
            /* activations   */ {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
            /* input ncoeffs */ std::tuple{std::array<int64_t, 2>{5, 5}},
            /* output ncoefs */ std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_data_)),
        u_ref(std::move(u_ref_)) {}

  // Load a random training sample; return true so train() re-queries inputs()
  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    current_u_ref = u_ref[idx];
    return true;
  }

  // Supervised loss: MSE between predicted and IGA reference coefficients
  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    return torch::mse_loss(outputs, current_u_ref);
  }
};

int main() {
  init();

  // ── Load dataset ──────────────────────────────────────────────────────────
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_all = tensors[1].to(torch::kDouble); // [500, 100]

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8); // 400
  int64_t N_test  = N - N_train;                    // 100

  auto G_train = G_all.slice(0, 0, N_train).contiguous();
  auto u_train = u_all.slice(0, 0, N_train).contiguous();
  auto G_test  = G_all.slice(0, N_train).contiguous();
  auto u_test  = u_all.slice(0, N_train).contiguous();

  Log() << "[E1] Dataset: " << N << " samples"
        << "  (train=" << N_train << ", test=" << N_test << ")\n";

  // ── Build & train ─────────────────────────────────────────────────────────
  PoissonNet net(G_train, u_train);

  net.options()
      .max_epoch(10000)
      .min_loss(1e-7)
      .min_loss_rel_change(1e-6);

  net.optimizerReset(torch::optim::AdamOptions(1e-3));

  Log() << "[E1] Training (50 -> 256 -> 256 -> 100)...\n";
  net.train();

  // ── Evaluate on test set ──────────────────────────────────────────────────
  double sum_rel_l2 = 0.0;
  double sum_mse    = 0.0;

  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();

    auto u_pred = net.output<0>().as_tensor();
    auto u_true = u_test[i];

    double err = (u_pred - u_true).norm().item<double>();
    double ref = u_true.norm().item<double>();

    sum_rel_l2 += (ref > 0.0 ? err / ref : err);
    sum_mse    += torch::mse_loss(u_pred, u_true).item<double>();
  }

  Log() << "[E1] Mean relative L2 error : " << sum_rel_l2 / N_test << "\n";
  Log() << "[E1] Mean MSE               : " << sum_mse    / N_test << "\n";

  // ── Save ──────────────────────────────────────────────────────────────────
  net.save("e1_poissonnet.pt");
  Log() << "[E1] Model saved to e1_poissonnet.pt\n";

  finalize();
  return 0;
}
