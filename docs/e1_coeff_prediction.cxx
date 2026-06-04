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
