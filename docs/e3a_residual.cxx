/**
 * E3a — Pure Residual Loss  (α = 1)
 *
 * Loss:  L_res = || A(G) û - b(G) ||²
 *
 * No reference labels used during training.
 * Tests whether the PDE residual signal alone is sufficient.
 *
 * Evaluation reports both:
 *   - Relative L2 coefficient error  (same metric as E1/E2)
 *   - Mean residual norm  ||A û - b||  (physical consistency)
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct ResidualNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;   // [N_train, 50]
  torch::Tensor A_data;   // [N_train, 100, 100]
  torch::Tensor b_data;   // [N_train, 100]

  torch::Tensor current_A;
  torch::Tensor current_b;

  ResidualNet(torch::Tensor G_data_, torch::Tensor A_data_, torch::Tensor b_data_)
      : Base(
            {256, 256},
            {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
            std::tuple{std::array<int64_t, 2>{5, 5}},
            std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_data_)),
        A_data(std::move(A_data_)),
        b_data(std::move(b_data_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    current_A = A_data[idx]; // [100, 100]
    current_b = b_data[idx]; // [100]
    return true;
  }

  // L_res = || A û - b ||²  (mean over 100 equations)
  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    auto residual = torch::mv(current_A, outputs) - current_b;
    return residual.pow(2).mean();
  }
};

int main() {
  init();

  // ── Load dataset ──────────────────────────────────────────────────────────
  // tensors[0]=G  [1]=u  [2]=u_val  [3]=A  [4]=b
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_all = tensors[1].to(torch::kDouble); // [500, 100]  (eval only)
  auto A_all = tensors[3].to(torch::kDouble); // [500, 100, 100]
  auto b_all = tensors[4].to(torch::kDouble); // [500, 100]

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train = G_all.slice(0, 0, N_train).contiguous();
  auto A_train = A_all.slice(0, 0, N_train).contiguous();
  auto b_train = b_all.slice(0, 0, N_train).contiguous();
  auto G_test  = G_all.slice(0, N_train).contiguous();
  auto u_test  = u_all.slice(0, N_train).contiguous();
  auto A_test  = A_all.slice(0, N_train).contiguous();
  auto b_test  = b_all.slice(0, N_train).contiguous();

  Log() << "[E3a] Dataset: " << N << "  (train=" << N_train
        << ", test=" << N_test << ")\n";

  // ── Build & train ─────────────────────────────────────────────────────────
  ResidualNet net(G_train, A_train, b_train);

  net.options()
      .max_epoch(10000)
      .min_loss(1e-10)
      .min_loss_rel_change(1e-6);

  net.optimizerReset(torch::optim::AdamOptions(1e-3));

  Log() << "[E3a] Training (pure residual loss, α=1)...\n";
  net.train();

  // ── Evaluate ──────────────────────────────────────────────────────────────
  double sum_rel_l2  = 0.0;
  double sum_res_l2  = 0.0;

  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();

    auto u_pred = net.output<0>().as_tensor();
    auto u_true = u_test[i];
    auto A_i    = A_test[i];
    auto b_i    = b_test[i];

    double err = (u_pred - u_true).norm().item<double>();
    double ref = u_true.norm().item<double>();
    sum_rel_l2 += (ref > 0.0 ? err / ref : err);

    // Physical consistency: residual norm ||A û - b||
    sum_res_l2 += (torch::mv(A_i, u_pred) - b_i).norm().item<double>();
  }

  Log() << "[E3a] Mean relative L2 error : " << sum_rel_l2 / N_test << "\n";
  Log() << "[E3a] Mean residual norm     : " << sum_res_l2 / N_test << "\n";

  net.save("e3a_residualnet.pt");
  Log() << "[E3a] Model saved to e3a_residualnet.pt\n";

  finalize();
  return 0;
}
