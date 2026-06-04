/**
 * E3c — Mixed Loss with Normalized Residual  (α = 0.5)
 *
 * Loss:  L_mix = 0.5 * || A(G) û - b(G) ||² / ||b(G)||²   (relative residual)
 *              + 0.5 * || û - u_ref ||²                     (supervised)
 *
 * Normalization rationale:
 *   Raw residual is O(1) while supervised MSE is O(1e-4) — a 4-order gap.
 *   Without normalization α=0.5 is numerically identical to α≈1 (pure residual).
 *   Dividing by ||b||² makes the residual term dimensionless (relative residual),
 *   so both terms are O(1) at initialization and the α weight is meaningful.
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct MixedNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;   // [N_train, 50]
  torch::Tensor u_ref;    // [N_train, 100]
  torch::Tensor A_data;   // [N_train, 100, 100]
  torch::Tensor b_data;   // [N_train, 100]

  torch::Tensor current_u_ref;
  torch::Tensor current_A;
  torch::Tensor current_b;

  static constexpr double alpha = 0.5;

  MixedNet(torch::Tensor G_data_, torch::Tensor u_ref_,
           torch::Tensor A_data_, torch::Tensor b_data_)
      : Base(
            {256, 256},
            {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
            std::tuple{std::array<int64_t, 2>{5, 5}},
            std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_data_)),
        u_ref(std::move(u_ref_)),
        A_data(std::move(A_data_)),
        b_data(std::move(b_data_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    current_u_ref = u_ref[idx];
    current_A     = A_data[idx];
    current_b     = b_data[idx];
    return true;
  }

  // L_mix = α * ||Aû-b||²/||b||²  +  (1-α) * ||û-u_ref||²/||u_ref||²
  // Both terms are relative errors in [0,1] range for reasonable û,
  // so α=0.5 gives genuinely equal weight to physics and supervision.
  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    auto residual      = torch::mv(current_A, outputs) - current_b;
    auto b_norm_sq     = current_b.pow(2).mean().detach()         + 1e-8;
    auto u_ref_norm_sq = current_u_ref.pow(2).mean().detach()     + 1e-8;
    auto L_res = residual.pow(2).mean()                   / b_norm_sq;
    auto L_sup = torch::mse_loss(outputs, current_u_ref)  / u_ref_norm_sq;
    return alpha * L_res + (1.0 - alpha) * L_sup;
  }
};

int main() {
  init();

  // ── Load dataset ──────────────────────────────────────────────────────────
  // tensors[0]=G  [1]=u  [2]=u_val  [3]=A  [4]=b
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_all = tensors[1].to(torch::kDouble); // [500, 100]
  auto A_all = tensors[3].to(torch::kDouble); // [500, 100, 100]
  auto b_all = tensors[4].to(torch::kDouble); // [500, 100]

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train = G_all.slice(0, 0, N_train).contiguous();
  auto u_train = u_all.slice(0, 0, N_train).contiguous();
  auto A_train = A_all.slice(0, 0, N_train).contiguous();
  auto b_train = b_all.slice(0, 0, N_train).contiguous();
  auto G_test  = G_all.slice(0, N_train).contiguous();
  auto u_test  = u_all.slice(0, N_train).contiguous();
  auto A_test  = A_all.slice(0, N_train).contiguous();
  auto b_test  = b_all.slice(0, N_train).contiguous();

  Log() << "[E3c] Dataset: " << N << "  (train=" << N_train
        << ", test=" << N_test << ")\n";

  // ── Build & train ─────────────────────────────────────────────────────────
  MixedNet net(G_train, u_train, A_train, b_train);

  net.options()
      .max_epoch(10000)
      .min_loss(1e-10)
      .min_loss_rel_change(1e-6);

  net.optimizerReset(torch::optim::AdamOptions(1e-3));

  Log() << "[E3c] Training (mixed loss, α=0.5)...\n";
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

    sum_res_l2 += (torch::mv(A_i, u_pred) - b_i).norm().item<double>();
  }

  Log() << "[E3c] Mean relative L2 error : " << sum_rel_l2 / N_test << "\n";
  Log() << "[E3c] Mean residual norm     : " << sum_res_l2 / N_test << "\n";

  net.save("e3c_mixednet.pt");
  Log() << "[E3c] Model saved to e3c_mixednet.pt\n";

  finalize();
  return 0;
}
