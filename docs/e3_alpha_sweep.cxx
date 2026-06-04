/**
 * E3 Alpha Sweep
 *
 * Trains one MixedNet per alpha in {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
 * using normalized mixed loss:
 *
 *   L = α * ||Aû-b||²/||b||²  +  (1-α) * ||û-u_ref||²/||u_ref||²
 *
 * Also reports the condition number of A (mean over training samples)
 * as evidence for why residual-based training is hard.
 *
 * Output: plain table → copy into report or script for plotting.
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

// ── Network ───────────────────────────────────────────────────────────────────
struct MixedNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data, u_ref, A_data, b_data;
  torch::Tensor cur_u_ref, cur_A, cur_b;
  double alpha;

  MixedNet(double alpha_,
           torch::Tensor G_, torch::Tensor u_ref_,
           torch::Tensor A_, torch::Tensor b_)
      : Base(
            {256, 256},
            {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
            std::tuple{std::array<int64_t, 2>{5, 5}},
            std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_)), u_ref(std::move(u_ref_)),
        A_data(std::move(A_)), b_data(std::move(b_)),
        alpha(alpha_) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    cur_u_ref = u_ref[idx];
    cur_A     = A_data[idx];
    cur_b     = b_data[idx];
    return true;
  }

  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    auto residual      = torch::mv(cur_A, outputs) - cur_b;
    auto b_sq          = cur_b.pow(2).mean().detach()     + 1e-8;
    auto u_ref_sq      = cur_u_ref.pow(2).mean().detach() + 1e-8;
    auto L_res = residual.pow(2).mean() / b_sq;
    auto L_sup = torch::mse_loss(outputs, cur_u_ref) / u_ref_sq;
    return alpha * L_res + (1.0 - alpha) * L_sup;
  }
};

// ── Condition number ──────────────────────────────────────────────────────────
double mean_condition_number(const torch::Tensor &A_all, int n_samples = 20) {
  double sum = 0.0;
  int64_t N  = A_all.size(0);
  for (int i = 0; i < n_samples; ++i) {
    auto A_i = A_all[i % N].to(torch::kDouble);
    auto S   = std::get<1>(torch::svd(A_i, /*some_vectors=*/false));
    double cond = (S[0] / S[S.size(0) - 1]).item<double>();
    sum += cond;
  }
  return sum / n_samples;
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
  init();

  // Load dataset  (tensors[0]=G [1]=u [2]=u_val [3]=A [4]=b)
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all = tensors[0].to(torch::kDouble);
  auto u_all = tensors[1].to(torch::kDouble);
  auto A_all = tensors[3].to(torch::kDouble);
  auto b_all = tensors[4].to(torch::kDouble);

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

  // ── Condition number ────────────────────────────────────────────────────────
  double cond = mean_condition_number(A_all);
  Log() << "\n[E3_sweep] Mean condition number of A (n=20): " << cond << "\n\n";

  // ── Alpha sweep ─────────────────────────────────────────────────────────────
  std::vector<double> alphas = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0};

  Log() << "[E3_sweep] alpha    rel_L2_error    residual_norm\n";
  Log() << "[E3_sweep] ------   ------------    -------------\n";

  for (double alpha : alphas) {
    MixedNet net(alpha, G_train, u_train, A_train, b_train);

    net.options()
        .max_epoch(10000)
        .min_loss(1e-10)
        .min_loss_rel_change(1e-6);
    net.optimizerReset(torch::optim::AdamOptions(1e-3));
    net.train();

    double sum_l2  = 0.0;
    double sum_res = 0.0;
    for (int64_t i = 0; i < N_test; ++i) {
      net.input<0>().from_tensor(G_test[i]);
      net.eval();
      auto u_pred = net.output<0>().as_tensor();
      auto u_true = u_test[i];

      double err = (u_pred - u_true).norm().item<double>();
      double ref = u_true.norm().item<double>();
      sum_l2  += (ref > 0.0 ? err / ref : err);
      sum_res += (torch::mv(A_test[i], u_pred) - b_test[i]).norm().item<double>();
    }

    double mean_l2  = sum_l2  / N_test;
    double mean_res = sum_res / N_test;

    Log() << "[E3_sweep] " << std::fixed << std::setprecision(2) << alpha
          << "       " << std::setprecision(6) << mean_l2
          << "      " << mean_res << "\n";
  }

  Log() << "\n[E3_sweep] Done.\n";
  finalize();
  return 0;
}
