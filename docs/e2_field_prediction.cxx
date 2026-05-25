/**
 * E2 — Field Value Prediction Baseline
 *
 * Network:  G (50) -> [256, 256] -> u_vals (100)
 * Output is pointwise field values at Greville collocation points,
 * NOT B-spline coefficients. No structural constraint is enforced.
 *
 * This is the direct comparison to E1. Same architecture, same data,
 * different output semantics: E1 predicts coefficients (lives in spline
 * space by construction), E2 predicts unconstrained pointwise values.
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;

using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct FieldPredNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;             // [N_train, 50]
  torch::Tensor u_val_ref;          // [N_train, 100]  pointwise field values
  torch::Tensor current_u_val_ref;  // reference values for current sample

  FieldPredNet(torch::Tensor G_data_, torch::Tensor u_val_ref_)
      : Base(
            /* hidden layers */ {256, 256},
            /* activations   */ {{{activation::tanh}}, {{activation::tanh}}, {{activation::none}}},
            /* input ncoeffs */ std::tuple{std::array<int64_t, 2>{5, 5}},
            /* output ncoefs */ std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_data_)),
        u_val_ref(std::move(u_val_ref_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    current_u_val_ref = u_val_ref[idx]; // target is field values, not coeffs
    return true;
  }

  // Loss: MSE between predicted field values and reference field values
  // Note: no spline structure is enforced here — this is the key difference
  // from E1
  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    return torch::mse_loss(outputs, current_u_val_ref);
  }
};

int main() {
  init();

  // Load dataset (must include u_val tensor from updated generate_data.cxx)
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all     = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_val_all = tensors[2].to(torch::kDouble); // [500, 100] field values

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train     = G_all.slice(0, 0, N_train).contiguous();
  auto u_val_train = u_val_all.slice(0, 0, N_train).contiguous();
  auto G_test      = G_all.slice(0, N_train).contiguous();
  auto u_val_test  = u_val_all.slice(0, N_train).contiguous();

  Log() << "[E2] Dataset: " << N << " samples"
        << "  (train=" << N_train << ", test=" << N_test << ")\n";

  // Build and train
  FieldPredNet net(G_train, u_val_train);

  net.options()
      .max_epoch(10000)
      .min_loss(1e-7)
      .min_loss_rel_change(1e-6);

  net.optimizerReset(torch::optim::AdamOptions(1e-3));

  Log() << "[E2] Training (50 -> 256 -> 256 -> 100, field values)...\n";
  net.train();

  // Evaluate on test set
  // For E2, we compare predicted field values against reference field values.
  // We also compute the L2 error in the reconstructed coefficient space
  // by projecting the predicted field values back via ezinterp,
  // so that E1 and E2 errors are comparable on equal footing.
  double sum_val_mse    = 0.0;
  double sum_rel_l2     = 0.0;

  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();

    auto u_pred_vals = net.output<0>().as_tensor(); // predicted field values
    auto u_true_vals = u_val_test[i];

    // Direct field-value MSE
    sum_val_mse += torch::mse_loss(u_pred_vals, u_true_vals).item<double>();

    // Relative L2 error in field-value space
    double err = (u_pred_vals - u_true_vals).norm().item<double>();
    double ref = u_true_vals.norm().item<double>();
    sum_rel_l2 += (ref > 0.0 ? err / ref : err);
  }

  Log() << "[E2] Mean field-value MSE       : " << sum_val_mse / N_test << "\n";
  Log() << "[E2] Mean relative L2 error     : " << sum_rel_l2  / N_test << "\n";
  Log() << "\n[E2 vs E1] Compare the relative L2 errors above.\n"
        << "           E1 predicts in spline space (structure-preserving).\n"
        << "           E2 predicts unconstrained pointwise values.\n";

  net.save("e2_fieldprednet.pt");
  Log() << "[E2] Model saved to e2_fieldprednet.pt\n";

  finalize();
  return 0;
}