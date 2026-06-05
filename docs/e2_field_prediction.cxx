/**
 * E2 — Field Value Prediction (Greville Points)
 *
 * Architecture identical to E1: 50 -> 512 -> 512 -> 512 -> 100
 * Training label: u_val (tensors[2]) — field values at 100 Greville points,
 *                 NOT B-spline coefficients.
 *
 * Evaluation:
 *   field_l2   — direct comparison: ||u_pred_val - u_val_ref|| / ||u_val_ref||
 *   coeff_l2   — back-project via M^{-1}: u_coeff = M^{-1} u_pred_val,
 *                then compare with u_ref (true coefficients)
 *   bc_viol    — max |u_pred_val| at boundary Greville indices
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;
using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

struct FieldNet
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data;
  torch::Tensor u_val_ref;      // [N, 100] field values at Greville points
  torch::Tensor cur_u_val;

  FieldNet(torch::Tensor G_, torch::Tensor u_val_)
      : Base(
            {512, 512, 512},
            {{{activation::tanh}}, {{activation::tanh}},
             {{activation::tanh}}, {{activation::none}}},
            std::tuple{std::array<int64_t, 2>{5, 5}},
            std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_)),
        u_val_ref(std::move(u_val_)) {}

  bool epoch(int64_t) override {
    int64_t idx = torch::randint(G_data.size(0), {1}).item<int64_t>();
    this->input<0>().from_tensor(G_data[idx]);
    cur_u_val = u_val_ref[idx];
    return true;
  }

  // MSE on field values (no spline structure in the loss)
  torch::Tensor loss(const torch::Tensor &outputs, int64_t) override {
    return torch::mse_loss(outputs, cur_u_val);
  }
};

int main() {
  init();

  // ── Load dataset ──────────────────────────────────────────────────────────
  // tensors[0]=G [1]=u_coeff [2]=u_val [3]=A [4]=b [5]=G_grid [6]=u_grid
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all     = tensors[0].to(torch::kDouble); // [500, 50]
  auto u_all     = tensors[1].to(torch::kDouble); // [500, 100]  true coefficients
  auto u_val_all = tensors[2].to(torch::kDouble); // [500, 100]  Greville field values

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train     = G_all.slice(0, 0, N_train).contiguous();
  auto u_val_train = u_val_all.slice(0, 0, N_train).contiguous();
  auto G_test      = G_all.slice(0, N_train).contiguous();
  auto u_test      = u_all.slice(0, N_train).contiguous();      // true coefficients
  auto u_val_test  = u_val_all.slice(0, N_train).contiguous();  // true field values

  Log() << "[E2] Dataset: " << N << "  (train=" << N_train << ", test=" << N_test << ")\n";
  Log() << "[E2] Architecture: 50 -> 512 -> 512 -> 512 -> 100 (field values)\n";

  // ── Build interpolation matrix M (Greville collocation) ───────────────────
  // Used for back-projection: u_coeff = M^{-1} * u_pred_val
  // M is the B-spline basis evaluated at Greville points — geometry-independent
  Geo G_id({5, 5}, init::linear);
  Var u_template({10, 10});
  auto zero_rhs = [](const std::array<torch::Tensor, 2>& xi)
      -> std::array<torch::Tensor, 1> {
    return {torch::zeros_like(xi[0])};
  };
  EZInterpolation interp(G_id, u_template, zero_rhs);
  interp.init();
  interp.assemble();
  auto M     = interp.lhs().to_dense();  // [100, 100]
  auto M_inv = M.inverse();              // precomputed once

  // ── Boundary Greville mask ────────────────────────────────────────────────
  // greville(false) returns TensorArray<2>: [ξ₁_coords, ξ₂_coords] each [100]
  auto g_pts = u_template.space<0>().greville(false);
  auto bm = (g_pts[0].abs() < 1e-10)
           | (torch::abs(g_pts[0] - 1.0) < 1e-10)
           | (g_pts[1].abs() < 1e-10)
           | (torch::abs(g_pts[1] - 1.0) < 1e-10);  // bool [100]
  Log() << "[E2] Boundary Greville points: " << bm.sum().item<int64_t>() << " / 100\n";

  // ── Train ─────────────────────────────────────────────────────────────────
  FieldNet net(G_train, u_val_train);
  net.options().min_loss(0.0).min_loss_rel_change(0.0).min_loss_change(0.0);

  double lr = 1e-3;
  for (int seg = 0; seg < 6; ++seg) {
    net.options().max_epoch(5000);
    net.optimizerReset(torch::optim::AdamOptions(lr));
    Log() << "[E2] Segment " << seg + 1 << "/6  lr=" << lr << "\n";
    net.train();
    lr *= 0.5;
  }

  // ── Evaluate ──────────────────────────────────────────────────────────────
  double sum_field_l2 = 0.0;
  double sum_coeff_l2 = 0.0;
  double max_bc_viol  = 0.0;

  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();

    // E2 output: 100 predicted field values at Greville points
    auto u_pred_val = net.output<0>().as_tensor();  // [100]

    // Field-space L2 (direct)
    double ef = (u_pred_val - u_val_test[i]).norm().item<double>();
    double rf = u_val_test[i].norm().item<double>();
    sum_field_l2 += (rf > 0.0 ? ef / rf : ef);

    // Coefficient-space L2 (back-projection via M^{-1})
    auto u_pred_coeff = torch::mv(M_inv, u_pred_val);
    double ec = (u_pred_coeff - u_test[i]).norm().item<double>();
    double rc = u_test[i].norm().item<double>();
    sum_coeff_l2 += (rc > 0.0 ? ec / rc : ec);

    // Boundary violation
    double bc = u_pred_val.masked_select(bm).abs().max().item<double>();
    max_bc_viol = std::max(max_bc_viol, bc);
  }

  Log() << "\n[E2] Mean field-space  rel-L2 : " << sum_field_l2 / N_test << "\n";
  Log() << "[E2] Mean coeff-space  rel-L2 : " << sum_coeff_l2 / N_test << "\n";
  Log() << "[E2] Max boundary violation   : " << max_bc_viol             << "\n";

  net.save("e2_fieldnet.pt");
  Log() << "[E2] Model saved to e2_fieldnet.pt\n";

  finalize();
  return 0;
}
