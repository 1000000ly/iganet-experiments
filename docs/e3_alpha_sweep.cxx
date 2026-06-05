/**
 * E3 — Loss Ablation (normalized mixed loss)
 *
 * L(α) = α * ||A û - b||² / ||b||²
 *       + (1-α) * ||û - u_ref||² / ||u_ref||²
 *
 * Architecture and training identical to E1:
 *   50 -> 512 -> 512 -> 512 -> 100
 *   30 000 epochs, 6 segments, LR: 1e-3 * 0.5^seg
 *
 * α ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
 *
 * Metrics per α:
 *   coeff_l2   — ||û - u_ref|| / ||u_ref||
 *   field_l2   — field-space relative L2 on 64×64 grid
 *   residual   — ||A û - b|| (unnormalized)
 *   bc_viol    — max |û| on boundary of parametric domain
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;
using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

// ── Network ───────────────────────────────────────────────────────────────────
struct E3Net
    : public IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>> {

  using Base = IgANet<torch::optim::Adam, std::tuple<Geo>, std::tuple<Var>>;

  torch::Tensor G_data, u_ref, A_data, b_data;
  torch::Tensor cur_u_ref, cur_A, cur_b;
  double alpha;

  E3Net(double alpha_,
        torch::Tensor G_, torch::Tensor u_,
        torch::Tensor A_, torch::Tensor b_)
      : Base(
            {512, 512, 512},
            {{{activation::tanh}}, {{activation::tanh}},
             {{activation::tanh}}, {{activation::none}}},
            std::tuple{std::array<int64_t, 2>{5, 5}},
            std::tuple{std::array<int64_t, 2>{10, 10}},
            init::greville),
        G_data(std::move(G_)), u_ref(std::move(u_)),
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
    auto residual  = torch::mv(cur_A, outputs) - cur_b;
    auto b_sq      = cur_b.pow(2).mean().detach()      + 1e-8;
    auto u_ref_sq  = cur_u_ref.pow(2).mean().detach()  + 1e-8;
    auto L_res = residual.pow(2).mean() / b_sq;
    auto L_sup = torch::mse_loss(outputs, cur_u_ref)   / u_ref_sq;
    return alpha * L_res + (1.0 - alpha) * L_sup;
  }
};

// ── Helper: train one network ─────────────────────────────────────────────────
void train_e3(E3Net& net) {
  net.options().min_loss(0.0).min_loss_rel_change(0.0).min_loss_change(0.0);
  double lr = 1e-3;
  for (int seg = 0; seg < 6; ++seg) {
    net.options().max_epoch(5000);
    net.optimizerReset(torch::optim::AdamOptions(lr));
    net.train();
    lr *= 0.5;
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
  init();

  // Load dataset  [0]=G [1]=u [2]=u_val [3]=A [4]=b [5]=G_grid [6]=u_grid
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, "dataset.pt");
  auto G_all      = tensors[0].to(torch::kDouble);
  auto u_all      = tensors[1].to(torch::kDouble);
  auto A_all      = tensors[3].to(torch::kDouble);
  auto b_all      = tensors[4].to(torch::kDouble);
  auto u_grid_all = tensors[6].to(torch::kDouble);

  int64_t N       = G_all.size(0);
  int64_t N_train = static_cast<int64_t>(N * 0.8);
  int64_t N_test  = N - N_train;

  auto G_train      = G_all.slice(0, 0, N_train).contiguous();
  auto u_train      = u_all.slice(0, 0, N_train).contiguous();
  auto A_train      = A_all.slice(0, 0, N_train).contiguous();
  auto b_train      = b_all.slice(0, 0, N_train).contiguous();
  auto G_test       = G_all.slice(0, N_train).contiguous();
  auto u_test       = u_all.slice(0, N_train).contiguous();
  auto A_test       = A_all.slice(0, N_train).contiguous();
  auto b_test       = b_all.slice(0, N_train).contiguous();
  auto u_grid_test  = u_grid_all.slice(0, N_train).contiguous();

  // Parametric 64×64 grid
  constexpr int64_t Ngrid = 64;
  auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
  auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);
  utils::TensorArray<2> xi_grid;
  xi_grid[0] = xi_1d.repeat_interleave(Ngrid);
  xi_grid[1] = xi_1d.repeat(Ngrid);

  // Header
  Log() << "\n[E3] Architecture: 50->512->512->512->100  |  30k epochs  |  LR decay\n";
  Log() << "[E3] N_train=" << N_train << "  N_test=" << N_test << "\n\n";
  Log() << std::fixed << std::setprecision(6);
  Log() << "[E3] alpha     coeff_l2    field_l2    residual    bc_viol\n";
  Log() << "[E3] ------    --------    --------    --------    -------\n";

  for (double alpha : {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}) {

    E3Net net(alpha, G_train, u_train, A_train, b_train);
    train_e3(net);

    double sum_coeff = 0, sum_field = 0, sum_res = 0;
    double max_bc = 0;

    for (int64_t i = 0; i < N_test; ++i) {
      net.input<0>().from_tensor(G_test[i]);
      net.eval();

      auto u_pred = net.output<0>().as_tensor();

      // coeff-space L2
      double ec = (u_pred - u_test[i]).norm().item<double>();
      double rc = u_test[i].norm().item<double>();
      sum_coeff += (rc > 0 ? ec / rc : ec);

      // field-space L2
      auto fv      = net.output<0>().space<0>().eval(xi_grid);
      auto u_pgrid = fv[0]->reshape({Ngrid, Ngrid});
      double ef    = (u_pgrid - u_grid_test[i]).norm().item<double>();
      double rf    = u_grid_test[i].norm().item<double>();
      sum_field   += (rf > 0 ? ef / rf : ef);

      // residual ||A û - b||
      sum_res += (torch::mv(A_test[i], u_pred) - b_test[i])
                     .norm().item<double>();

      // boundary violation (edges of 64×64 grid)
      double bc = std::max({
          u_pgrid[0].abs().max().item<double>(),
          u_pgrid[Ngrid-1].abs().max().item<double>(),
          u_pgrid.select(0, 0).abs().max().item<double>(),
          u_pgrid.select(0, Ngrid-1).abs().max().item<double>()
      });
      max_bc = std::max(max_bc, bc);
    }

    Log() << "[E3] " << std::setw(6) << alpha
          << "     " << sum_coeff / N_test
          << "    " << sum_field  / N_test
          << "    " << sum_res    / N_test
          << "    " << max_bc     << "\n";
  }

  finalize();
  return 0;
}
