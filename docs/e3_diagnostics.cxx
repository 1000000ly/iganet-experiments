/**
 * E3 Diagnostics
 *
 * Diag 2: fill α ∈ (0, 0.05) with α ∈ {0.001, 0.005, 0.01}
 *          same config as E3 sweep (30k epochs, LR decay 1e-3 → 3.1e-5)
 *
 * Diag 3: α = 1.0 with three LR configs
 *   3a. init_lr = 1e-4,  6 segments × 5k epochs  (LR halved each seg)
 *   3b. init_lr = 1e-5,  6 segments × 5k epochs
 *   3c. init_lr = 1e-3, 10 segments × 5k epochs  (50k total)
 */

#include <iganet.h>
#include <solver/ezsolver.hpp>

using namespace iganet;
using Geo = S<UniformBSpline<double, 2, 2, 2>>;
using Var = S<UniformBSpline<double, 1, 3, 3>>;

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
    auto residual = torch::mv(cur_A, outputs) - cur_b;
    auto b_sq     = cur_b.pow(2).mean().detach()     + 1e-8;
    auto u_ref_sq = cur_u_ref.pow(2).mean().detach() + 1e-8;
    auto L_res = residual.pow(2).mean() / b_sq;
    auto L_sup = torch::mse_loss(outputs, cur_u_ref)  / u_ref_sq;
    return alpha * L_res + (1.0 - alpha) * L_sup;
  }
};

// Train: n_segs segments of epochs_per_seg, LR halved each segment
void train_net(E3Net& net, double init_lr,
               int n_segs = 6, int epochs_per_seg = 5000) {
  net.options().min_loss(0.0).min_loss_rel_change(0.0).min_loss_change(0.0);
  double lr = init_lr;
  for (int s = 0; s < n_segs; ++s) {
    net.options().max_epoch(epochs_per_seg);
    net.optimizerReset(torch::optim::AdamOptions(lr));
    net.train();
    lr *= 0.5;
  }
}

// Evaluate 4 metrics on test set
struct Metrics { double coeff_l2, field_l2, residual, bc_viol; };

Metrics evaluate(E3Net& net,
                 const torch::Tensor& G_test,
                 const torch::Tensor& u_test,
                 const torch::Tensor& A_test,
                 const torch::Tensor& b_test,
                 const torch::Tensor& u_grid_test,
                 const utils::TensorArray<2>& xi_grid,
                 int64_t Ngrid, int64_t N_test) {
  double sc = 0, sf = 0, sr = 0, mb = 0;
  for (int64_t i = 0; i < N_test; ++i) {
    net.input<0>().from_tensor(G_test[i]);
    net.eval();
    auto u_pred = net.output<0>().as_tensor();

    double ec = (u_pred - u_test[i]).norm().item<double>();
    double rc = u_test[i].norm().item<double>();
    sc += (rc > 0 ? ec / rc : ec);

    auto fv   = net.output<0>().space<0>().eval(xi_grid);
    auto ug   = fv[0]->reshape({Ngrid, Ngrid});
    double ef = (ug - u_grid_test[i]).norm().item<double>();
    double rf = u_grid_test[i].norm().item<double>();
    sf += (rf > 0 ? ef / rf : ef);

    sr += (torch::mv(A_test[i], u_pred) - b_test[i]).norm().item<double>();

    double bc = std::max({
        ug[0].abs().max().item<double>(),
        ug[Ngrid-1].abs().max().item<double>(),
        ug.select(0,0).abs().max().item<double>(),
        ug.select(0,Ngrid-1).abs().max().item<double>()
    });
    mb = std::max(mb, bc);
  }
  return {sc/N_test, sf/N_test, sr/N_test, mb};
}

int main() {
  init();

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
  constexpr int64_t Ngrid = 64;

  auto G_tr = G_all.slice(0,0,N_train).contiguous();
  auto u_tr = u_all.slice(0,0,N_train).contiguous();
  auto A_tr = A_all.slice(0,0,N_train).contiguous();
  auto b_tr = b_all.slice(0,0,N_train).contiguous();
  auto G_te = G_all.slice(0,N_train).contiguous();
  auto u_te = u_all.slice(0,N_train).contiguous();
  auto A_te = A_all.slice(0,N_train).contiguous();
  auto b_te = b_all.slice(0,N_train).contiguous();
  auto ug_te= u_grid_all.slice(0,N_train).contiguous();

  auto xi_opts = torch::TensorOptions().dtype(torch::kDouble);
  auto xi_1d   = torch::linspace(0.0, 1.0, Ngrid, xi_opts);
  utils::TensorArray<2> xi_grid;
  xi_grid[0] = xi_1d.repeat_interleave(Ngrid);
  xi_grid[1] = xi_1d.repeat(Ngrid);

  Log() << std::fixed << std::setprecision(6);
  Log() << "\n[Diag2] α ∈ {0.001, 0.005, 0.01}  |  30k epochs, init_lr=1e-3\n";
  Log() << "[Diag2] alpha     coeff_l2    field_l2    residual    bc_viol\n";
  Log() << "[Diag2] ------    --------    --------    --------    -------\n";

  for (double alpha : {0.001, 0.005, 0.01}) {
    E3Net net(alpha, G_tr, u_tr, A_tr, b_tr);
    train_net(net, 1e-3, 6, 5000);
    auto m = evaluate(net, G_te, u_te, A_te, b_te, ug_te, xi_grid, Ngrid, N_test);
    Log() << "[Diag2] " << std::setw(6) << alpha
          << "     " << m.coeff_l2 << "    " << m.field_l2
          << "    " << m.residual  << "    " << m.bc_viol << "\n";
  }

  Log() << "\n[Diag3] α=1.0  |  three LR configs\n";
  Log() << "[Diag3] config         coeff_l2    field_l2    residual    bc_viol\n";
  Log() << "[Diag3] ------         --------    --------    --------    -------\n";

  // 3a: init_lr=1e-4, 6 segs × 5k
  {
    E3Net net(1.0, G_tr, u_tr, A_tr, b_tr);
    train_net(net, 1e-4, 6, 5000);
    auto m = evaluate(net, G_te, u_te, A_te, b_te, ug_te, xi_grid, Ngrid, N_test);
    Log() << "[Diag3] lr=1e-4,30k    " << m.coeff_l2
          << "    " << m.field_l2 << "    " << m.residual << "    " << m.bc_viol << "\n";
  }

  // 3b: init_lr=1e-5, 6 segs × 5k
  {
    E3Net net(1.0, G_tr, u_tr, A_tr, b_tr);
    train_net(net, 1e-5, 6, 5000);
    auto m = evaluate(net, G_te, u_te, A_te, b_te, ug_te, xi_grid, Ngrid, N_test);
    Log() << "[Diag3] lr=1e-5,30k    " << m.coeff_l2
          << "    " << m.field_l2 << "    " << m.residual << "    " << m.bc_viol << "\n";
  }

  // 3c: init_lr=1e-3, 10 segs × 5k = 50k epochs
  {
    E3Net net(1.0, G_tr, u_tr, A_tr, b_tr);
    train_net(net, 1e-3, 10, 5000);
    auto m = evaluate(net, G_te, u_te, A_te, b_te, ug_te, xi_grid, Ngrid, N_test);
    Log() << "[Diag3] lr=1e-3,50k    " << m.coeff_l2
          << "    " << m.field_l2 << "    " << m.residual << "    " << m.bc_viol << "\n";
  }

  finalize();
  return 0;
}
