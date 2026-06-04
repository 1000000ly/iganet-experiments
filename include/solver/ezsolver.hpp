/**
   @file solver/ezinterp.hpp

   @brief Isogeometric analysis solver

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <net/igabase.hpp>
#include <solver/igasolver.hpp>
#include <splines/bspline.hpp>
#include <utils/matrix.hpp>

namespace iganet {

/// @brief Easy-to-use solver base class
///
/// This abstract class implements an easy-to-use solver that takes a
/// geometry map and a variable function space as inputs. Note that
/// this is an abstract class. A derived class needs to implement the
/// member functions assembleLhs and assembleRhs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>
class EZSolverBase : public iganet::IgASolver<std::tuple<GeometryMap>, std::tuple<Variable>>,
                     public iganet::IgANetCustomizable<std::tuple<GeometryMap>, std::tuple<Variable>> {
  
protected:
  /// @brief Type of the base class
  using Base = iganet::IgASolver<std::tuple<GeometryMap>, std::tuple<Variable>>;
  
  /// @brief Collocation points
  Base::template collPts_t<0> collPts_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<std::tuple<GeometryMap>, std::tuple<Variable>>;

  /// @brief Knot indices of the geometry map
  Customizable::template input_interior_knot_indices_t<0> G_knot_indices_;

  /// @brief Knot indices of the geometry map at the boundary
  Customizable::template input_boundary_knot_indices_t<0> G_knot_indices_boundary_;

  /// @brief Knot indices of variables
  Customizable::template output_interior_knot_indices_t<0> var_knot_indices_;

  /// @brief Knot indices of variables at the boundary
  Customizable::template output_boundary_knot_indices_t<0> var_knot_indices_boundary_;

public:
  /// @brief Constructor
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  EZSolverBase(const std::array<int64_t, GeometryMapNumCoeffs> &geometryMapNumCoeffs,
               const std::array<int64_t, VariableNumCoeffs> &variableNumCoeffs)
    : Base(std::make_tuple(geometryMapNumCoeffs),
           std::make_tuple(variableNumCoeffs)) {}
  
  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }
  
  
  /// @brief Returns a constant reference to the geometry
  auto const &G() const { return Base::template input<0>(); }
  
  /// @brief Returns a non-constant reference to the geometry
  auto &G() { return Base::template input<0>(); }
  
  /// @brief Returns a constant reference to the variable
  auto const &u() const { return Base::template output<0>(); }
  
  /// @brief Returns a non-constant reference to the variable
  auto &u() { return Base::template output<0>(); }
  
  /// @brief Initializes the solver
  void init() override {
    collPts_ =
      Base::template collPts<0>(iganet::collPts::greville);
    G_knot_indices_ =
      G().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
    G_knot_indices_boundary_ =
      G().template find_knot_indices<iganet::functionspace::boundary>(collPts_.second);    
    var_knot_indices_ =
      u().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);
    var_knot_indices_boundary_ =
      u().template find_knot_indices<iganet::functionspace::boundary>(collPts_.second);    
  }
};

/// @brief Easy-to-use solver class
///
/// This class implements an easy-to-use solver that takes a
/// geometry map and a variable function space as inputs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>  
class EZSolver : public EZSolverBase<GeometryMap, Variable> {
private:
  /// @brief Base class
  using Base = EZSolverBase<GeometryMap, Variable>;
  
  /// @brief Right-hand side function
  std::function<
    std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                              const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                              )
    > rhs_;

public:
  /// @brief Constructor
  EZSolver(const GeometryMap& geometryMap, const Variable& variable,
           const std::function<
           std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                                     const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                                     )
           >& rhs)
    : EZSolverBase<GeometryMap, Variable>(geometryMap.template space<0>().ncoeffs(), variable.template space<0>().ncoeffs()),
      rhs_(rhs) {
    // Copy actual control points from the input geometry map.
    // The base class constructs G() with greville init (identity map);
    // we overwrite it here so the physical-domain Laplacian sees the
    // correct (potentially perturbed) geometry.
    this->G().from_tensor(geometryMap.as_tensor());
  }
  
  /// @brief Assembles the left-hand side as the physical-domain Laplacian
  ///
  /// Uses the geometry map G to pull the physical Laplacian back to the
  /// parametric domain:
  ///   Δ_x B ≈ g^{11} ∂²B/∂ξ₁²  +  2 g^{12} ∂²B/∂ξ₁∂ξ₂  +  g^{22} ∂²B/∂ξ₂²
  /// where g^{kl} = (J⁻¹ J⁻ᵀ)_{kl} and J = ∂G/∂ξ.
  /// (Leading-order term; Christoffel corrections omitted.)
  void assembleLhs() override {

    // ── Parametric second derivatives of basis functions ─────────────────────
    auto S_xx = this->u().template eval_basfunc<functionspace::interior,
                                                (deriv::dx^2)>(Base::collPts_.first,
                                                               Base::var_knot_indices_);
    auto S_xy = this->u().template eval_basfunc<functionspace::interior,
                                                deriv::dx + deriv::dy>(Base::collPts_.first,
                                                               Base::var_knot_indices_);
    auto S_yy = this->u().template eval_basfunc<functionspace::interior,
                                                (deriv::dy^2)>(Base::collPts_.first,
                                                               Base::var_knot_indices_);

    // ── Mass matrix (boundary conditions: u = f on ∂Ω) ───────────────────────
    auto M    = this->u().eval_basfunc(Base::collPts_.first, Base::var_knot_indices_);
    auto mask = (Base::collPts_.first[0] == 0.0) | (Base::collPts_.first[0] == 1.0)
              | (Base::collPts_.first[1] == 0.0) | (Base::collPts_.first[1] == 1.0);

    // ── Jacobian of G at the collocation points ───────────────────────────────
    // G().space<0>() : UniformBSpline<double, 2, 2, 2>  (geoDim = 2)
    // eval<deriv::dx> → BlockTensor<Tensor,1,2>  = {∂Gx/∂ξ₁, ∂Gy/∂ξ₁}
    auto dG1 = this->G().template space<0>().template eval<deriv::dx>(Base::collPts_.first);
    auto dG2 = this->G().template space<0>().template eval<deriv::dy>(Base::collPts_.first);

    // J = [[a, b],   a = ∂Gx/∂ξ₁   b = ∂Gx/∂ξ₂
    //      [c, d]]   c = ∂Gy/∂ξ₁   d = ∂Gy/∂ξ₂
    const auto& a = *dG1[0];  // [ncolpts]
    const auto& c = *dG1[1];
    const auto& b = *dG2[0];
    const auto& d = *dG2[1];

    auto det_sq = (a*d - b*c).pow(2) + 1e-14;  // regularise to avoid ÷0

    // Contravariant metric:  g^{kl} = (J⁻¹ J⁻ᵀ)_{kl}
    auto g11 = (b*b + d*d) / det_sq;   // [ncolpts]
    auto g12 = -(a*b + c*d) / det_sq;
    auto g22 = (a*a + c*c) / det_sq;

    // Physical Laplacian applied to basis functions
    // S_xx shape: [local_support, ncolpts];  g11 shape: [ncolpts]
    // → broadcast g11.unsqueeze(0) to [1, ncolpts] then [local_support, ncolpts]
    auto phys_lhs = g11.unsqueeze(0) * S_xx
                  + 2.0 * g12.unsqueeze(0) * S_xy
                  + g22.unsqueeze(0) * S_yy;

    Base::lhs_ = iganet::utils::to_sparseCsrTensor(Base::var_knot_indices_,
                                                   this->u().template space<0>().degrees(),
                                                   this->u().template space<0>().ncoeffs(),
                                                   torch::where(mask, M, phys_lhs).t(),
                                                   { this->u().template space<0>().ncumcoeffs(),
                                                      this->u().template space<0>().ncumcoeffs() });
  }
  
  /// @brief Assembles the right-hand side from the given function
  void assembleRhs() override {
    Base::rhs_ = rhs_(Base::collPts().first)[0];     
  }
};
  
/// @brief Easy-to-use interpolation class
///
/// This class implements an easy-to-use interpolation that takes a
/// geometry map and a variable function space as inputs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>  
class EZInterpolation : public EZSolverBase<GeometryMap, Variable> {
private:
  /// @brief Base class
  using Base = EZSolverBase<GeometryMap, Variable>;
  
  /// @brief Right-hand side function
  std::function<
    std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                              const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                              )
    > rhs_;
  
public:
  /// @brief Constructor
  EZInterpolation(const GeometryMap& geometryMap, const Variable& variable,
                  const std::function<
                  std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                                            const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                                            )
                  >& rhs)
    : EZSolverBase<GeometryMap, Variable>(geometryMap.template space<0>().ncoeffs(), variable.template space<0>().ncoeffs()), rhs_(rhs) {}
  
  /// @brief Assembles the left-hand side as the mass matrix
  void assembleLhs() override {    
    Base::lhs_ = iganet::utils::to_sparseCsrTensor(Base::var_knot_indices_,
                                                   this->u().template space<0>().degrees(),
                                                   this->u().template space<0>().ncoeffs(),
                                                   this->u().eval_basfunc(Base::collPts_.first, Base::var_knot_indices_).t(),
                                                   { this->u().template space<0>().ncumcoeffs(),
                                                      this->u().template space<0>().ncumcoeffs() });
  }
  
  /// @brief Assembles the right-hand side from the given function
  void assembleRhs() override {
    Base::rhs_ = rhs_(Base::collPts().first)[0];     
  }
};
  
/// @brief Easy-to-use interpolation function
///
/// This function interpolates the given mapping functions in the
/// Greville points of the variable function space
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>
auto ezinterp(const GeometryMap& geometryMap,
              const Variable& variable,
              const std::function<std::array<torch::Tensor, Variable::template geoDim<0>()>(const std::array<torch::Tensor, Variable::template parDim<0>()> &)>
              mapping) {

  EZInterpolation interp(geometryMap, variable, mapping);
  interp.init();
  interp.assemble();
  return interp.solve().clone();
}

/// @brief Easy-to-use Poisson solver function
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>
auto ezpoisson(const GeometryMap& geometryMap,
               const Variable& variable,
               const std::function<std::array<torch::Tensor, Variable::template geoDim<0>()>(const std::array<torch::Tensor, Variable::template parDim<0>()> &)>
               rhs) {

  EZSolver solver(geometryMap, variable, rhs);
  solver.init();
  solver.assemble();
  return solver.solve().clone();
}  
  
} // namespace iganet
