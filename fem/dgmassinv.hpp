// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DGMASSINV_HPP
#define MFEM_DGMASSINV_HPP

#include "../linalg/operator.hpp"
#include "fespace.hpp"
#include "kernel_dispatch.hpp"
#include <memory>

namespace mfem
{

/// @brief Solver for the discontinuous Galerkin mass matrix.
///
/// This class performs a @a local (diagonally preconditioned) conjugate
/// gradient iteration for each element. Optionally, a change of basis is
/// performed to iterate on a better-conditioned system. This class fully
/// supports execution on device (GPU). For vector-valued copies of a scalar
/// DG space, each component is inverted independently in one batched action;
/// both Ordering::byNODES and Ordering::byVDIM are supported.
class DGMassInverse : public Solver
{
protected:
   DG_FECollection fec; ///< FE collection in requested basis.
   FiniteElementSpace fes; ///< FE space in requested basis.
   const DofToQuad *d2q; ///< Change of basis. Not owned.
   Array<real_t> B_; ///< Inverse of change of basis.
   Array<real_t> Bt_; ///< Inverse of change of basis, transposed.
   std::unique_ptr<class BilinearForm> M; ///< Mass bilinear form.
   class MassIntegrator *m; ///< Mass integrator, owned by the form @ref M.
   Vector diag_inv; ///< Jacobi preconditioner.
   real_t rel_tol = 1e-12; ///< Relative CG tolerance.
   real_t abs_tol = 1e-12; ///< Absolute CG tolerance.
   int max_iter = 100; ///< Maximum number of CG iterations;
   bool diagonal_mass = false; ///< The PA mass map is structurally diagonal.

   /// @name Intermediate vectors needed for CG three-term recurrence.
   ///@{
   mutable Vector r_, d_, z_, b2_;
   ///@}

   /// @brief Protected constructor, used internally.
   ///
   /// Custom coefficient and integration rule are used if @a coeff and @a ir
   /// are non-NULL.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient *coeff,
                 const IntegrationRule *ir, int btype);
public:
   /// @brief Construct the DG inverse mass operator for @a fes_.
   ///
   /// The basis type @a btype determines which basis should be used internally
   /// in the solver. This <b>does not</b> have to be the same basis as @a fes_.
   /// The best choice is typically BasisType::GaussLegendre because it is
   /// well-preconditioned by its diagonal.
   ///
   /// The solution and right-hand side used for the solver are not affected by
   /// this basis (they correspond to the basis of @a fes_). @a btype is only
   /// used internally, and only has an effect on the convergence rate. If
   /// @a fes_ has vector dimension greater than one, the scalar mass inverse is
   /// applied independently to every component while preserving its ordering.
   DGMassInverse(const FiniteElementSpace &fes_,
                 int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                 int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// Coefficient @a coeff and IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                 const IntegrationRule &ir, int btype=BasisType::GaussLegendre);
   /// @brief Construct the DG inverse mass operator for @a fes_ with
   /// IntegrationRule @a ir.
   ///
   /// @sa DGMassInverse(FiniteElementSpace&, int) for information about @a
   /// btype.
   DGMassInverse(const FiniteElementSpace &fes_, const IntegrationRule &ir,
                 int btype=BasisType::GaussLegendre);
   /// @brief Solve the system M b = u.
   ///
   /// If @ref iterative_mode is @a true, @a u is used as an initial guess.
   void Mult(const Vector &b, Vector &u) const override;
   /// Same as Mult() since the mass matrix is symmetric.
   void MultTranspose(const Vector &b, Vector &u) const override { Mult(b, u); }
   /// Not implemented. Aborts.
   void SetOperator(const Operator &op) override;
   /// Set the relative tolerance.
   void SetRelTol(const real_t rel_tol_);
   /// Set the absolute tolerance.
   void SetAbsTol(const real_t abs_tol_);
   /// Set the maximum number of iterations.
   void SetMaxIter(const int max_iter_);
   /// Recompute operator and preconditioner (when coefficient or mesh changes).
   void Update();

   ~DGMassInverse();

   /// @brief Solve the system M b = u. <b>Not part of the public interface.</b>
   /// @note This member function must be public because it defines an
   /// extended lambda used in an mfem::forall kernel (nvcc limitation)
   template<int DIM, int D1D = 0, int Q1D = 0>
   void DGMassCGIteration(const Vector &b_, Vector &u_) const;

   using CGKernelType = void(DGMassInverse::*)(const Vector &b_, Vector &u) const;
   MFEM_REGISTER_KERNELS(CGKernels, CGKernelType, (int, int, int));
};

/** @brief Adapt a fixed DGMassInverse for use as an iterative-solver
    preconditioner.

    IterativeSolver::SetOperator forwards the system operator to its
    preconditioner. DGMassInverse is tied to the finite-element space and mass
    operator supplied at construction, so its own SetOperator deliberately
    rejects that call. This non-owning adapter ignores the forwarded system
    operator and applies the existing inverse directly, without the additional
    full-vector clear performed by a one-block BlockDiagonalPreconditioner. */
class DGMassInversePreconditioner : public Solver
{
private:
   DGMassInverse &inverse;

public:
   explicit DGMassInversePreconditioner(DGMassInverse &inverse_)
      : Solver(inverse_.Height(), inverse_.Width()), inverse(inverse_) { }

   void Mult(const Vector &x, Vector &y) const override
   { inverse.Mult(x, y); }

   void MultTranspose(const Vector &x, Vector &y) const override
   { inverse.MultTranspose(x, y); }

   void SetOperator(const Operator &) override { }
};

/** @brief Element-local L2 load for the vorticity of an equal-order vector
    field.

    The input is a byNODES vector made from @a dim copies of one scalar broken
    space. In 2D the output is the scalar curl; in 3D it is the three-component
    curl. Applying DGMassInverse completes the local L2 projection. */
class LocalVorticityProjectionOperator : public Operator
{
private:
   FiniteElementSpace &fes;
   const IntegrationRule &ir;
   const ElementRestrictionOperator *element_restriction = nullptr; ///< Not owned.
   const QuadratureInterpolator *quadrature_interpolator = nullptr; ///< Not owned.
   const DofToQuad *maps = nullptr; ///< Not owned.
   int dim, curl_dim, scalar_size, ne, nd, nq, nd1d, nq1d;
   mutable Vector element_values, quadrature_derivatives;
   mutable Vector quadrature_curl, element_load;

public:
   LocalVorticityProjectionOperator(FiniteElementSpace &fes_,
                                    const IntegrationRule &ir_);
   void Mult(const Vector &x, Vector &y) const override;
};

/** @brief Boundary load for minus viscosity times normal curl of a projected
    vorticity field on selected physical boundary attributes. */
class CurlVorticityBoundaryIntegrator : public Operator
{
private:
   FiniteElementSpace &fes;
   Array<int> marker;
   const IntegrationRule &ir;
   const FaceRestriction *face_restriction = nullptr; ///< Not owned.
   const FaceQuadratureInterpolator *face_interpolator = nullptr; ///< Not owned.
   const DofToQuad *maps = nullptr; ///< Not owned.
   Array<int> active_boundary;
   int dim, curl_dim, scalar_size, nf, nq, face_dofs, nd1d, nq1d;
   mutable Vector face_values, face_derivatives, quadrature_load, face_load;
   real_t viscosity = 1.0;

public:
   CurlVorticityBoundaryIntegrator(FiniteElementSpace &fes_,
                                   const Array<int> &marker_,
                                   const IntegrationRule &ir_);
   void SetViscosity(real_t viscosity_);
   real_t GetViscosity() const { return viscosity; }
   void Mult(const Vector &x, Vector &y) const override;
};

/** @brief Per-element velocity scales and divergence diagnostic for an
    equal-order vector field.

    The input contains @a dim scalar L-vectors in byNODES ordering. Mult()
    preserves the original behavior and returns the magnitude of the vector
    mean. ComputeMeasures() additionally returns the mean velocity magnitude
    and the root-mean-square physical divergence. All measures use the same
    quadrature rule and refresh the matching physical element volumes. */
class ElementMeanMagnitudeOperator : public Operator
{
private:
   FiniteElementSpace &fes;
   const IntegrationRule &ir;
   const ElementRestrictionOperator *element_restriction = nullptr; ///< Not owned.
   const QuadratureInterpolator *quadrature_interpolator = nullptr; ///< Not owned.
   int dim, scalar_size, ne, nd, nq;
   mutable Vector element_values, quadrature_values, quadrature_derivatives;
   mutable Vector volumes, scratch_mean_magnitude, scratch_divergence_rms;

public:
   ElementMeanMagnitudeOperator(FiniteElementSpace &fes_,
                                const IntegrationRule &ir_);
   void Mult(const Vector &x, Vector &y) const override;
   /** Compute the magnitude of the vector mean, the mean vector magnitude,
       and the RMS physical divergence for every local element. */
   void ComputeMeasures(const Vector &x, Vector &vector_mean,
                        Vector &mean_magnitude,
                        Vector &divergence_rms) const;
   /** Compute @a scale times the most recently evaluated mean magnitude
       times the square root of the physical element volume. */
   void ComputeTau(const Vector &mean, real_t scale, Vector &tau) const;
   const Vector &GetElementVolumes() const { return volumes; }
};

/** @brief Element-local Fehn ALE convective CFL rate.

    The input is a broken vector field in byNODES ordering containing the ALE
    relative velocity @f$u-u_G@f$. At every supplied quadrature point this
    operator evaluates

    @f[ \left\|J^{-T}(u-u_G)\right\|_2, @f]

    and Mult() returns its maximum for every local element. ComputeMax()
    performs the same native action and returns the maximum over local
    elements. The caller owns the MPI maximum and the
    @f$\mathrm{Cr}/k^{1.5}@f$ scaling from Fehn et al.'s adaptive CFL
    condition. */
class FehnALECFLRateOperator : public Operator
{
private:
   FiniteElementSpace &fes;
   const IntegrationRule &ir;
   const ElementRestrictionOperator *element_restriction = nullptr; ///< Not owned.
   const QuadratureInterpolator *quadrature_interpolator = nullptr; ///< Not owned.
   int dim, ne, nd, nq;
   mutable Vector element_values, quadrature_values, element_rates;

public:
   FehnALECFLRateOperator(FiniteElementSpace &fes_,
                          const IntegrationRule &ir_);
   void Mult(const Vector &relative_velocity,
             Vector &element_rate) const override;
   real_t ComputeMax(const Vector &relative_velocity) const;
   const Vector &GetElementRates() const { return element_rates; }
};

/** @brief Native one-dimensional free-surface kinematic operator.

    The packed input contains elevation, horizontal velocity, and vertical
    velocity in three scalar byNODES blocks. The output is
    M^{-1}(M u_y - (u_x d_x eta,phi) + F_LF), with a local
    Lax--Friedrichs correction on every interior trace junction. */
class SurfaceKinematicOperator : public Operator
{
private:
   FiniteElementSpace &fes;
   const IntegrationRule &ir;
   const ElementRestrictionOperator *element_restriction = nullptr; ///< Not owned.
   const QuadratureInterpolator *quadrature_interpolator = nullptr; ///< Not owned.
   const FaceRestriction *face_restriction = nullptr; ///< Not owned.
   const DofToQuad *maps = nullptr; ///< Not owned.
   int scalar_size, ne, nd, nq, nd1d, nq1d;
   real_t upwind_factor;
   Vector inverse_mass, orientation, orientation_face;
   mutable Vector eta_face, velocity_face, face_flux, volume_flux, load;
   mutable Vector element_values, eta_derivatives, velocity_values;
   mutable Vector quadrature_load, element_load, element_rate;
   mutable Vector packed_input;

public:
   SurfaceKinematicOperator(FiniteElementSpace &fes_,
                            const IntegrationRule &ir_,
                            real_t upwind_factor_ = 1.0);
   void SetUpwindFactor(real_t value);
   real_t GetUpwindFactor() const { return upwind_factor; }
   void Mult(const Vector &x, Vector &y) const override;
   /// Assemble only the two-trace LLF junction correction (before M^{-1}).
   void FaceMult(const Vector &elevation, const Vector &velocity_x,
                 Vector &face_load) const;
   /// Apply from three native scalar blocks without constructing device-unsafe aliases.
   void Mult3(const Vector &elevation, const Vector &velocity_x,
              const Vector &velocity_y, Vector &rate) const;
};

/** @brief Native kinematic operator on a two-dimensional quadrilateral
    free-surface submesh embedded in three-dimensional space.

    The packed input contains elevation, the two horizontal velocity
    components, and vertical velocity in four scalar byNODES blocks. The
    output is

        M^{-1}(M u_z - (u_x d_x eta + u_y d_y eta, phi) + F_LF),

    where @a F_LF is the strong-form local Lax--Friedrichs correction on every
    interior or periodic surface edge. Physical boundary edges use the graph
    wall condition (zero normal transport), for which the correction is zero.
    The surface geometry is captured at construction and remains the flat
    reference manifold while the parent volume mesh moves. */
class SurfaceKinematicOperator2D : public Operator
{
private:
   FiniteElementSpace &fes;
   const IntegrationRule &ir;
   const IntegrationRule *face_ir = nullptr; ///< Not owned (global rule table).
   const ElementRestrictionOperator *element_restriction = nullptr; ///< Not owned.
   const FaceRestriction *face_restriction = nullptr; ///< Not owned.
   const QuadratureInterpolator *quadrature_interpolator = nullptr; ///< Not owned.
   const DofToQuad *maps = nullptr; ///< Not owned.
   const DofToQuad *face_maps = nullptr; ///< Not owned.
   int scalar_size, ne, nd, nq, nd1d, nq1d;
   int nf, face_nd, face_nq, sdim;
   real_t upwind_factor;
   Vector inverse_mass, volume_weights, gradient_map;
   Vector face_weights, face_normals;
   mutable Vector element_values, eta_derivatives, velocity_values;
   mutable Vector quadrature_load, element_load, element_rate;
   mutable Vector face_element_values, face_values;
   mutable Vector face_quadrature_flux, face_element_flux;
   mutable Vector volume_flux, load, packed_input;

public:
   SurfaceKinematicOperator2D(FiniteElementSpace &fes_,
                              const IntegrationRule &ir_,
                              real_t upwind_factor_ = 1.0);
   void SetUpwindFactor(real_t value);
   real_t GetUpwindFactor() const { return upwind_factor; }
   void Mult(const Vector &x, Vector &y) const override;
   /// Assemble only the two-trace LLF edge correction (before M^{-1}).
   void FaceMult(const Vector &elevation, const Vector &velocity_x,
                 const Vector &velocity_y, Vector &face_load) const;
   void Mult4(const Vector &elevation, const Vector &velocity_x,
              const Vector &velocity_y, const Vector &velocity_z,
              Vector &rate) const;
};

} // namespace mfem

#endif
