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

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "dgmassinv_kernels.hpp"
#include "../general/forall.hpp"

#include <cmath>
#include <limits>

namespace mfem
{

namespace
{

bool IsIdentityDofToQuad(const DofToQuad *maps)
{
   if (!maps || maps->mode != DofToQuad::TENSOR ||
       maps->ndof != maps->nqpt)
   {
      return false;
   }

   const int n = maps->ndof;
   if (maps->B.Size() != n*n || maps->Bt.Size() != n*n)
   {
      return false;
   }

   const real_t *B = maps->B.HostRead();
   const real_t *Bt = maps->Bt.HostRead();
   const real_t tol = 64.0*std::numeric_limits<real_t>::epsilon();
   for (int d = 0; d < n; ++d)
   {
      for (int q = 0; q < n; ++q)
      {
         const real_t expected = (q == d) ? 1.0 : 0.0;
         // Using <= also rejects NaN. Check both stored maps instead of
         // inferring collocation from a named basis or quadrature family.
         if (!(std::abs(B[q + n*d] - expected) <= tol) ||
             !(std::abs(Bt[d + n*q] - expected) <= tol))
         {
            return false;
         }
      }
   }
   return true;
}

} // namespace

struct DGMassInvKernels { DGMassInvKernels(); };

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_orig,
                             Coefficient *coeff,
                             const IntegrationRule *ir,
                             int btype)
   : Solver(fes_orig.GetTrueVSize()),
     fec(fes_orig.GetMaxElementOrder(),
         fes_orig.GetMesh()->Dimension(),
         btype,
         fes_orig.GetTypicalFE()->GetMapType()),
     fes(fes_orig.GetMesh(), &fec, 1, fes_orig.GetOrdering())
{
   static DGMassInvKernels kernels;

   MFEM_VERIFY(fes_orig.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes_orig.IsVariableOrder(), "Variable orders not supported.");
   const auto *l2_fec =
      dynamic_cast<const L2_FECollection*>(fes_orig.FEColl());
   MFEM_VERIFY(l2_fec, "Space must use an L2 finite-element collection.");
   const int vdim = fes_orig.GetVDim();
   MFEM_VERIFY(vdim > 0, "Space must have positive vector dimension.");
   MFEM_VERIFY(fes_orig.GetVSize() == vdim*fes.GetVSize() &&
               fes_orig.GetTrueVSize() == vdim*fes.GetTrueVSize(),
               "Original DG space must consist of copies of the internal "
               "scalar DG space.");

   const int btype_orig = l2_fec->GetBasisType();

   if (btype_orig == btype)
   {
      // No change of basis required
      d2q = nullptr;
   }
   else
   {
      // original basis to solver basis
      const auto mode = DofToQuad::TENSOR;
      const FiniteElement &fe_orig = *fes_orig.GetTypicalFE();
      const FiniteElement &fe = *fes.GetTypicalFE();
      d2q = &fe_orig.GetDofToQuad(fe.GetNodes(), mode);

      const int n = d2q->ndof;
      Array<real_t> B_inv = d2q->B; // deep copy
      Array<int> ipiv(n);
      // solver basis to original
      LUFactors lu(B_inv.HostReadWrite(), ipiv.HostWrite());
      lu.Factor(n);
      B_.SetSize(n*n);
      lu.GetInverseMatrix(n, B_.HostWrite());
      Bt_.SetSize(n*n);
      DenseMatrix B_matrix(B_.HostReadWrite(), n, n);
      DenseMatrix Bt_matrix(Bt_.HostWrite(), n, n);
      Bt_matrix.Transpose(B_matrix);
   }

   if (coeff) { m = new MassIntegrator(*coeff, ir); }
   else { m = new MassIntegrator(ir); }

   // The mass operator and its diagonal are scalar and shared by every
   // component. The Krylov work vectors retain the input space's full size.
   diag_inv.SetSize(fes.GetTrueVSize());
   // Workspace vectors used for CG
   r_.SetSize(height);
   // byVDIM is not component-contiguous. Keep its solution in byNODES layout
   // during the batched solve, then scatter it back on completion.
   if (vdim > 1 && fes_orig.GetOrdering() == Ordering::byVDIM)
   {
      b2_.SetSize(height);
   }

   M.reset(new BilinearForm(&fes));
   M->AddDomainIntegrator(m); // M assumes ownership of m
   M->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // Assemble the bilinear form and its diagonal (for preconditioning).
   Update();
}

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                             int btype)
   : DGMassInverse(fes_, &coeff, nullptr, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, Coefficient &coeff,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, &coeff, &ir, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, nullptr, &ir, btype) { }

DGMassInverse::DGMassInverse(const FiniteElementSpace &fes_, int btype)
   : DGMassInverse(fes_, nullptr, nullptr, btype) { }

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const real_t rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const real_t abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const int max_iter_) { max_iter = max_iter_; }

void DGMassInverse::Update()
{
   M->Assemble();
   diagonal_mass = IsIdentityDofToQuad(m->maps);
   // The identity-map path is an exact diagonal solve and never enters CG.
   // Keep only the transformed-RHS workspace in that case.
   d_.SetSize(diagonal_mass ? 0 : height);
   z_.SetSize(diagonal_mass ? 0 : height);
   M->AssembleDiagonal(diag_inv);
   diag_inv.Reciprocal();
}

DGMassInverse::~DGMassInverse() = default;

template<int DIM, int D1D, int Q1D>
void DGMassInverse::DGMassCGIteration(const Vector &b_, Vector &u_) const
{
   using namespace internal; // host/device kernel functions

   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;
   const int ND = static_cast<int>(pow(d1d, DIM));
   const int scalar_size = ND*NE;
   const int VDIM = height/scalar_size;
   const bool BY_VDIM = VDIM > 1 && fes.GetOrdering() == Ordering::byVDIM;
   const bool DIAGONAL_MASS = diagonal_mass;

   const real_t *B = DIAGONAL_MASS ? nullptr : m->maps->B.Read();
   const real_t *Bt = DIAGONAL_MASS ? nullptr : m->maps->Bt.Read();
   const real_t *pa_data = DIAGONAL_MASS ? nullptr : m->pa_data.Read();
   const auto dinv = diag_inv.Read();
   const auto b_orig = b_.Read();
   auto r = r_.Write();
   real_t *d = DIAGONAL_MASS ? nullptr : d_.Write();
   real_t *z = DIAGONAL_MASS ? nullptr : z_.Write();
   auto u_orig = u_.ReadWrite();
   auto u = BY_VDIM ? b2_.Write() : u_orig;

   const real_t RELTOL = rel_tol;
   const real_t ABSTOL = abs_tol;
   const int MAXIT = max_iter;
   const bool IT_MODE = iterative_mode;
   const bool CHANGE_BASIS = (d2q != nullptr);

   const real_t *d2q_B = nullptr; // matrix to transform initial guess
   const real_t *q2d_B = nullptr; // matrix to transform solution
   const real_t *q2d_Bt = nullptr; // matrix to transform RHS
   if (CHANGE_BASIS)
   {
      d2q_B = d2q->B.Read();
      q2d_B = B_.Read();
      q2d_Bt = Bt_.Read();
   }

   static constexpr int NB = Q1D ? Q1D : 1; // block size

   // Keep the components of each element adjacent in the launch order so the
   // second component reuses its basis and PA data from cache on CPUs. Each
   // component retains an independent element-local CG recurrence.
   mfem::forall_2D(NE*VDIM, NB, NB, [=] MFEM_HOST_DEVICE (int ec)
   {
      const int c = ec % VDIM;
      const int e = ec / VDIM;
      const int offset = c*scalar_size;
      const real_t *rc = r + offset;
      const real_t *dc = DIAGONAL_MASS ? nullptr : d + offset;
      const real_t *zc = DIAGONAL_MASS ? nullptr : z + offset;
      real_t *rw = r + offset;
      real_t *dw = DIAGONAL_MASS ? nullptr : d + offset;
      real_t *zw = DIAGONAL_MASS ? nullptr : z + offset;
      real_t *uw = u + offset;

      const int tid = MFEM_THREAD_ID(x) + NB*MFEM_THREAD_ID(y);
      const int bxy = MFEM_THREAD_SIZE(x)*MFEM_THREAD_SIZE(y);
      auto R = DeviceMatrix(rw, ND, NE);
      auto U = DeviceMatrix(uw, ND, NE);

      // Gather the RHS into the component-contiguous recurrence layout. For a
      // byVDIM input, gather the initial guess as well; its solution remains in
      // b2_ until the final scatter, so arbitrary component layouts are safe.
      for (int i = tid; i < ND; i += bxy)
      {
         const int sdof = i + ND*e;
         const int vdof = BY_VDIM ? c + VDIM*sdof : sdof + offset;
         R(i,e) = b_orig[vdof];
         if (!DIAGONAL_MASS)
         {
            if (!IT_MODE) { U(i,e) = 0.0; }
            else if (BY_VDIM) { U(i,e) = u_orig[vdof]; }
         }
      }
      MFEM_SYNC_THREAD;

      // Perform change of basis if needed
      if (CHANGE_BASIS)
      {
         // Transform RHS
         DGMassBasis<DIM,D1D>(e, NE, q2d_Bt, rw, rw, d1d);
         if (IT_MODE && !DIAGONAL_MASS)
         {
            // Transform initial guess
            DGMassBasis<DIM,D1D>(e, NE, d2q_B, uw, uw, d1d);
         }
      }

      // A square identity dof-to-quadrature map makes B^T D B exactly
      // diagonal. In that structural case the assembled diagonal is the
      // inverse itself and no element-local Krylov recurrence is needed.
      if (DIAGONAL_MASS)
      {
         DGMassPreconditioner(e, NE, ND, dinv, rc, uw);
      }
      else
      {
         // Compute first residual
         if (IT_MODE)
         {
            DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data,
                                     uw, zw, d1d, q1d);
            DGMassAxpy(e, NE, ND, 1.0, rc, -1.0, zc, rw); // r = b - A*u
         }

         DGMassPreconditioner(e, NE, ND, dinv, rc, zw);
         DGMassAxpy(e, NE, ND, 1.0, zc, 0.0, zc, dw); // d = z

         real_t nom = DGMassDot<NB>(e, NE, ND, dc, rc);
         real_t r0 = fmax(nom*RELTOL*RELTOL, ABSTOL*ABSTOL);
         if (nom >= 0.0 && nom > r0)
         {
            DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data,
                                     dc, zw, d1d, q1d);
            real_t den = DGMassDot<NB>(e, NE, ND, zc, dc);
            if (den <= 0.0)
            {
               DGMassDot<NB>(e, NE, ND, dc, dc);
               // d2 > 0 => not positive definite
            }
            if (den != 0.0)
            {
               // start iteration
               int i = 1;
               while (true)
               {
                  const real_t alpha = nom/den;
                  DGMassAxpy(e, NE, ND, 1.0, uw, alpha, dc, uw);
                  DGMassAxpy(e, NE, ND, 1.0, rc, -alpha, zc, rw);

                  DGMassPreconditioner(e, NE, ND, dinv, rc, zw);

                  real_t betanom = DGMassDot<NB>(e, NE, ND, rc, zc);
                  if (betanom < 0.0 || betanom <= r0) { break; }
                  if (++i > MAXIT) { break; }

                  const real_t beta = betanom/nom;
                  DGMassAxpy(e, NE, ND, 1.0, zc, beta, dc, dw);
                  DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data,
                                           dc, zw, d1d, q1d);
                  den = DGMassDot<NB>(e, NE, ND, dc, zc);
                  if (den <= 0.0)
                  {
                     DGMassDot<NB>(e, NE, ND, dc, dc);
                     // d2 > 0 => not positive definite
                     if (den == 0.0) { break; }
                  }
                  nom = betanom;
               }
            }
         }
      }

      if (CHANGE_BASIS)
      {
         DGMassBasis<DIM,D1D>(e, NE, q2d_B, uw, uw, d1d);
      }

      if (BY_VDIM)
      {
         for (int i = tid; i < ND; i += bxy)
         {
            const int sdof = i + ND*e;
            u_orig[c + VDIM*sdof] = U(i,e);
         }
         MFEM_SYNC_THREAD;
      }
   });
}

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   MFEM_VERIFY(Mu.Size() == width && u.Size() == height,
               "Input and output vectors have incompatible sizes.");
   if (height == 0) { return; }

   // Dispatch to templated version based on dim, d1d, and q1d.
   const int dim = fes.GetMesh()->Dimension();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   CGKernels::Run(dim, d1d, q1d, *this, Mu, u);
}

DGMassInvKernels::DGMassInvKernels()
{
   using k = DGMassInverse::CGKernels;
   // 2D
   k::Specialization<2,1,1>::Add();
   k::Specialization<2,2,2>::Add();
   k::Specialization<2,3,3>::Add();
   k::Specialization<2,3,5>::Add();
   k::Specialization<2,4,4>::Add();
   k::Specialization<2,4,6>::Add();
   k::Specialization<2,5,5>::Add();
   k::Specialization<2,5,7>::Add();
   k::Specialization<2,6,6>::Add();
   k::Specialization<2,6,8>::Add();
   // 3D
   k::Specialization<3,2,2>::Add();
   k::Specialization<3,2,3>::Add();
   k::Specialization<3,3,3>::Add();
   k::Specialization<3,3,4>::Add();
   k::Specialization<3,3,5>::Add();
   k::Specialization<3,4,4>::Add();
   k::Specialization<3,4,5>::Add();
   k::Specialization<3,4,6>::Add();
   k::Specialization<3,4,8>::Add();
   k::Specialization<3,5,5>::Add();
   k::Specialization<3,5,6>::Add();
   k::Specialization<3,5,7>::Add();
   k::Specialization<3,5,8>::Add();
   k::Specialization<3,6,6>::Add();
   k::Specialization<3,6,7>::Add();
}

/// @cond Suppress_Doxygen_warnings

template <int DIM, int D1D, int Q1D>
DGMassInverse::CGKernelType DGMassInverse::CGKernels::Kernel()
{
   return &DGMassInverse::DGMassCGIteration<DIM,D1D,Q1D>;
}

DGMassInverse::CGKernelType DGMassInverse::CGKernels::Fallback(
   int dim, int, int)
{
   if (dim == 1) { return &DGMassInverse::DGMassCGIteration<1>; }
   else if (dim == 2) { return &DGMassInverse::DGMassCGIteration<2>; }
   else if (dim == 3) { return &DGMassInverse::DGMassCGIteration<3>; }
   else { MFEM_ABORT("Unsupported dimension."); }
}

/// @endcond

} // namespace mfem
