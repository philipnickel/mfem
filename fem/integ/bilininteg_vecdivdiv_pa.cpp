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

#include "../bilininteg.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

namespace
{

void PAVectorDivDivSetup2D(const int q1d, const int ne,
                           const Array<real_t> &weights,
                           const Vector &jacobians,
                           const Vector &element_coefficient,
                           Vector &op)
{
   const int nq = q1d*q1d;
   const auto W = Reshape(weights.Read(), nq);
   const auto J = Reshape(jacobians.Read(), nq, 2, 2, ne);
   const auto C = Reshape(element_coefficient.Read(), ne);
   auto R = Reshape(op.Write(), nq, 2, 2, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         const real_t J11 = J(q,0,0,e);
         const real_t J12 = J(q,0,1,e);
         const real_t J21 = J(q,1,0,e);
         const real_t J22 = J(q,1,1,e);
         const real_t detJ = J11*J22 - J21*J12;
         // R = sqrt(w det(J) tau) J^{-1}
         //   = sqrt(w tau / det(J)) adj(J). The action applies R twice,
         // avoiding a stored rank-four quadrature tensor.
         const real_t scale = sqrt(W(q)*C(e)/detJ);
         R(q,0,0,e) =  scale*J22;
         R(q,0,1,e) = -scale*J12;
         R(q,1,0,e) = -scale*J21;
         R(q,1,1,e) =  scale*J11;
      }
   });
}

void PAVectorDivDivSetup3D(const int q1d, const int ne,
                           const Array<real_t> &weights,
                           const Vector &jacobians,
                           const Vector &element_coefficient,
                           Vector &op)
{
   const int nq = q1d*q1d*q1d;
   const auto W = Reshape(weights.Read(), nq);
   const auto J = Reshape(jacobians.Read(), nq, 3, 3, ne);
   const auto C = Reshape(element_coefficient.Read(), ne);
   auto R = Reshape(op.Write(), nq, 3, 3, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int q = 0; q < nq; ++q)
      {
         const real_t a = J(q,0,0,e), b = J(q,0,1,e), c = J(q,0,2,e);
         const real_t d = J(q,1,0,e), f = J(q,1,2,e);
         const real_t g = J(q,2,0,e), h = J(q,2,1,e), i = J(q,2,2,e);
         const real_t jj = J(q,1,1,e);
         const real_t A00 = jj*i - f*h;
         const real_t A01 = c*h - b*i;
         const real_t A02 = b*f - c*jj;
         const real_t A10 = f*g - d*i;
         const real_t A11 = a*i - c*g;
         const real_t A12 = c*d - a*f;
         const real_t A20 = d*h - jj*g;
         const real_t A21 = b*g - a*h;
         const real_t A22 = a*jj - b*d;
         const real_t detJ = a*A00 + b*A10 + c*A20;
         const real_t scale = sqrt(W(q)*C(e)/detJ);
         R(q,0,0,e) = scale*A00;
         R(q,0,1,e) = scale*A01;
         R(q,0,2,e) = scale*A02;
         R(q,1,0,e) = scale*A10;
         R(q,1,1,e) = scale*A11;
         R(q,1,2,e) = scale*A12;
         R(q,2,0,e) = scale*A20;
         R(q,2,1,e) = scale*A21;
         R(q,2,2,e) = scale*A22;
      }
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
void PAVectorDivDivApply2D(const int ne,
                           const Array<real_t> &b,
                           const Array<real_t> &g,
                           const Array<real_t> &bt,
                           const Array<real_t> &gt,
                           const Vector &op_,
                           const Vector &x_, Vector &y_,
                           const int d1d = 0, const int q1d = 0)
{
   if (ne == 0) { return; }

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Bt = Reshape(bt.Read(), D1D, Q1D);
   const auto Gt = Reshape(gt.Read(), D1D, Q1D);
   const auto R = Reshape(op_.Read(), Q1D*Q1D, 2, 2, ne);
   const auto X = Reshape(x_.Read(), D1D, D1D, 2, ne);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, 2, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int max_D1D = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int max_Q1D = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;

      real_t grad[max_Q1D][max_Q1D][2];
      real_t div[max_Q1D][max_Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx) { div[qy][qx] = 0.0; }
      }

      // Evaluate the two reference gradients and contract them with R.
      for (int c = 0; c < 2; ++c)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               grad[qy][qx][0] = 0.0;
               grad[qy][qx][1] = 0.0;
            }
         }
         for (int dy = 0; dy < D1D; ++dy)
         {
            real_t gradX[max_Q1D][2];
            for (int qx = 0; qx < Q1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < D1D; ++dx)
            {
               const real_t value = X(dx,dy,c,e);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  gradX[qx][0] += value*G(qx,dx);
                  gradX[qx][1] += value*B(qx,dx);
               }
            }
            for (int qy = 0; qy < Q1D; ++qy)
            {
               const real_t by = B(qy,dy);
               const real_t gy = G(qy,dy);
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  grad[qy][qx][0] += gradX[qx][0]*by;
                  grad[qy][qx][1] += gradX[qx][1]*gy;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + qy*Q1D;
               div[qy][qx] += grad[qy][qx][0]*R(q,0,c,e)
                              + grad[qy][qx][1]*R(q,1,c,e);
            }
         }
      }

      // Apply the transpose contraction directly, without a global Q-vector.
      for (int c = 0; c < 2; ++c)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + qy*Q1D;
               grad[qy][qx][0] = div[qy][qx]*R(q,0,c,e);
               grad[qy][qx][1] = div[qy][qx]*R(q,1,c,e);
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            real_t gradX[max_D1D][2];
            for (int dx = 0; dx < D1D; ++dx)
            {
               gradX[dx][0] = 0.0;
               gradX[dx][1] = 0.0;
            }
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int dx = 0; dx < D1D; ++dx)
               {
                  gradX[dx][0] += grad[qy][qx][0]*Gt(dx,qx);
                  gradX[dx][1] += grad[qy][qx][1]*Bt(dx,qx);
               }
            }
            for (int dy = 0; dy < D1D; ++dy)
            {
               const real_t by = Bt(dy,qy);
               const real_t gy = Gt(dy,qy);
               for (int dx = 0; dx < D1D; ++dx)
               {
                  Y(dx,dy,c,e) += gradX[dx][0]*by + gradX[dx][1]*gy;
               }
            }
         }
      }
   });
}

void PAVectorDivDivApply2DDispatch(const int ne,
                                   const Array<real_t> &b,
                                   const Array<real_t> &g,
                                   const Array<real_t> &bt,
                                   const Array<real_t> &gt,
                                   const Vector &op,
                                   const Vector &x, Vector &y,
                                   const int d1d, const int q1d)
{
   // Compile-time sizes keep the per-element temporaries exact-sized and let
   // the compiler fully unroll the small tensor contractions used by the
   // common collocated DG spaces. Retain the runtime kernel for overintegrated
   // and uncommon shapes.
   if (d1d == q1d)
   {
      switch (d1d)
      {
         case 2:
            return PAVectorDivDivApply2D<2,2>(ne,b,g,bt,gt,op,x,y);
         case 3:
            return PAVectorDivDivApply2D<3,3>(ne,b,g,bt,gt,op,x,y);
         case 4:
            return PAVectorDivDivApply2D<4,4>(ne,b,g,bt,gt,op,x,y);
         case 5:
            return PAVectorDivDivApply2D<5,5>(ne,b,g,bt,gt,op,x,y);
         case 6:
            return PAVectorDivDivApply2D<6,6>(ne,b,g,bt,gt,op,x,y);
         case 7:
            return PAVectorDivDivApply2D<7,7>(ne,b,g,bt,gt,op,x,y);
         case 8:
            return PAVectorDivDivApply2D<8,8>(ne,b,g,bt,gt,op,x,y);
         case 9:
            return PAVectorDivDivApply2D<9,9>(ne,b,g,bt,gt,op,x,y);
      }
   }
   PAVectorDivDivApply2D(ne,b,g,bt,gt,op,x,y,d1d,q1d);
}

void PAVectorDivDivApply3D(const int ne,
                           const Array<real_t> &b,
                           const Array<real_t> &g,
                           const Array<real_t> &bt,
                           const Array<real_t> &gt,
                           const Vector &op_,
                           const Vector &x_, Vector &y_,
                           Vector &quadrature_divergence,
                           const int d1d, const int q1d)
{
   if (ne == 0) { return; }
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(q1d <= DeviceDofQuadLimits::Get().MAX_Q1D, "");

   const auto B = Reshape(b.Read(), q1d, d1d);
   const auto G = Reshape(g.Read(), q1d, d1d);
   const auto Bt = Reshape(bt.Read(), d1d, q1d);
   const auto Gt = Reshape(gt.Read(), d1d, q1d);
   const int nq = q1d*q1d*q1d;
   const auto R = Reshape(op_.Read(), nq, 3, 3, ne);
   const auto X = Reshape(x_.Read(), d1d, d1d, d1d, 3, ne);
   auto QD = Reshape(quadrature_divergence.Write(), q1d, q1d, q1d, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int max_Q1D = DofQuadLimits::MAX_Q1D;
      real_t grad_xy[max_Q1D][max_Q1D][3];
      for (int qz = 0; qz < q1d; ++qz)
      {
         for (int qy = 0; qy < q1d; ++qy)
         {
            for (int qx = 0; qx < q1d; ++qx) { QD(qx,qy,qz,e) = 0.0; }
         }
      }
      for (int component = 0; component < 3; ++component)
      {
         for (int dz = 0; dz < d1d; ++dz)
         {
            for (int qy = 0; qy < q1d; ++qy)
            {
               for (int qx = 0; qx < q1d; ++qx)
               {
                  grad_xy[qy][qx][0] = 0.0;
                  grad_xy[qy][qx][1] = 0.0;
                  grad_xy[qy][qx][2] = 0.0;
               }
            }
            for (int dy = 0; dy < d1d; ++dy)
            {
               real_t grad_x[max_Q1D][2];
               for (int qx = 0; qx < q1d; ++qx)
               {
                  grad_x[qx][0] = 0.0;
                  grad_x[qx][1] = 0.0;
               }
               for (int dx = 0; dx < d1d; ++dx)
               {
                  const real_t value = X(dx,dy,dz,component,e);
                  for (int qx = 0; qx < q1d; ++qx)
                  {
                     grad_x[qx][0] += value*G(qx,dx);
                     grad_x[qx][1] += value*B(qx,dx);
                  }
               }
               for (int qy = 0; qy < q1d; ++qy)
               {
                  const real_t by = B(qy,dy), gy = G(qy,dy);
                  for (int qx = 0; qx < q1d; ++qx)
                  {
                     grad_xy[qy][qx][0] += grad_x[qx][0]*by;
                     grad_xy[qy][qx][1] += grad_x[qx][1]*gy;
                     grad_xy[qy][qx][2] += grad_x[qx][1]*by;
                  }
               }
            }
            for (int qz = 0; qz < q1d; ++qz)
            {
               const real_t bz = B(qz,dz), gz = G(qz,dz);
               for (int qy = 0; qy < q1d; ++qy)
               {
                  for (int qx = 0; qx < q1d; ++qx)
                  {
                     const int q = qx + q1d*(qy + q1d*qz);
                     QD(qx,qy,qz,e) +=
                        grad_xy[qy][qx][0]*bz*R(q,0,component,e) +
                        grad_xy[qy][qx][1]*bz*R(q,1,component,e) +
                        grad_xy[qy][qx][2]*gz*R(q,2,component,e);
                  }
               }
            }
         }
      }
   });

   const auto Div = Reshape(quadrature_divergence.Read(), q1d, q1d, q1d, ne);
   auto Y = Reshape(y_.ReadWrite(), d1d, d1d, d1d, 3, ne);
   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int max_D1D = DofQuadLimits::MAX_D1D;
      real_t grad_xy[max_D1D][max_D1D][3];
      for (int component = 0; component < 3; ++component)
      {
         for (int qz = 0; qz < q1d; ++qz)
         {
            for (int dy = 0; dy < d1d; ++dy)
            {
               for (int dx = 0; dx < d1d; ++dx)
               {
                  grad_xy[dy][dx][0] = 0.0;
                  grad_xy[dy][dx][1] = 0.0;
                  grad_xy[dy][dx][2] = 0.0;
               }
            }
            for (int qy = 0; qy < q1d; ++qy)
            {
               real_t grad_x[max_D1D][3];
               for (int dx = 0; dx < d1d; ++dx)
               {
                  grad_x[dx][0] = 0.0;
                  grad_x[dx][1] = 0.0;
                  grad_x[dx][2] = 0.0;
               }
               for (int qx = 0; qx < q1d; ++qx)
               {
                  const int q = qx + q1d*(qy + q1d*qz);
                  const real_t value = Div(qx,qy,qz,e);
                  for (int dx = 0; dx < d1d; ++dx)
                  {
                     grad_x[dx][0] += value*R(q,0,component,e)*Gt(dx,qx);
                     grad_x[dx][1] += value*R(q,1,component,e)*Bt(dx,qx);
                     grad_x[dx][2] += value*R(q,2,component,e)*Bt(dx,qx);
                  }
               }
               for (int dy = 0; dy < d1d; ++dy)
               {
                  const real_t by = Bt(dy,qy), gy = Gt(dy,qy);
                  for (int dx = 0; dx < d1d; ++dx)
                  {
                     grad_xy[dy][dx][0] += grad_x[dx][0]*by;
                     grad_xy[dy][dx][1] += grad_x[dx][1]*gy;
                     grad_xy[dy][dx][2] += grad_x[dx][2]*by;
                  }
               }
            }
            for (int dz = 0; dz < d1d; ++dz)
            {
               const real_t bz = Bt(dz,qz), gz = Gt(dz,qz);
               for (int dy = 0; dy < d1d; ++dy)
               {
                  for (int dx = 0; dx < d1d; ++dx)
                  {
                     Y(dx,dy,dz,component,e) +=
                        grad_xy[dy][dx][0]*bz +
                        grad_xy[dy][dx][1]*bz +
                        grad_xy[dy][dx][2]*gz;
                  }
               }
            }
         }
      }
   });
}

} // namespace

VectorDivDivIntegrator::VectorDivDivIntegrator(
   const Vector &element_coefficient_)
{
   SetElementCoefficient(element_coefficient_);
}

void VectorDivDivIntegrator::SetElementCoefficient(
   const Vector &element_coefficient_)
{
   element_coefficient.SetSize(element_coefficient_.Size(),
                               Device::GetMemoryType());
   element_coefficient.UseDevice(true);
   element_coefficient = element_coefficient_;
}

void VectorDivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY((dim == 2 || dim == 3) && mesh->SpaceDimension() == dim,
               "VectorDivDivIntegrator supports full-dimensional 2D/3D meshes");
   MFEM_VERIFY(fes.GetVDim() == dim,
               "VectorDivDivIntegrator requires vdim == mesh dimension");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "VectorDivDivIntegrator requires Ordering::byNODES");
   MFEM_VERIFY(fes.IsDGSpace(),
               "VectorDivDivIntegrator requires a discontinuous space");
   MFEM_VERIFY(!fes.IsVariableOrder(),
               "VectorDivDivIntegrator does not support variable order");
   MFEM_VERIFY(dynamic_cast<const L2_FECollection*>(fes.FEColl()) != nullptr,
               "VectorDivDivIntegrator requires an L2 finite-element collection");

   const FiniteElement &el = *fes.GetTypicalFE();
   MFEM_VERIFY(el.GetMapType() == FiniteElement::VALUE,
               "VectorDivDivIntegrator requires VALUE-mapped L2 elements");
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   MFEM_VERIFY(dynamic_cast<const NodalTensorFiniteElement*>(&el) != nullptr &&
               el.GetGeomType() == geometry,
               "VectorDivDivIntegrator requires tensor-product elements");

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      ElementTransformation *T = mesh->GetTypicalElementTransformation();
      MFEM_VERIFY(T != nullptr, "Unable to determine a default integration rule");
      ir = &MassIntegrator::GetRule(el, el, *T);
   }

   ne = fes.GetNE();
   MFEM_VERIFY(element_coefficient.Size() == ne,
               "VectorDivDivIntegrator needs one coefficient per local element");
   if (ne > 0)
   {
      MFEM_VERIFY(element_coefficient.Min() >= 0.0,
                  "VectorDivDivIntegrator requires nonnegative coefficients");
   }

   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   const int nq = ir->GetNPoints();
   const int expected_dofs = dim == 2 ? dofs1D*dofs1D : dofs1D*dofs1D*dofs1D;
   const int expected_quad = dim == 2 ? quad1D*quad1D : quad1D*quad1D*quad1D;
   MFEM_VERIFY(el.GetDof() == expected_dofs && nq == expected_quad,
               "VectorDivDivIntegrator requires tensor-product rules and elements");

   pa_data.SetSize(dim*dim*nq*ne, Device::GetMemoryType());
   quadrature_divergence.SetSize(nq*ne, Device::GetMemoryType());
   quadrature_divergence.UseDevice(true);
   if (ne == 0) { return; }
   const GeometricFactors *geom = mesh->GetGeometricFactors(
                                     *ir, GeometricFactors::JACOBIANS);
   if (dim == 2)
   {
      PAVectorDivDivSetup2D(quad1D, ne, ir->GetWeights(), geom->J,
                            element_coefficient, pa_data);
   }
   else
   {
      PAVectorDivDivSetup3D(quad1D, ne, ir->GetWeights(), geom->J,
                            element_coefficient, pa_data);
   }
}

void VectorDivDivIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (dim == 2)
   {
      PAVectorDivDivApply2DDispatch(ne, maps->B, maps->G, maps->Bt, maps->Gt,
                                    pa_data, x, y, dofs1D, quad1D);
   }
   else
   {
      PAVectorDivDivApply3D(ne, maps->B, maps->G, maps->Bt, maps->Gt,
                            pa_data, x, y, quadrature_divergence,
                            dofs1D, quad1D);
   }
}

void VectorDivDivIntegrator::AddMultTransposePA(const Vector &x,
                                                Vector &y) const
{
   AddMultPA(x, y);
}

} // namespace mfem
