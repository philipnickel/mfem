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
#include "quadinterpolator.hpp"
#include "qspace.hpp"
#include "restriction.hpp"
#include "../general/forall.hpp"

#include <cmath>
#include <limits>

namespace mfem
{

SurfaceKinematicOperator::SurfaceKinematicOperator(
   FiniteElementSpace &fes_, const IntegrationRule &ir_,
   real_t upwind_factor_)
   : Operator(fes_.GetVSize(), 3 * fes_.GetVSize()), fes(fes_), ir(ir_),
     scalar_size(fes_.GetVSize()), ne(fes_.GetNE()),
     nd(fes_.GetTypicalFE()->GetDof()), nq(ir_.GetNPoints()),
     nd1d(0), nq1d(0), upwind_factor(upwind_factor_),
     orientation(scalar_size)
{
   MFEM_VERIFY(fes.GetMesh()->Dimension() == 1,
               "surface kinematics requires a one-dimensional trace mesh");
   MFEM_VERIFY(fes.GetVDim() == 1 && fes.GetVSize() == fes.GetTrueVSize(),
               "surface kinematics requires a scalar broken trace space");
   SetUpwindFactor(upwind_factor_);
   element_restriction = fes.GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   quadrature_interpolator = fes.GetQuadratureInterpolator(ir);
   maps = &fes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   nd1d = maps->ndof;
   nq1d = maps->nqpt;
   MFEM_VERIFY(element_restriction != nullptr &&
               element_restriction->Height() == ne * nd &&
               quadrature_interpolator != nullptr &&
               nd1d == nd && nq1d == nq,
               "surface kinematics requires tensor segment elements and rule");
   quadrature_interpolator->EnableTensorProducts();
   quadrature_interpolator->SetOutputLayout(QVectorLayout::byNODES);
   face_restriction = fes.GetFaceRestriction(
                         ElementDofOrdering::NATIVE, FaceType::Interior);
   MFEM_VERIFY(face_restriction != nullptr,
               "surface kinematics requires an interior face restriction");

   orientation = 0.0;
   Array<int> dofs;
   IntegrationPoint left, right;
   left.Set1w(0.0, 1.0);
   right.Set1w(1.0, 1.0);
   Vector x_left, x_right;
   for (int e = 0; e < fes.GetNE(); ++e)
   {
      fes.GetElementDofs(e, dofs);
      MFEM_VERIFY(dofs.Size() >= 2,
                  "surface element must have two endpoint dofs");
      ElementTransformation &T = *fes.GetElementTransformation(e);
      T.Transform(left, x_left);
      T.Transform(right, x_right);
      const bool native_is_physical = x_left[0] < x_right[0];
      const int left_dof = native_is_physical ? dofs[0] : dofs.Last();
      const int right_dof = native_is_physical ? dofs.Last() : dofs[0];
      MFEM_VERIFY(left_dof >= 0 && right_dof >= 0,
                  "oriented scalar dofs are not supported");
      orientation[left_dof] = -1.0;
      orientation[right_dof] = 1.0;
   }
   const int face_size = face_restriction->Height();
   MFEM_VERIFY(face_size % 2 == 0,
               "surface face restriction must be double valued");
   orientation_face.SetSize(face_size);
   eta_face.SetSize(face_size);
   velocity_face.SetSize(face_size);
   face_flux.SetSize(face_size);
   volume_flux.SetSize(scalar_size);
   load.SetSize(scalar_size);
   // The tail stores the signed, static one-dimensional Jacobians. The flat
   // reference surface never moves, so they share the persistent element
   // scratch allocation without another step-time vector.
   // One scalar input scratch precedes the three restricted component
   // blocks.  Copying a packed component with a device kernel avoids making
   // a host-valid alias of a device-newer input Vector (the Python packed
   // free-surface history path exercises exactly that case).
   element_values.SetSize(4 * ne * nd + ne * nq);
   eta_derivatives.SetSize(ne * nq);
   velocity_values.SetSize(2 * ne * nq);
   quadrature_load.SetSize(ne * nq);
   element_load.SetSize(ne * nd);
   element_rate.SetSize(ne * nd);
   packed_input.SetSize(3 * scalar_size);
   face_restriction->Mult(orientation, orientation_face);
   orientation_face.HostRead();
   for (int f = 0; f < face_size / 2; ++f)
   {
      const real_t a = orientation_face[2*f];
      const real_t b = orientation_face[2*f + 1];
      MFEM_VERIFY(a * b == -1.0,
                  "surface face restriction must pair left/right endpoints");
   }
   inverse_mass.SetSize(ne * nd * nd);
   Vector shape(nd);
   DenseMatrix mass(nd), inverse_matrix;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &T = *fes.GetElementTransformation(e);
      mass = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         fe.CalcShape(ip, shape);
         element_values[4*ne*nd + q + nq*e] = T.Jacobian()(0, 0);
         AddMult_a_VVt(ip.weight * std::abs(T.Weight()), shape, mass);
      }
      DenseMatrixInverse factor(mass, true);
      factor.GetInverseMatrix(inverse_matrix);
      for (int j = 0; j < nd; ++j)
      {
         for (int i = 0; i < nd; ++i)
         {
            inverse_mass[i + nd*(j + nd*e)] = inverse_matrix(i, j);
         }
      }
   }
   inverse_mass.UseDevice(true);
   orientation.UseDevice(true);
   orientation_face.UseDevice(true);
   eta_face.UseDevice(true);
   velocity_face.UseDevice(true);
   face_flux.UseDevice(true);
   volume_flux.UseDevice(true);
   load.UseDevice(true);
   element_values.UseDevice(true);
   eta_derivatives.UseDevice(true);
   velocity_values.UseDevice(true);
   quadrature_load.UseDevice(true);
   element_load.UseDevice(true);
   element_rate.UseDevice(true);
   packed_input.UseDevice(true);
}

void SurfaceKinematicOperator::SetUpwindFactor(real_t value)
{
   MFEM_VERIFY(value >= 0.0 && std::isfinite(value),
               "surface upwind factor must be finite and non-negative");
   upwind_factor = value;
}

void SurfaceKinematicOperator::FaceMult(const Vector &elevation,
                                        const Vector &velocity_x,
                                        Vector &face_load) const
{
   MFEM_VERIFY(elevation.Size() == scalar_size &&
               velocity_x.Size() == scalar_size,
               "surface kinematic face input has wrong size");
   face_load.SetSize(scalar_size);
   face_load.UseDevice(true);
   face_load = 0.0;
   if (face_restriction->Height() == 0) { return; }

   Vector component;
   component.MakeRef(element_values, 0, scalar_size);
   real_t *scalar = element_values.ReadWrite();
   const real_t *E = elevation.Read();
   mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
   {
      scalar[i] = E[i];
   });
   face_restriction->Mult(component, eta_face);
   const real_t *U = velocity_x.Read();
   mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
   {
      scalar[i] = U[i];
   });
   face_restriction->Mult(component, velocity_face);

   const int nfaces = face_restriction->Height() / 2;
   const real_t *O = orientation_face.Read();
   const real_t *EF = eta_face.Read();
   const real_t *UF = velocity_face.Read();
   real_t *F = face_flux.Write();
   const real_t alpha = upwind_factor;
   mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int f)
   {
      const bool first_is_left = O[2*f] > 0.0;
      const int left = 2*f + (first_is_left ? 0 : 1);
      const int right = 2*f + (first_is_left ? 1 : 0);
      const real_t eta_left = EF[left];
      const real_t eta_right = EF[right];
      const real_t u_left = UF[left];
      const real_t u_right = UF[right];
      const real_t speed = 0.5 * (u_left + u_right);
      const real_t radius = alpha * fmax(fabs(u_left), fabs(u_right));
      const real_t jump = eta_left - eta_right;
      F[left] = 0.5 * (speed - radius) * jump;
      F[right] = 0.5 * (speed + radius) * jump;
   });
   face_restriction->MultTranspose(face_flux, face_load);
}

void SurfaceKinematicOperator::Mult(const Vector &x, Vector &y) const
{
   const bool debug_device = Device::Allows(Backend::DEBUG_DEVICE);
   MFEM_VERIFY(x.Size() == Width(), "surface kinematic input has wrong size");
   const real_t *X = x.Read();
   // Preserve the immutable signed-Jacobian tail populated at construction.
   real_t *component_data = element_values.ReadWrite();
   for (int c = 0; c < 3; ++c)
   {
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         component_data[i] = X[i + c*scalar_size];
      });
      Vector component, e_component;
      component.MakeRef(element_values, 0, scalar_size);
      e_component.MakeRef(element_values, (1 + c) * ne * nd, ne * nd);
      element_restriction->Mult(component, e_component);
   }
   if (debug_device)
   {
      MFEM_VERIFY(element_values.CheckFinite() == 0,
                  "surface restricted values are not finite");
   }
   Vector eta_element, velocity_x_element, velocity_y_element;
   Vector velocity_x_q, velocity_y_q;
   eta_element.MakeRef(element_values, ne * nd, ne * nd);
   velocity_x_element.MakeRef(element_values, 2 * ne * nd, ne * nd);
   velocity_y_element.MakeRef(element_values, 3 * ne * nd, ne * nd);
   velocity_x_q.MakeRef(velocity_values, 0, ne * nq);
   velocity_y_q.MakeRef(velocity_values, ne * nq, ne * nq);
   quadrature_interpolator->Derivatives(eta_element, eta_derivatives);
   quadrature_interpolator->Values(velocity_x_element, velocity_x_q);
   quadrature_interpolator->Values(velocity_y_element, velocity_y_q);
   if (debug_device)
   {
      MFEM_VERIFY(eta_derivatives.CheckFinite() == 0,
                  "surface eta derivatives are not finite");
      MFEM_VERIFY(velocity_values.CheckFinite() == 0,
                  "surface velocity values are not finite");
   }

   const real_t *W = ir.GetWeights().Read();
   const real_t *J = element_values.Read() + 4*ne*nd;
   const real_t *E = eta_derivatives.Read();
   const real_t *U = velocity_values.Read();
   real_t *Q = quadrature_load.Write();
   const int nq_ne = nq * ne;
   mfem::forall(ne * nq, [=] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % nq;
      const int e = i / nq;
      const real_t jacobian = J[q + nq*e];
      const real_t value = U[i + nq_ne] - U[i] * E[i] / jacobian;
      Q[i] = W[q] * fabs(jacobian) * value;
   });
   if (debug_device)
   {
      MFEM_VERIFY(quadrature_load.CheckFinite() == 0,
                  "surface quadrature load is not finite");
   }

   const real_t *B = maps->B.Read();
   const real_t *QL = quadrature_load.Read();
   real_t *EL = element_load.Write();
   mfem::forall(ne * nd, [=] MFEM_HOST_DEVICE(int i)
   {
      const int d = i % nd;
      const int e = i / nd;
      real_t value = 0.0;
      for (int q = 0; q < nq1d; ++q)
      {
         value += B[q + nq1d*d] * QL[q + nq*e];
      }
      EL[i] = value;
   });
   load = 0.0;
   element_restriction->MultTranspose(element_load, load);
   if (debug_device)
   {
      MFEM_VERIFY(load.CheckFinite() == 0,
                  "surface volume load is not finite");
   }

   if (face_restriction->Height())
   {
      Vector component;
      component.MakeRef(element_values, 0, scalar_size);
      const real_t *packed = x.Read();
      real_t *scalar = element_values.ReadWrite();
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         scalar[i] = packed[i];
      });
      face_restriction->Mult(component, eta_face);
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         scalar[i] = packed[i + scalar_size];
      });
      face_restriction->Mult(component, velocity_face);
      if (debug_device)
      {
         MFEM_VERIFY(eta_face.CheckFinite() == 0,
                     "surface eta face values are not finite");
         MFEM_VERIFY(velocity_face.CheckFinite() == 0,
                     "surface velocity face values are not finite");
      }
      const int nfaces = face_restriction->Height() / 2;
      const real_t *O = orientation_face.Read();
      const real_t *EF = eta_face.Read();
      const real_t *UF = velocity_face.Read();
      real_t *F = face_flux.Write();
      const real_t alpha = upwind_factor;
      mfem::forall(nfaces, [=] MFEM_HOST_DEVICE(int f)
      {
         const bool first_is_left = O[2*f] > 0.0;
         const int left = 2*f + (first_is_left ? 0 : 1);
         const int right = 2*f + (first_is_left ? 1 : 0);
         const real_t eta_left = EF[left];
         const real_t eta_right = EF[right];
         const real_t u_left = UF[left];
         const real_t u_right = UF[right];
         const real_t speed = 0.5 * (u_left + u_right);
         const real_t radius = alpha * fmax(fabs(u_left), fabs(u_right));
         const real_t jump = eta_left - eta_right;
         const real_t flux_left = 0.5 * (speed - radius) * jump;
         const real_t flux_right = 0.5 * (speed + radius) * jump;
         F[left] = flux_left;
         F[right] = flux_right;
      });
      volume_flux = 0.0;
      face_restriction->MultTranspose(face_flux, volume_flux);
      load += volume_flux;
      if (debug_device)
      {
         MFEM_VERIFY(load.CheckFinite() == 0,
                     "surface face-corrected load is not finite");
      }
   }

   element_restriction->Mult(load, element_load);
   const real_t *IM = inverse_mass.Read();
   const real_t *L = element_load.Read();
   real_t *R = element_rate.Write();
   mfem::forall(ne * nd, [=] MFEM_HOST_DEVICE(int i)
   {
      const int d = i % nd;
      const int e = i / nd;
      real_t value = 0.0;
      for (int j = 0; j < nd; ++j)
      {
         value += IM[d + nd*(j + nd*e)] * L[j + nd*e];
      }
      R[i] = value;
   });
   y.SetSize(scalar_size);
   y.UseDevice(true);
   y = 0.0;
   element_restriction->MultTranspose(element_rate, y);
   if (debug_device)
   {
      MFEM_VERIFY(y.CheckFinite() == 0,
                  "surface kinematic result is not finite");
   }
}

void SurfaceKinematicOperator::Mult3(const Vector &elevation,
                                     const Vector &velocity_x,
                                     const Vector &velocity_y,
                                     Vector &rate) const
{
   MFEM_VERIFY(elevation.Size() == scalar_size &&
               velocity_x.Size() == scalar_size &&
               velocity_y.Size() == scalar_size,
               "surface kinematic scalar input has wrong size");
   const real_t *E = elevation.Read();
   const real_t *U = velocity_x.Read();
   const real_t *V = velocity_y.Read();
   real_t *X = packed_input.Write();
   mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
   {
      X[i] = E[i];
      X[i + scalar_size] = U[i];
      X[i + 2*scalar_size] = V[i];
   });
   if (Device::Allows(Backend::DEBUG_DEVICE))
   {
      MFEM_VERIFY(packed_input.CheckFinite() == 0,
                  "surface packed input is not finite");
   }
   Mult(packed_input, rate);
}

SurfaceKinematicOperator2D::SurfaceKinematicOperator2D(
   FiniteElementSpace &fes_, const IntegrationRule &ir_,
   real_t upwind_factor_)
   : Operator(fes_.GetVSize(), 4 * fes_.GetVSize()), fes(fes_), ir(ir_),
     scalar_size(fes_.GetVSize()), ne(fes_.GetNE()),
     nd(fes_.GetTypicalFE()->GetDof()), nq(ir_.GetNPoints()),
     nd1d(0), nq1d(0), nf(0), face_nd(0), face_nq(0),
     sdim(fes_.GetMesh()->SpaceDimension()),
     upwind_factor(upwind_factor_)
{
   MFEM_VERIFY(fes.GetMesh()->Dimension() == 2 && sdim >= 2,
               "2D surface kinematics requires a two-dimensional trace mesh");
   MFEM_VERIFY(fes.GetVDim() == 1 && fes.GetVSize() == fes.GetTrueVSize(),
               "2D surface kinematics requires a scalar broken trace space");
   MFEM_VERIFY(fes.GetTypicalFE()->GetGeomType() == Geometry::SQUARE,
               "2D surface kinematics requires quadrilateral elements");
   SetUpwindFactor(upwind_factor_);

   element_restriction = fes.GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   quadrature_interpolator = fes.GetQuadratureInterpolator(ir);
   maps = &fes.GetTypicalFE()->GetDofToQuad(ir, DofToQuad::TENSOR);
   nd1d = maps->ndof;
   nq1d = maps->nqpt;
   MFEM_VERIFY(element_restriction != nullptr &&
               element_restriction->Height() == ne * nd &&
               quadrature_interpolator != nullptr &&
               nd1d * nd1d == nd && nq1d * nq1d == nq,
               "2D surface kinematics requires tensor quadrilaterals and rule");
   quadrature_interpolator->EnableTensorProducts();
   quadrature_interpolator->SetOutputLayout(QVectorLayout::byNODES);

   face_ir = &IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
   const FiniteElement &trace_fe = *fes.GetTypicalTraceElement();
   face_maps = &trace_fe.GetDofToQuad(*face_ir, DofToQuad::TENSOR);
   face_nd = face_maps->ndof;
   face_nq = face_maps->nqpt;
   face_restriction = fes.GetFaceRestriction(
                         ElementDofOrdering::LEXICOGRAPHIC,
                         FaceType::Interior, L2FaceValues::DoubleValued);
   FaceQuadratureSpace faces(*fes.GetMesh(), *face_ir, FaceType::Interior);
   nf = faces.GetNumFaces();
   MFEM_VERIFY((nf == 0 || face_restriction != nullptr) &&
               (!face_restriction ||
                face_restriction->Height() == 2 * nf * face_nd),
               "2D surface interior-face restriction has the wrong layout");

   inverse_mass.SetSize(ne * nd * nd);
   volume_weights.SetSize(ne * nq);
   gradient_map.SetSize(4 * ne * nq);
   Vector shape(nd);
   DenseMatrix mass(nd), inverse_matrix;
   for (int e = 0; e < ne; ++e)
   {
      const FiniteElement &fe = *fes.GetFE(e);
      ElementTransformation &T = *fes.GetElementTransformation(e);
      mass = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T.SetIntPoint(&ip);
         const DenseMatrix &J = T.Jacobian();
         MFEM_VERIFY(J.Height() >= 2 && J.Width() == 2,
                     "surface element Jacobian has the wrong shape");
         const real_t J00 = J(0, 0);
         const real_t J01 = J(0, 1);
         const real_t J10 = J(1, 0);
         const real_t J11 = J(1, 1);
         const real_t det_horizontal = J00 * J11 - J01 * J10;
         MFEM_VERIFY(std::abs(det_horizontal) > 0.0 &&
                     std::isfinite(det_horizontal),
                     "surface graph map has a singular horizontal Jacobian");
         const int base = 4 * (q + nq * e);
         gradient_map[base + 0] =  J11 / det_horizontal;
         gradient_map[base + 1] = -J01 / det_horizontal;
         gradient_map[base + 2] = -J10 / det_horizontal;
         gradient_map[base + 3] =  J00 / det_horizontal;
         volume_weights[q + nq * e] = ip.weight * std::abs(T.Weight());
         fe.CalcShape(ip, shape);
         AddMult_a_VVt(volume_weights[q + nq * e], shape, mass);
      }
      DenseMatrixInverse factor(mass, true);
      factor.GetInverseMatrix(inverse_matrix);
      for (int j = 0; j < nd; ++j)
      {
         for (int i = 0; i < nd; ++i)
         {
            inverse_mass[i + nd * (j + nd * e)] = inverse_matrix(i, j);
         }
      }
   }

   face_weights.SetSize(nf * face_nq);
   face_normals.SetSize(2 * nf * face_nq);
   Mesh &mesh = *fes.GetMesh();
   for (int f = 0; f < nf; ++f)
   {
      const int mesh_face = faces.GetMeshFaceIndex(f);
      FaceElementTransformations *T = mesh.GetFaceElementTransformations(mesh_face);
      MFEM_VERIFY(T != nullptr && T->Elem1 != nullptr && T->Face != nullptr,
                  "surface interior face transformation is incomplete");
      const IntegrationPoint &center =
         Geometries.GetCenter(T->Elem1->GetGeometryType());
      Vector element_center, face_point;
      T->Elem1->Transform(center, element_center);
      for (int q = 0; q < face_nq; ++q)
      {
         const IntegrationPoint &ip = face_ir->IntPoint(q);
         T->SetAllIntPoints(&ip);
         const DenseMatrix &J = T->Face->Jacobian();
         MFEM_VERIFY(J.Height() >= 2 && J.Width() == 1,
                     "surface edge Jacobian has the wrong shape");
         const real_t tangent_x = J(0, 0);
         const real_t tangent_y = J(1, 0);
         const real_t horizontal_measure =
            std::sqrt(tangent_x * tangent_x + tangent_y * tangent_y);
         MFEM_VERIFY(horizontal_measure > 0.0 && std::isfinite(horizontal_measure),
                     "surface edge has zero horizontal measure");
         real_t normal_x = tangent_y / horizontal_measure;
         real_t normal_y = -tangent_x / horizontal_measure;
         T->Elem1->Transform(T->GetElement1IntPoint(), face_point);
         if (normal_x * (face_point[0] - element_center[0]) +
             normal_y * (face_point[1] - element_center[1]) < 0.0)
         {
            normal_x = -normal_x;
            normal_y = -normal_y;
         }
         face_normals[2 * (q + face_nq * f) + 0] = normal_x;
         face_normals[2 * (q + face_nq * f) + 1] = normal_y;
         face_weights[q + face_nq * f] =
            ip.weight * std::abs(T->Face->Weight());
      }
   }

   element_values.SetSize(5 * ne * nd);
   eta_derivatives.SetSize(2 * ne * nq);
   velocity_values.SetSize(3 * ne * nq);
   quadrature_load.SetSize(ne * nq);
   element_load.SetSize(ne * nd);
   element_rate.SetSize(ne * nd);
   const int face_size = 2 * nf * face_nd;
   face_element_values.SetSize(4 * face_size);
   face_values.SetSize(4 * 2 * nf * face_nq);
   face_quadrature_flux.SetSize(2 * nf * face_nq);
   face_element_flux.SetSize(face_size);
   volume_flux.SetSize(scalar_size);
   load.SetSize(scalar_size);
   packed_input.SetSize(4 * scalar_size);

   inverse_mass.UseDevice(true);
   volume_weights.UseDevice(true);
   gradient_map.UseDevice(true);
   face_weights.UseDevice(true);
   face_normals.UseDevice(true);
   element_values.UseDevice(true);
   eta_derivatives.UseDevice(true);
   velocity_values.UseDevice(true);
   quadrature_load.UseDevice(true);
   element_load.UseDevice(true);
   element_rate.UseDevice(true);
   face_element_values.UseDevice(true);
   face_values.UseDevice(true);
   face_quadrature_flux.UseDevice(true);
   face_element_flux.UseDevice(true);
   volume_flux.UseDevice(true);
   load.UseDevice(true);
   packed_input.UseDevice(true);
}

void SurfaceKinematicOperator2D::SetUpwindFactor(real_t value)
{
   MFEM_VERIFY(value >= 0.0 && std::isfinite(value),
               "surface upwind factor must be finite and non-negative");
   upwind_factor = value;
}

void SurfaceKinematicOperator2D::FaceMult(const Vector &elevation,
                                          const Vector &velocity_x,
                                          const Vector &velocity_y,
                                          Vector &face_load) const
{
   MFEM_VERIFY(elevation.Size() == scalar_size &&
               velocity_x.Size() == scalar_size &&
               velocity_y.Size() == scalar_size,
               "2D surface kinematic face input has wrong size");
   face_load.SetSize(scalar_size);
   face_load.UseDevice(true);
   face_load = 0.0;
   if (nf == 0) { return; }

   const Vector *components[3] = {&elevation, &velocity_x, &velocity_y};
   const int face_size = 2 * nf * face_nd;
   for (int c = 0; c < 3; ++c)
   {
      Vector component, restricted;
      component.MakeRef(element_values, 0, scalar_size);
      const real_t *source = components[c]->Read();
      real_t *scalar = element_values.Write();
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         scalar[i] = source[i];
      });
      restricted.MakeRef(face_element_values, c * face_size, face_size);
      face_restriction->Mult(component, restricted);
   }

   const real_t *FB = face_maps->B.Read();
   const real_t *FE = face_element_values.Read();
   real_t *FV = face_values.Write();
   const int values_per_component = 2 * nf * face_nq;
   mfem::forall(3 * values_per_component, [=] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % face_nq;
      const int side = (i / face_nq) % 2;
      const int face = (i / (2 * face_nq)) % nf;
      const int component_index = i / values_per_component;
      real_t value = 0.0;
      for (int d = 0; d < face_nd; ++d)
      {
         const int source =
            d + face_nd * (side + 2 * (face + nf * component_index));
         value += FB[q + face_nq * d] * FE[source];
      }
      FV[i] = value;
   });

   const real_t *N = face_normals.Read();
   const real_t *FW = face_weights.Read();
   const real_t *S = face_values.Read();
   real_t *FQ = face_quadrature_flux.Write();
   const real_t alpha = upwind_factor;
   mfem::forall(nf * face_nq, [=] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % face_nq;
      const int face = i / face_nq;
      const int side0 = q + face_nq * (0 + 2 * face);
      const int side1 = q + face_nq * (1 + 2 * face);
      const real_t normal_x = N[2 * i + 0];
      const real_t normal_y = N[2 * i + 1];
      const real_t eta0 = S[side0];
      const real_t eta1 = S[side1];
      const real_t ux0 = S[side0 + values_per_component];
      const real_t ux1 = S[side1 + values_per_component];
      const real_t uy0 = S[side0 + 2 * values_per_component];
      const real_t uy1 = S[side1 + 2 * values_per_component];
      const real_t speed0 = ux0 * normal_x + uy0 * normal_y;
      const real_t speed1 = ux1 * normal_x + uy1 * normal_y;
      const real_t speed = 0.5 * (speed0 + speed1);
      const real_t radius = alpha * fmax(fabs(speed0), fabs(speed1));
      const real_t jump = eta0 - eta1;
      const real_t scale = FW[i];
      FQ[side0] = scale * 0.5 * (speed - radius) * jump;
      FQ[side1] = scale * 0.5 * (speed + radius) * jump;
   });

   const real_t *FBt = face_maps->Bt.Read();
   const real_t *FQL = face_quadrature_flux.Read();
   real_t *FEL = face_element_flux.Write();
   mfem::forall(2 * nf * face_nd, [=] MFEM_HOST_DEVICE(int i)
   {
      const int d = i % face_nd;
      const int side = (i / face_nd) % 2;
      const int face = i / (2 * face_nd);
      real_t value = 0.0;
      for (int q = 0; q < face_nq; ++q)
      {
         value += FBt[d + face_nd * q] *
                  FQL[q + face_nq * (side + 2 * face)];
      }
      FEL[i] = value;
   });
   face_restriction->MultTranspose(face_element_flux, face_load);
}

void SurfaceKinematicOperator2D::Mult(const Vector &x, Vector &y) const
{
   const bool debug_device = Device::Allows(Backend::DEBUG_DEVICE);
   MFEM_VERIFY(x.Size() == Width(), "2D surface kinematic input has wrong size");
   const real_t *X = x.Read();
   real_t *scratch = element_values.Write();
   for (int c = 0; c < 4; ++c)
   {
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         scratch[i] = X[i + c * scalar_size];
      });
      Vector component, restricted;
      component.MakeRef(element_values, 0, scalar_size);
      restricted.MakeRef(element_values, (1 + c) * ne * nd, ne * nd);
      element_restriction->Mult(component, restricted);
   }

   Vector eta_element;
   eta_element.MakeRef(element_values, ne * nd, ne * nd);
   quadrature_interpolator->Derivatives(eta_element, eta_derivatives);
   for (int c = 0; c < 3; ++c)
   {
      Vector velocity_element, velocity_q;
      velocity_element.MakeRef(element_values, (2 + c) * ne * nd, ne * nd);
      velocity_q.MakeRef(velocity_values, c * ne * nq, ne * nq);
      quadrature_interpolator->Values(velocity_element, velocity_q);
   }
   if (debug_device)
   {
      MFEM_VERIFY(eta_derivatives.CheckFinite() == 0 &&
                  velocity_values.CheckFinite() == 0,
                  "2D surface quadrature state is not finite");
   }

   const real_t *D = eta_derivatives.Read();
   const real_t *U = velocity_values.Read();
   const real_t *G = gradient_map.Read();
   const real_t *VW = volume_weights.Read();
   real_t *Q = quadrature_load.Write();
   const int nq_ne = nq * ne;
   mfem::forall(nq_ne, [=] MFEM_HOST_DEVICE(int i)
   {
      const int q = i % nq;
      const int e = i / nq;
      const int base = 4 * i;
      // QVectorLayout::byNODES stores reference derivatives as
      // (quadrature point, component, derivative, element). For this scalar
      // space the derivative index therefore sits inside the element block,
      // not in one global derivative block.
      const real_t derivative_0 = D[q + nq * (0 + 2 * e)];
      const real_t derivative_1 = D[q + nq * (1 + 2 * e)];
      const real_t gradient_x =
         G[base + 0] * derivative_0 + G[base + 2] * derivative_1;
      const real_t gradient_y =
         G[base + 1] * derivative_0 + G[base + 3] * derivative_1;
      Q[i] = VW[i] *
             (U[i + 2 * nq_ne] - U[i] * gradient_x -
              U[i + nq_ne] * gradient_y);
   });

   const real_t *B = maps->B.Read();
   const real_t *QL = quadrature_load.Read();
   real_t *EL = element_load.Write();
   mfem::forall(ne * nd, [=] MFEM_HOST_DEVICE(int i)
   {
      const int d1 = i % nd1d;
      const int d2 = (i / nd1d) % nd1d;
      const int e = i / nd;
      real_t value = 0.0;
      for (int q2 = 0; q2 < nq1d; ++q2)
      {
         for (int q1 = 0; q1 < nq1d; ++q1)
         {
            value += B[q1 + nq1d * d1] * B[q2 + nq1d * d2] *
                     QL[q1 + nq1d * q2 + nq * e];
         }
      }
      EL[i] = value;
   });
   load = 0.0;
   element_restriction->MultTranspose(element_load, load);

   if (nf > 0)
   {
      const int face_size = 2 * nf * face_nd;
      for (int c = 0; c < 4; ++c)
      {
         Vector component, restricted;
         component.MakeRef(element_values, 0, scalar_size);
         const real_t *packed = x.Read();
         real_t *scalar = element_values.Write();
         mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
         {
            scalar[i] = packed[i + c * scalar_size];
         });
         restricted.MakeRef(face_element_values, c * face_size, face_size);
         face_restriction->Mult(component, restricted);
      }

      const real_t *FB = face_maps->B.Read();
      const real_t *FE = face_element_values.Read();
      real_t *FV = face_values.Write();
      const int values_per_component = 2 * nf * face_nq;
      mfem::forall(4 * values_per_component, [=] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % face_nq;
         const int side = (i / face_nq) % 2;
         const int face = (i / (2 * face_nq)) % nf;
         const int component = i / values_per_component;
         real_t value = 0.0;
         for (int d = 0; d < face_nd; ++d)
         {
            const int source =
               d + face_nd * (side + 2 * (face + nf * component));
            value += FB[q + face_nq * d] * FE[source];
         }
         FV[i] = value;
      });

      const real_t *N = face_normals.Read();
      const real_t *FW = face_weights.Read();
      const real_t *S = face_values.Read();
      real_t *FQ = face_quadrature_flux.Write();
      const real_t alpha = upwind_factor;
      mfem::forall(nf * face_nq, [=] MFEM_HOST_DEVICE(int i)
      {
         const int q = i % face_nq;
         const int face = i / face_nq;
         const int side0 = q + face_nq * (0 + 2 * face);
         const int side1 = q + face_nq * (1 + 2 * face);
         const real_t normal_x = N[2 * i + 0];
         const real_t normal_y = N[2 * i + 1];
         const real_t eta0 = S[side0];
         const real_t eta1 = S[side1];
         const real_t ux0 = S[side0 + values_per_component];
         const real_t ux1 = S[side1 + values_per_component];
         const real_t uy0 = S[side0 + 2 * values_per_component];
         const real_t uy1 = S[side1 + 2 * values_per_component];
         const real_t speed0 = ux0 * normal_x + uy0 * normal_y;
         const real_t speed1 = ux1 * normal_x + uy1 * normal_y;
         const real_t speed = 0.5 * (speed0 + speed1);
         const real_t radius = alpha * fmax(fabs(speed0), fabs(speed1));
         const real_t jump = eta0 - eta1;
         const real_t scale = FW[i];
         FQ[side0] = scale * 0.5 * (speed - radius) * jump;
         FQ[side1] = scale * 0.5 * (speed + radius) * jump;
      });

      const real_t *FBt = face_maps->Bt.Read();
      const real_t *FQL = face_quadrature_flux.Read();
      real_t *FEL = face_element_flux.Write();
      mfem::forall(2 * nf * face_nd, [=] MFEM_HOST_DEVICE(int i)
      {
         const int d = i % face_nd;
         const int side = (i / face_nd) % 2;
         const int face = i / (2 * face_nd);
         real_t value = 0.0;
         for (int q = 0; q < face_nq; ++q)
         {
            value += FBt[d + face_nd * q] *
                     FQL[q + face_nq * (side + 2 * face)];
         }
         FEL[i] = value;
      });
      volume_flux = 0.0;
      face_restriction->MultTranspose(face_element_flux, volume_flux);
      load += volume_flux;
   }

   element_restriction->Mult(load, element_load);
   const real_t *IM = inverse_mass.Read();
   const real_t *L = element_load.Read();
   real_t *R = element_rate.Write();
   mfem::forall(ne * nd, [=] MFEM_HOST_DEVICE(int i)
   {
      const int d = i % nd;
      const int e = i / nd;
      real_t value = 0.0;
      for (int j = 0; j < nd; ++j)
      {
         value += IM[d + nd * (j + nd * e)] * L[j + nd * e];
      }
      R[i] = value;
   });
   y.SetSize(scalar_size);
   y.UseDevice(true);
   y = 0.0;
   element_restriction->MultTranspose(element_rate, y);
   if (debug_device)
   {
      MFEM_VERIFY(y.CheckFinite() == 0,
                  "2D surface kinematic result is not finite");
   }
}

void SurfaceKinematicOperator2D::Mult4(
   const Vector &elevation, const Vector &velocity_x,
   const Vector &velocity_y, const Vector &velocity_z, Vector &rate) const
{
   MFEM_VERIFY(elevation.Size() == scalar_size &&
               velocity_x.Size() == scalar_size &&
               velocity_y.Size() == scalar_size &&
               velocity_z.Size() == scalar_size,
               "2D surface kinematic scalar input has wrong size");
   const real_t *E = elevation.Read();
   const real_t *U = velocity_x.Read();
   const real_t *V = velocity_y.Read();
   const real_t *W = velocity_z.Read();
   real_t *X = packed_input.Write();
   mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
   {
      X[i] = E[i];
      X[i + scalar_size] = U[i];
      X[i + 2 * scalar_size] = V[i];
      X[i + 3 * scalar_size] = W[i];
   });
   Mult(packed_input, rate);
}

ElementMeanMagnitudeOperator::ElementMeanMagnitudeOperator(
   FiniteElementSpace &fes_, const IntegrationRule &ir_)
   : Operator(fes_.GetNE(),
              fes_.GetMesh()->Dimension() * fes_.GetVSize()),
     fes(fes_), ir(ir_), dim(fes_.GetMesh()->Dimension()),
     scalar_size(fes_.GetVSize()), ne(fes_.GetNE()),
     nd(fes_.GetTypicalFE()->GetDof()), nq(ir_.GetNPoints()),
     element_values(2 * ne * nd), quadrature_values(dim * ne * nq),
     quadrature_derivatives(dim * dim * ne * nq), volumes(ne),
     scratch_mean_magnitude(ne), scratch_divergence_rms(ne)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "element mean magnitude supports 2D and 3D meshes");
   MFEM_VERIFY(fes.GetVDim() == 1,
               "element mean magnitude requires a scalar base space");
   MFEM_VERIFY(fes.GetVSize() == fes.GetTrueVSize(),
               "element mean magnitude requires a broken scalar space");
   element_restriction = fes.GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   quadrature_interpolator = fes.GetQuadratureInterpolator(ir);
   MFEM_VERIFY(element_restriction != nullptr &&
               element_restriction->Height() == ne * nd,
               "element mean magnitude requires a tensor element restriction");
   MFEM_VERIFY(quadrature_interpolator != nullptr,
               "element mean magnitude requires a quadrature interpolator");
   quadrature_interpolator->EnableTensorProducts();
   quadrature_interpolator->SetOutputLayout(QVectorLayout::byNODES);
   element_values.UseDevice(true);
   quadrature_values.UseDevice(true);
   quadrature_derivatives.UseDevice(true);
   volumes.UseDevice(true);
   scratch_mean_magnitude.UseDevice(true);
   scratch_divergence_rms.UseDevice(true);
}

void ElementMeanMagnitudeOperator::Mult(const Vector &x, Vector &y) const
{
   ComputeMeasures(x, y, scratch_mean_magnitude, scratch_divergence_rms);
}

void ElementMeanMagnitudeOperator::ComputeMeasures(
   const Vector &x, Vector &vector_mean, Vector &mean_magnitude,
   Vector &divergence_rms) const
{
   MFEM_VERIFY(x.Size() == Width(), "element mean input has the wrong size");
   vector_mean.SetSize(Height());
   mean_magnitude.SetSize(Height());
   divergence_rms.SetSize(Height());
   vector_mean.UseDevice(true);
   mean_magnitude.UseDevice(true);
   divergence_rms.UseDevice(true);
   for (int c = 0; c < dim; ++c)
   {
      Vector component, restricted_component, q_component, q_derivatives;
      component.MakeRef(element_values, 0, ne * nd);
      restricted_component.MakeRef(element_values, ne * nd, ne * nd);
      q_component.MakeRef(quadrature_values, c * ne * nq, ne * nq);
      q_derivatives.MakeRef(quadrature_derivatives,
                            c * dim * ne * nq, dim * ne * nq);
      const real_t *X = x.Read();
      real_t *C = component.Write();
      const int offset = c * scalar_size;
      mfem::forall(scalar_size, [=] MFEM_HOST_DEVICE(int i)
      {
         C[i] = X[offset + i];
      });
      element_restriction->Mult(component, restricted_component);
      quadrature_interpolator->Values(restricted_component, q_component);
      quadrature_interpolator->PhysDerivatives(restricted_component,
                                               q_derivatives);
   }

   const GeometricFactors *geom = fes.GetMesh()->GetGeometricFactors(
                                     ir, GeometricFactors::DETERMINANTS);
   const real_t *W = ir.GetWeights().Read();
   const real_t *J = geom->detJ.Read();
   const real_t *Q = quadrature_values.Read();
   const real_t *D = quadrature_derivatives.Read();
   real_t *V = volumes.Write();
   real_t *VM = vector_mean.Write();
   real_t *MM = mean_magnitude.Write();
   real_t *DR = divergence_rms.Write();
   const int nqe = nq * ne;
   mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e)
   {
      real_t integral[3] = {0.0, 0.0, 0.0};
      real_t magnitude_integral = 0.0;
      real_t divergence_squared_integral = 0.0;
      real_t volume = 0.0;
      for (int q = 0; q < nq; ++q)
      {
         const real_t weight = W[q] * fabs(J[q + nq * e]);
         volume += weight;
         real_t magnitude2 = 0.0;
         real_t divergence = 0.0;
         for (int c = 0; c < dim; ++c)
         {
            const real_t value = Q[q + nq * e + c * nqe];
            integral[c] += weight * value;
            magnitude2 += value * value;
            // Each component owns a scalar derivative block laid out as
            // (quadrature point, physical derivative, element).
            divergence += D[q + nq * (c + dim * e) + c * dim * nqe];
         }
         magnitude_integral += weight * sqrt(magnitude2);
         divergence_squared_integral += weight * divergence * divergence;
      }
      V[e] = volume;
      real_t magnitude2 = 0.0;
      for (int c = 0; c < dim; ++c)
      {
         const real_t mean = integral[c] / volume;
         magnitude2 += mean * mean;
      }
      VM[e] = sqrt(magnitude2);
      MM[e] = magnitude_integral / volume;
      DR[e] = sqrt(divergence_squared_integral / volume);
   });
}

void ElementMeanMagnitudeOperator::ComputeTau(
   const Vector &mean, real_t scale, Vector &tau) const
{
   MFEM_VERIFY(mean.Size() == ne, "element mean vector has the wrong size");
   tau.SetSize(ne);
   tau.UseDevice(true);
   const real_t *M = mean.Read();
   const real_t *V = volumes.Read();
   real_t *T = tau.Write();
   const real_t inverse_dimension = 1.0 / dim;
   mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e)
   {
      T[e] = scale * M[e] * pow(V[e], inverse_dimension);
   });
}

FehnALECFLRateOperator::FehnALECFLRateOperator(
   FiniteElementSpace &fes_, const IntegrationRule &ir_)
   : Operator(fes_.GetNE(), fes_.GetVSize()), fes(fes_), ir(ir_),
     dim(fes_.GetMesh()->Dimension()), ne(fes_.GetNE()),
     nd(fes_.GetTypicalFE()->GetDof()), nq(ir_.GetNPoints()),
     element_values(dim * ne * nd), quadrature_values(dim * ne * nq),
     element_rates(ne)
{
   MFEM_VERIFY(dim == 2 || dim == 3,
               "Fehn ALE CFL rate supports 2D and 3D meshes");
   MFEM_VERIFY(fes.GetMesh()->SpaceDimension() == dim,
               "Fehn ALE CFL rate requires a full-dimensional mesh");
   MFEM_VERIFY(fes.GetVDim() == dim,
               "Fehn ALE CFL rate requires one velocity component per dimension");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "Fehn ALE CFL rate requires byNODES vector ordering");
   MFEM_VERIFY(fes.GetVSize() == fes.GetTrueVSize(),
               "Fehn ALE CFL rate requires a broken vector space");
   element_restriction = fes.GetElementRestriction(
                            ElementDofOrdering::LEXICOGRAPHIC);
   quadrature_interpolator = fes.GetQuadratureInterpolator(ir);
   MFEM_VERIFY(element_restriction != nullptr &&
               element_restriction->Height() == dim * ne * nd,
               "Fehn ALE CFL rate requires a tensor element restriction");
   MFEM_VERIFY(quadrature_interpolator != nullptr,
               "Fehn ALE CFL rate requires a quadrature interpolator");
   quadrature_interpolator->EnableTensorProducts();
   quadrature_interpolator->SetOutputLayout(QVectorLayout::byNODES);
   element_values.UseDevice(true);
   quadrature_values.UseDevice(true);
   element_rates.UseDevice(true);
}

void FehnALECFLRateOperator::Mult(const Vector &relative_velocity,
                                  Vector &element_rate) const
{
   MFEM_VERIFY(relative_velocity.Size() == Width(),
               "Fehn ALE CFL relative velocity has the wrong size");
   element_rate.SetSize(Height());
   element_rate.UseDevice(true);
   if (ne == 0) { return; }

   element_restriction->Mult(relative_velocity, element_values);
   quadrature_interpolator->Values(element_values, quadrature_values);
   const GeometricFactors *geom = fes.GetMesh()->GetGeometricFactors(
                                     ir, GeometricFactors::JACOBIANS);
   const int NE = ne;
   const int NQ = nq;
   const real_t *Q = quadrature_values.Read();
   real_t *R = element_rate.Write();

   if (dim == 2)
   {
      auto U = Reshape(Q, NQ, 2, NE);
      auto J = Reshape(geom->J.Read(), NQ, 2, 2, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
      {
         real_t maximum = 0.0;
         for (int q = 0; q < NQ; ++q)
         {
            const real_t a = J(q,0,0,e);
            const real_t b = J(q,0,1,e);
            const real_t c = J(q,1,0,e);
            const real_t d = J(q,1,1,e);
            const real_t inverse_det = 1.0 / (a*d - b*c);
            const real_t u0 = U(q,0,e);
            const real_t u1 = U(q,1,e);
            const real_t w0 = (d*u0 - c*u1) * inverse_det;
            const real_t w1 = (a*u1 - b*u0) * inverse_det;
            maximum = fmax(maximum, sqrt(w0*w0 + w1*w1));
         }
         R[e] = maximum;
      });
   }
   else
   {
      auto U = Reshape(Q, NQ, 3, NE);
      auto J = Reshape(geom->J.Read(), NQ, 3, 3, NE);
      mfem::forall(NE, [=] MFEM_HOST_DEVICE(int e)
      {
         real_t maximum = 0.0;
         for (int q = 0; q < NQ; ++q)
         {
            const real_t a = J(q,0,0,e);
            const real_t b = J(q,0,1,e);
            const real_t c = J(q,0,2,e);
            const real_t d = J(q,1,0,e);
            const real_t f = J(q,1,2,e);
            const real_t g = J(q,2,0,e);
            const real_t h = J(q,2,1,e);
            const real_t i = J(q,2,2,e);
            const real_t j11 = J(q,1,1,e);
            const real_t determinant =
               a*(j11*i - f*h) - b*(d*i - f*g) + c*(d*h - j11*g);
            const real_t inverse_det = 1.0 / determinant;
            const real_t u0 = U(q,0,e);
            const real_t u1 = U(q,1,e);
            const real_t u2 = U(q,2,e);
            const real_t w0 = ((j11*i-f*h)*u0 + (f*g-d*i)*u1 +
                               (d*h-j11*g)*u2) * inverse_det;
            const real_t w1 = ((c*h-b*i)*u0 + (a*i-c*g)*u1 +
                               (b*g-a*h)*u2) * inverse_det;
            const real_t w2 = ((b*f-c*j11)*u0 + (c*d-a*f)*u1 +
                               (a*j11-b*d)*u2) * inverse_det;
            maximum = fmax(maximum, sqrt(w0*w0 + w1*w1 + w2*w2));
         }
         R[e] = maximum;
      });
   }
}

real_t FehnALECFLRateOperator::ComputeMax(
   const Vector &relative_velocity) const
{
   Mult(relative_velocity, element_rates);
   return ne > 0 ? element_rates.Max() : 0.0;
}

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
