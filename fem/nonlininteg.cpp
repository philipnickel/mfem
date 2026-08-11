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

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
static constexpr int ALE_FACE_MAX_D1D = 10;
static constexpr int ALE_FACE_MAX_Q1D = 10;
#else
static constexpr int ALE_FACE_MAX_D1D = 16;
static constexpr int ALE_FACE_MAX_Q1D = 16;
#endif

static void PAALEConvectionVolumeApply2D(
   const int history_order, const int d1d, const int q1d, const int ne,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &jacobians, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE volume PA size exceeds the configured limit");
   const int dim = 2;
   const int vdim = dim * (history_order + 1);
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto G = Reshape(maps.G.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto J = Reshape(jacobians.Read(), q1d, q1d, dim, dim, ne);
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, d1d, vdim, ne);
   auto y = Reshape(y_.ReadWrite(), d1d, d1d, vdim, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE(int element)
   {
      real_t flux[ALE_FACE_MAX_Q1D][ALE_FACE_MAX_Q1D][3];
      for (int q2 = 0; q2 < q1d; ++q2)
      {
         for (int q1 = 0; q1 < q1d; ++q1)
         {
            real_t state[12] = {0.0};
            real_t gradient[12][2] = {{0.0}};
            for (int d2 = 0; d2 < d1d; ++d2)
            {
               for (int d1 = 0; d1 < d1d; ++d1)
               {
                  const real_t basis = B(q1, d1) * B(q2, d2);
                  const real_t derivative0 = G(q1, d1) * B(q2, d2);
                  const real_t derivative1 = B(q1, d1) * G(q2, d2);
                  for (int field = 0; field < vdim; ++field)
                  {
                     const real_t value = x(d1, d2, field, element);
                     state[field] += basis * value;
                     gradient[field][0] += derivative0 * value;
                     gradient[field][1] += derivative1 * value;
                  }
               }
            }

            const real_t J11 = J(q1, q2, 0, 0, element);
            const real_t J12 = J(q1, q2, 0, 1, element);
            const real_t J21 = J(q1, q2, 1, 0, element);
            const real_t J22 = J(q1, q2, 1, 1, element);
            const real_t adjugate[2][2] =
            {
               {J22, -J12},
               {-J21, J11}
            };
            for (int component = 0; component < dim; ++component)
            {
               real_t load = 0.0;
               for (int history = 0; history < history_order; ++history)
               {
                  const int velocity = dim * history + component;
                  for (int physical = 0; physical < dim; ++physical)
                  {
                     const real_t relative =
                        state[dim * history + physical] -
                        state[dim * history_order + physical];
                     for (int reference = 0; reference < dim; ++reference)
                     {
                        load += beta[history] * relative *
                           gradient[velocity][reference] *
                           adjugate[reference][physical];
                     }
                  }
               }
               flux[q1][q2][component] =
                  weight[q1 + q1d * q2] * load;
            }
         }
      }

      for (int d2 = 0; d2 < d1d; ++d2)
      {
         for (int d1 = 0; d1 < d1d; ++d1)
         {
            for (int component = 0; component < dim; ++component)
            {
               real_t load = 0.0;
               for (int q2 = 0; q2 < q1d; ++q2)
               {
                  for (int q1 = 0; q1 < q1d; ++q1)
                  {
                     load += Bt(d1, q1) * Bt(d2, q2) *
                             flux[q1][q2][component];
                  }
               }
               y(d1, d2, component, element) += load;
            }
         }
      }
   });
}

static void PAALEConvectionVolumeApply3D(
   const int history_order, const int d1d, const int q1d, const int ne,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &jacobians, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE volume PA size exceeds the configured limit");
   const int dim = 3;
   const int vdim = dim * (history_order + 1);
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto G = Reshape(maps.G.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto J = Reshape(jacobians.Read(), q1d, q1d, q1d, dim, dim, ne);
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, d1d, d1d, vdim, ne);
   auto y = Reshape(y_.ReadWrite(), d1d, d1d, d1d, vdim, ne);

   mfem::forall(ne, [=] MFEM_HOST_DEVICE(int element)
   {
      real_t flux[ALE_FACE_MAX_Q1D][ALE_FACE_MAX_Q1D]
                 [ALE_FACE_MAX_Q1D][3];
      for (int q3 = 0; q3 < q1d; ++q3)
      {
         for (int q2 = 0; q2 < q1d; ++q2)
         {
            for (int q1 = 0; q1 < q1d; ++q1)
            {
               real_t state[12] = {0.0};
               real_t gradient[12][3] = {{0.0}};
               for (int d3 = 0; d3 < d1d; ++d3)
               {
                  for (int d2 = 0; d2 < d1d; ++d2)
                  {
                     for (int d1 = 0; d1 < d1d; ++d1)
                     {
                        const real_t b1 = B(q1, d1);
                        const real_t b2 = B(q2, d2);
                        const real_t b3 = B(q3, d3);
                        const real_t basis = b1 * b2 * b3;
                        const real_t derivative0 = G(q1, d1) * b2 * b3;
                        const real_t derivative1 = b1 * G(q2, d2) * b3;
                        const real_t derivative2 = b1 * b2 * G(q3, d3);
                        for (int field = 0; field < vdim; ++field)
                        {
                           const real_t value =
                              x(d1, d2, d3, field, element);
                           state[field] += basis * value;
                           gradient[field][0] += derivative0 * value;
                           gradient[field][1] += derivative1 * value;
                           gradient[field][2] += derivative2 * value;
                        }
                     }
                  }
               }

               const real_t J11 = J(q1, q2, q3, 0, 0, element);
               const real_t J12 = J(q1, q2, q3, 0, 1, element);
               const real_t J13 = J(q1, q2, q3, 0, 2, element);
               const real_t J21 = J(q1, q2, q3, 1, 0, element);
               const real_t J22 = J(q1, q2, q3, 1, 1, element);
               const real_t J23 = J(q1, q2, q3, 1, 2, element);
               const real_t J31 = J(q1, q2, q3, 2, 0, element);
               const real_t J32 = J(q1, q2, q3, 2, 1, element);
               const real_t J33 = J(q1, q2, q3, 2, 2, element);
               const real_t adjugate[3][3] =
               {
                  {J22 * J33 - J23 * J32,
                   J32 * J13 - J12 * J33,
                   J12 * J23 - J22 * J13},
                  {J31 * J23 - J21 * J33,
                   J11 * J33 - J13 * J31,
                   J21 * J13 - J11 * J23},
                  {J21 * J32 - J31 * J22,
                   J31 * J12 - J11 * J32,
                   J11 * J22 - J12 * J21}
               };
               for (int component = 0; component < dim; ++component)
               {
                  real_t load = 0.0;
                  for (int history = 0; history < history_order; ++history)
                  {
                     const int velocity = dim * history + component;
                     for (int physical = 0; physical < dim; ++physical)
                     {
                        const real_t relative =
                           state[dim * history + physical] -
                           state[dim * history_order + physical];
                        for (int reference = 0; reference < dim; ++reference)
                        {
                           load += beta[history] * relative *
                              gradient[velocity][reference] *
                              adjugate[reference][physical];
                        }
                     }
                  }
                  const int q = q1 + q1d * (q2 + q1d * q3);
                  flux[q1][q2][q3][component] = weight[q] * load;
               }
            }
         }
      }

      for (int d3 = 0; d3 < d1d; ++d3)
      {
         for (int d2 = 0; d2 < d1d; ++d2)
         {
            for (int d1 = 0; d1 < d1d; ++d1)
            {
               for (int component = 0; component < dim; ++component)
               {
                  real_t load = 0.0;
                  for (int q3 = 0; q3 < q1d; ++q3)
                  {
                     for (int q2 = 0; q2 < q1d; ++q2)
                     {
                        for (int q1 = 0; q1 < q1d; ++q1)
                        {
                           load += Bt(d1, q1) * Bt(d2, q2) * Bt(d3, q3) *
                                   flux[q1][q2][q3][component];
                        }
                     }
                  }
                  y(d1, d2, d3, component, element) += load;
               }
            }
         }
      }
   });
}

static void PAALEConvectionInteriorApply2D(
   const int history_order, const real_t upwind, const int d1d,
   const int q1d, const int nf, const Array<real_t> &weights,
   const DofToQuad &maps, const Vector &determinants,
   const Vector &normals, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE interior PA face size exceeds the configured limit");
   const int dim = 2;
   const int vdim = dim * (history_order + 1);
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, dim, nf);
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, vdim, 2, nf);
   auto y = Reshape(y_.ReadWrite(), d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D][3][2];
      for (int q = 0; q < q1d; ++q)
      {
         real_t state[12][2];
         for (int component = 0; component < vdim; ++component)
         {
            state[component][0] = 0.0;
            state[component][1] = 0.0;
            for (int dof = 0; dof < d1d; ++dof)
            {
               const real_t basis = B(q, dof);
               state[component][0] += basis * x(dof, component, 0, face);
               state[component][1] += basis * x(dof, component, 1, face);
            }
         }
         for (int component = 0; component < dim; ++component)
         {
            flux[q][component][0] = 0.0;
            flux[q][component][1] = 0.0;
         }
         for (int history = 0; history < history_order; ++history)
         {
            real_t speed = 0.0;
            for (int component = 0; component < dim; ++component)
            {
               const int velocity = dim * history + component;
               const int grid = dim * history_order + component;
               speed += 0.5 * (state[velocity][0] + state[velocity][1] -
                               state[grid][0] - state[grid][1]) *
                        normal(q, component, face);
            }
            const real_t dissipation = upwind * fabs(speed);
            const real_t first = 0.5 * (-speed + dissipation);
            const real_t second = 0.5 * (-speed - dissipation);
            const real_t scale = weight[q] * det(q, face) * beta[history];
            for (int component = 0; component < dim; ++component)
            {
               const int velocity = dim * history + component;
               const real_t jump = state[velocity][0] - state[velocity][1];
               flux[q][component][0] += scale * first * jump;
               flux[q][component][1] += scale * second * jump;
            }
         }
      }
      for (int dof = 0; dof < d1d; ++dof)
      {
         for (int component = 0; component < dim; ++component)
         {
            real_t first = 0.0;
            real_t second = 0.0;
            for (int q = 0; q < q1d; ++q)
            {
               first += Bt(dof, q) * flux[q][component][0];
               second += Bt(dof, q) * flux[q][component][1];
            }
            y(dof, component, 0, face) += first;
            y(dof, component, 1, face) += second;
         }
      }
   });
}

static void PAALEConvectionInteriorApply3D(
   const int history_order, const real_t upwind, const int d1d,
   const int q1d, const int nf, const Array<real_t> &weights,
   const DofToQuad &maps, const Vector &determinants,
   const Vector &normals, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE interior PA face size exceeds the configured limit");
   const int dim = 3;
   const int vdim = dim * (history_order + 1);
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, q1d, dim, nf);
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, d1d, vdim, 2, nf);
   auto y = Reshape(y_.ReadWrite(), d1d, d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D][ALE_FACE_MAX_Q1D][3][2];
      for (int q2 = 0; q2 < q1d; ++q2)
      {
         for (int q1 = 0; q1 < q1d; ++q1)
         {
            real_t state[12][2];
            for (int component = 0; component < vdim; ++component)
            {
               state[component][0] = 0.0;
               state[component][1] = 0.0;
               for (int d2 = 0; d2 < d1d; ++d2)
               {
                  for (int d1 = 0; d1 < d1d; ++d1)
                  {
                     const real_t basis = B(q1, d1) * B(q2, d2);
                     state[component][0] +=
                        basis * x(d1, d2, component, 0, face);
                     state[component][1] +=
                        basis * x(d1, d2, component, 1, face);
                  }
               }
            }
            for (int component = 0; component < dim; ++component)
            {
               flux[q1][q2][component][0] = 0.0;
               flux[q1][q2][component][1] = 0.0;
            }
            for (int history = 0; history < history_order; ++history)
            {
               real_t speed = 0.0;
               for (int component = 0; component < dim; ++component)
               {
                  const int velocity = dim * history + component;
                  const int grid = dim * history_order + component;
                  speed += 0.5 * (state[velocity][0] + state[velocity][1] -
                                  state[grid][0] - state[grid][1]) *
                           normal(q1, q2, component, face);
               }
               const real_t dissipation = upwind * fabs(speed);
               const real_t first = 0.5 * (-speed + dissipation);
               const real_t second = 0.5 * (-speed - dissipation);
               const real_t scale =
                  weight[q1 + q2 * q1d] * det(q1, q2, face) * beta[history];
               for (int component = 0; component < dim; ++component)
               {
                  const int velocity = dim * history + component;
                  const real_t jump = state[velocity][0] - state[velocity][1];
                  flux[q1][q2][component][0] += scale * first * jump;
                  flux[q1][q2][component][1] += scale * second * jump;
               }
            }
         }
      }
      for (int d2 = 0; d2 < d1d; ++d2)
      {
         for (int d1 = 0; d1 < d1d; ++d1)
         {
            for (int component = 0; component < dim; ++component)
            {
               real_t first = 0.0;
               real_t second = 0.0;
               for (int q2 = 0; q2 < q1d; ++q2)
               {
                  for (int q1 = 0; q1 < q1d; ++q1)
                  {
                     const real_t basis = Bt(d1, q1) * Bt(d2, q2);
                     first += basis * flux[q1][q2][component][0];
                     second += basis * flux[q1][q2][component][1];
                  }
               }
               y(d1, d2, component, 0, face) += first;
               y(d1, d2, component, 1, face) += second;
            }
         }
      }
   });
}

static void PAALEConvectionBoundaryApply2D(
   const int history_order, const bool continuity_scratch,
   const real_t upwind, const int d1d, const int q1d, const int nf,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &determinants, const Vector &normals,
   const Vector &datum_values, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE boundary PA face size exceeds the configured limit");
   const int dim = 2;
   const int vdim = dim * (history_order + 1 + (continuity_scratch ? 1 : 0));
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, dim, nf);
   const bool constant_datum = datum_values.Size() == dim;
   auto datum = datum_values.Read();
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, vdim, 2, nf);
   auto y = Reshape(y_.ReadWrite(), d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D][3];
      for (int q = 0; q < q1d; ++q)
      {
         real_t state[15];
         for (int component = 0; component < vdim; ++component)
         {
            state[component] = 0.0;
            for (int dof = 0; dof < d1d; ++dof)
            {
               state[component] += B(q, dof) * x(dof, component, 0, face);
            }
         }
         real_t speed = 0.0;
         for (int component = 0; component < dim; ++component)
         {
            const int grid = dim * history_order + component;
            const real_t prescribed = constant_datum ? datum[component] :
               datum[component + dim * (q + q1d * face)];
            speed += (prescribed - state[grid]) *
                     normal(q, component, face);
         }
         const real_t coefficient = upwind * fabs(speed) - speed;
         const real_t scale = weight[q] * det(q, face) * coefficient;
         for (int component = 0; component < dim; ++component)
         {
            real_t correction = 0.0;
            for (int history = 0; history < history_order; ++history)
            {
               const real_t prescribed = constant_datum ? datum[component] :
                  datum[component + dim * (q + q1d * face)];
               correction += beta[history] *
                  (state[dim * history + component] - prescribed);
            }
            flux[q][component] = scale * correction;
         }
      }
      for (int dof = 0; dof < d1d; ++dof)
      {
         for (int component = 0; component < dim; ++component)
         {
            real_t load = 0.0;
            for (int q = 0; q < q1d; ++q)
            {
               load += Bt(dof, q) * flux[q][component];
            }
            y(dof, component, 0, face) += load;
         }
      }
   });
}

static void PAALEConvectionBoundaryApply3D(
   const int history_order, const bool continuity_scratch,
   const real_t upwind, const int d1d, const int q1d, const int nf,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &determinants, const Vector &normals,
   const Vector &datum_values, const Vector &beta_weights,
   const Vector &x_, Vector &y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE boundary PA face size exceeds the configured limit");
   const int dim = 3;
   const int vdim = dim * (history_order + 1 + (continuity_scratch ? 1 : 0));
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, q1d, dim, nf);
   const bool constant_datum = datum_values.Size() == dim;
   auto datum = datum_values.Read();
   auto beta = beta_weights.Read();
   auto weight = weights.Read();
   auto x = Reshape(x_.Read(), d1d, d1d, vdim, 2, nf);
   auto y = Reshape(y_.ReadWrite(), d1d, d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D][ALE_FACE_MAX_Q1D][3];
      for (int q2 = 0; q2 < q1d; ++q2)
      {
         for (int q1 = 0; q1 < q1d; ++q1)
         {
            real_t state[15];
            for (int component = 0; component < vdim; ++component)
            {
               state[component] = 0.0;
               for (int d2 = 0; d2 < d1d; ++d2)
               {
                  for (int d1 = 0; d1 < d1d; ++d1)
                  {
                     state[component] += B(q1, d1) * B(q2, d2) *
                        x(d1, d2, component, 0, face);
                  }
               }
            }
            real_t speed = 0.0;
            for (int component = 0; component < dim; ++component)
            {
               const int grid = dim * history_order + component;
               const int point = q1 + q1d * (q2 + q1d * face);
               const real_t prescribed = constant_datum ? datum[component] :
                  datum[component + dim * point];
               speed += (prescribed - state[grid]) *
                        normal(q1, q2, component, face);
            }
            const real_t coefficient = upwind * fabs(speed) - speed;
            const real_t scale = weight[q1 + q2 * q1d] *
                                 det(q1, q2, face) * coefficient;
            for (int component = 0; component < dim; ++component)
            {
               real_t correction = 0.0;
               for (int history = 0; history < history_order; ++history)
               {
                  const int point = q1 + q1d * (q2 + q1d * face);
                  const real_t prescribed = constant_datum ? datum[component] :
                     datum[component + dim * point];
                  correction += beta[history] *
                     (state[dim * history + component] - prescribed);
               }
               flux[q1][q2][component] = scale * correction;
            }
         }
      }
      for (int d2 = 0; d2 < d1d; ++d2)
      {
         for (int d1 = 0; d1 < d1d; ++d1)
         {
            for (int component = 0; component < dim; ++component)
            {
               real_t load = 0.0;
               for (int q2 = 0; q2 < q1d; ++q2)
               {
                  for (int q1 = 0; q1 < q1d; ++q1)
                  {
                     load += Bt(d1, q1) * Bt(d2, q2) *
                             flux[q1][q2][component];
                  }
               }
               y(d1, d2, component, 0, face) += load;
            }
         }
      }
   });
}

static void PAALEPressureBoundaryApply2D(
   const int history_order, const bool continuity_scratch,
   const int d1d, const int q1d, const int nf, const int ne,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &determinants, const Vector &normals,
   const Vector &basis_, const Vector &derivative_,
   const Vector &inverse_jacobian_, const Array<int> &boundary_elements,
   const Vector &delta_weights, const Vector &face_x_,
   const Vector &element_x_, Vector &face_y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE pressure-boundary PA face size exceeds the configured limit");
   const int dim = 2;
   const int vdim = dim * (history_order + 1 + (continuity_scratch ? 1 : 0));
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, dim, nf);
   auto basis = Reshape(basis_.Read(), d1d, dim, q1d, nf);
   auto derivative = Reshape(derivative_.Read(), d1d, dim, q1d, nf);
   auto inverse_jacobian =
      Reshape(inverse_jacobian_.Read(), dim, dim, q1d, nf);
   auto elements = boundary_elements.Read();
   auto delta = delta_weights.Read();
   auto weight = weights.Read();
   auto face_x = Reshape(face_x_.Read(), d1d, vdim, 2, nf);
   auto element_x = Reshape(element_x_.Read(), d1d, d1d, vdim, ne);
   auto face_y = Reshape(face_y_.ReadWrite(), d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D];
      const int element = elements[face];
      for (int q = 0; q < q1d; ++q)
      {
         real_t state[15];
         for (int component = 0; component < vdim; ++component)
         {
            state[component] = 0.0;
            for (int dof = 0; dof < d1d; ++dof)
            {
               state[component] += B(q, dof) *
                                   face_x(dof, component, 0, face);
            }
         }
         real_t pressure = 0.0;
         for (int history = 0; history < history_order; ++history)
         {
            real_t acceleration[3] = {0.0, 0.0, 0.0};
            real_t relative[3] = {0.0, 0.0, 0.0};
            for (int direction = 0; direction < dim; ++direction)
            {
               relative[direction] =
                  state[dim * history + direction] -
                  state[dim * history_order + direction];
            }
            for (int component = 0; component < dim; ++component)
            {
               for (int physical = 0; physical < dim; ++physical)
               {
                  real_t gradient = 0.0;
                  for (int d2 = 0; d2 < d1d; ++d2)
                  {
                     for (int d1 = 0; d1 < d1d; ++d1)
                     {
                        const real_t reference0 =
                           derivative(d1, 0, q, face) * basis(d2, 1, q, face);
                        const real_t reference1 =
                           basis(d1, 0, q, face) * derivative(d2, 1, q, face);
                        const real_t physical_derivative =
                           reference0 * inverse_jacobian(0, physical, q, face) +
                           reference1 * inverse_jacobian(1, physical, q, face);
                        gradient += element_x(d1, d2,
                                              dim * history + component,
                                              element) * physical_derivative;
                     }
                  }
                  acceleration[component] += gradient * relative[physical];
               }
            }
            real_t normal_acceleration = 0.0;
            for (int component = 0; component < dim; ++component)
            {
               normal_acceleration +=
                  acceleration[component] * normal(q, component, face);
            }
            pressure += delta[history] * normal_acceleration;
         }
         flux[q] = weight[q] * det(q, face) * pressure;
      }
      for (int dof = 0; dof < d1d; ++dof)
      {
         real_t load = 0.0;
         for (int q = 0; q < q1d; ++q)
         {
            load += Bt(dof, q) * flux[q];
         }
         face_y(dof, dim, 0, face) += load;
      }
   });
}

static void PAALEPressureBoundaryApply3D(
   const int history_order, const bool continuity_scratch,
   const int d1d, const int q1d, const int nf, const int ne,
   const Array<real_t> &weights, const DofToQuad &maps,
   const Vector &determinants, const Vector &normals,
   const Vector &basis_, const Vector &derivative_,
   const Vector &inverse_jacobian_, const Array<int> &boundary_elements,
   const Vector &delta_weights, const Vector &face_x_,
   const Vector &element_x_, Vector &face_y_)
{
   MFEM_VERIFY(d1d <= ALE_FACE_MAX_D1D && q1d <= ALE_FACE_MAX_Q1D,
               "ALE pressure-boundary PA face size exceeds the configured limit");
   const int dim = 3;
   const int vdim = dim * (history_order + 1 + (continuity_scratch ? 1 : 0));
   const int nq = q1d * q1d;
   auto B = Reshape(maps.B.Read(), q1d, d1d);
   auto Bt = Reshape(maps.Bt.Read(), d1d, q1d);
   auto det = Reshape(determinants.Read(), q1d, q1d, nf);
   auto normal = Reshape(normals.Read(), q1d, q1d, dim, nf);
   auto basis = Reshape(basis_.Read(), d1d, dim, nq, nf);
   auto derivative = Reshape(derivative_.Read(), d1d, dim, nq, nf);
   auto inverse_jacobian =
      Reshape(inverse_jacobian_.Read(), dim, dim, nq, nf);
   auto elements = boundary_elements.Read();
   auto delta = delta_weights.Read();
   auto weight = weights.Read();
   auto face_x = Reshape(face_x_.Read(), d1d, d1d, vdim, 2, nf);
   auto element_x =
      Reshape(element_x_.Read(), d1d, d1d, d1d, vdim, ne);
   auto face_y = Reshape(face_y_.ReadWrite(), d1d, d1d, vdim, 2, nf);

   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int face)
   {
      real_t flux[ALE_FACE_MAX_Q1D][ALE_FACE_MAX_Q1D];
      const int element = elements[face];
      for (int q2 = 0; q2 < q1d; ++q2)
      {
         for (int q1 = 0; q1 < q1d; ++q1)
         {
            const int q = q1 + q1d * q2;
            real_t state[15];
            for (int component = 0; component < vdim; ++component)
            {
               state[component] = 0.0;
               for (int d2 = 0; d2 < d1d; ++d2)
               {
                  for (int d1 = 0; d1 < d1d; ++d1)
                  {
                     state[component] += B(q1, d1) * B(q2, d2) *
                        face_x(d1, d2, component, 0, face);
                  }
               }
            }
            real_t pressure = 0.0;
            for (int history = 0; history < history_order; ++history)
            {
               real_t acceleration[3] = {0.0, 0.0, 0.0};
               real_t relative[3] = {0.0, 0.0, 0.0};
               for (int direction = 0; direction < dim; ++direction)
               {
                  relative[direction] =
                     state[dim * history + direction] -
                     state[dim * history_order + direction];
               }
               for (int component = 0; component < dim; ++component)
               {
                  for (int physical = 0; physical < dim; ++physical)
                  {
                     real_t gradient = 0.0;
                     for (int d3 = 0; d3 < d1d; ++d3)
                     {
                        for (int d2 = 0; d2 < d1d; ++d2)
                        {
                           for (int d1 = 0; d1 < d1d; ++d1)
                           {
                              const real_t reference0 =
                                 derivative(d1, 0, q, face) *
                                 basis(d2, 1, q, face) *
                                 basis(d3, 2, q, face);
                              const real_t reference1 =
                                 basis(d1, 0, q, face) *
                                 derivative(d2, 1, q, face) *
                                 basis(d3, 2, q, face);
                              const real_t reference2 =
                                 basis(d1, 0, q, face) *
                                 basis(d2, 1, q, face) *
                                 derivative(d3, 2, q, face);
                              const real_t physical_derivative =
                                 reference0 * inverse_jacobian(0, physical, q, face) +
                                 reference1 * inverse_jacobian(1, physical, q, face) +
                                 reference2 * inverse_jacobian(2, physical, q, face);
                              gradient += element_x(
                                 d1, d2, d3, dim * history + component,
                                 element) * physical_derivative;
                           }
                        }
                     }
                     acceleration[component] += gradient * relative[physical];
                  }
               }
               real_t normal_acceleration = 0.0;
               for (int component = 0; component < dim; ++component)
               {
                  normal_acceleration += acceleration[component] *
                                         normal(q1, q2, component, face);
               }
               pressure += delta[history] * normal_acceleration;
            }
            flux[q1][q2] = weight[q] * det(q1, q2, face) * pressure;
         }
      }
      for (int d2 = 0; d2 < d1d; ++d2)
      {
         for (int d1 = 0; d1 < d1d; ++d1)
         {
            real_t load = 0.0;
            for (int q2 = 0; q2 < q1d; ++q2)
            {
               for (int q1 = 0; q1 < q1d; ++q1)
               {
                  load += Bt(d1, q1) * Bt(d2, q2) * flux[q1][q2];
               }
            }
            face_y(d1, d2, dim, 0, face) += load;
         }
      }
   });
}

ALEConvectionVolumeIntegrator::ALEConvectionVolumeIntegrator(
   int order, const Vector &beta_weights)
   : history_order(order), beta(&beta_weights)
{
   MFEM_VERIFY(history_order > 0 && history_order <= 3,
               "native ALE volume integrator supports BDF/EX order one to three");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE beta vector must match the history order");
}

void ALEConvectionVolumeIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   const int element_dim = el.GetDim();
   MFEM_VERIFY(element_dim == 2 || element_dim == 3,
               "ALEConvectionVolumeIntegrator supports 2D and 3D elements");
   const int vdim = element_dim * (history_order + 1);
   const int dof = el.GetDof();
   MFEM_VERIFY(elfun.Size() == vdim * dof,
               "packed ALE volume state has the wrong size");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE history weights changed size after construction");

   elvect.SetSize(elfun.Size());
   elvect = 0.0;
   shape.SetSize(dof);
   dshape.SetSize(dof, element_dim);
   const IntegrationRule *ir = IntRule ? IntRule :
      &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 2);

   for (int point = 0; point < ir->GetNPoints(); ++point)
   {
      const IntegrationPoint &ip = ir->IntPoint(point);
      Tr.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(Tr, dshape);
      real_t state[12] = {0.0};
      for (int field = 0; field < vdim; ++field)
      {
         const int offset = field * dof;
         for (int j = 0; j < dof; ++j)
         {
            state[field] += elfun(offset + j) * shape(j);
         }
      }

      const real_t scale = ip.weight * Tr.Weight();
      for (int history = 0; history < history_order; ++history)
      {
         for (int component = 0; component < element_dim; ++component)
         {
            real_t acceleration = 0.0;
            const int velocity = element_dim * history + component;
            const int velocity_offset = velocity * dof;
            for (int physical = 0; physical < element_dim; ++physical)
            {
               real_t gradient = 0.0;
               for (int j = 0; j < dof; ++j)
               {
                  gradient += elfun(velocity_offset + j) *
                              dshape(j, physical);
               }
               const real_t relative =
                  state[element_dim * history + physical] -
                  state[element_dim * history_order + physical];
               acceleration += relative * gradient;
            }
            const real_t load = scale * (*beta)(history) * acceleration;
            const int output_offset = component * dof;
            for (int j = 0; j < dof; ++j)
            {
               elvect(output_offset + j) += load * shape(j);
            }
         }
      }
   }
}

void ALEConvectionVolumeIntegrator::AssemblePA(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ALEConvectionVolumeIntegrator supports 2D and 3D meshes");
   MFEM_VERIFY(fes.GetVDim() == dim * (history_order + 1),
               "packed ALE volume space has the wrong vector dimension");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "packed ALE volume space requires byNODES ordering");
   MFEM_VERIFY(!fes.IsVariableOrder(),
               "ALE volume PA requires a uniform tensor-product space");
   const FiniteElement &element = *fes.GetTypicalFE();
   const IntegrationRule *ir = IntRule ? IntRule :
      &IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 2);
   if (!IntRule) { SetIntRule(ir); }
   const MemoryType memory =
      pa_mt == MemoryType::DEFAULT ? Device::GetDeviceMemoryType() : pa_mt;
   ne = fes.GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, memory);
   maps = &element.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   MFEM_VERIFY(nq == (dim == 2 ? quad1D * quad1D :
                     quad1D * quad1D * quad1D),
               "ALE volume PA requires a tensor-product integration rule");
}

void ALEConvectionVolumeIntegrator::AddMultPA(
   const Vector &x, Vector &y) const
{
   if (ne == 0) { return; }
   MFEM_VERIFY(maps && geom,
               "assemble the ALE volume PA kernel before applying it");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE history weights changed size after construction");
   const IntegrationRule &ir = *IntRule;
   if (dim == 2)
   {
      PAALEConvectionVolumeApply2D(
         history_order, dofs1D, quad1D, ne, ir.GetWeights(), *maps,
         geom->J, *beta, x, y);
   }
   else
   {
      PAALEConvectionVolumeApply3D(
         history_order, dofs1D, quad1D, ne, ir.GetWeights(), *maps,
         geom->J, *beta, x, y);
   }
}

ALEConvectionInteriorIntegrator::ALEConvectionInteriorIntegrator(
   int order, real_t upwind_factor, const Vector &beta_weights)
   : history_order(order),
     upwind(upwind_factor),
     beta(&beta_weights)
{
   MFEM_VERIFY(history_order > 0 && history_order <= 3,
               "native ALE interior integrator supports BDF/EX order one to three");
   MFEM_VERIFY(upwind >= 0.0, "ALE upwind factor must be non-negative");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE beta vector must match the history order");
}

void ALEConvectionInteriorIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   MFEM_VERIFY(Tr.Elem2No >= 0,
               "ALEConvectionInteriorIntegrator requires an interior face");
   const int dim = Tr.GetSpaceDim();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ALEConvectionInteriorIntegrator supports 2D and 3D meshes");
   const int vdim = dim * (history_order + 1);
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE history weights changed size after construction");

   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();
   const int offset2 = vdim * dof1;
   MFEM_VERIFY(elfun.Size() == vdim * (dof1 + dof2),
               "packed ALE interior state has the wrong size");
   elvect.SetSize(elfun.Size());
   elvect = 0.0;
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   normal.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &IntRules.Get(Tr.GetGeometryType(),
                         2 * std::max(el1.GetOrder(), el2.GetOrder()) + 2);
   }

   for (int point = 0; point < ir->GetNPoints(); point++)
   {
      const IntegrationPoint &face_ip = ir->IntPoint(point);
      Tr.SetAllIntPoints(&face_ip);
      const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &ip2 = Tr.GetElement2IntPoint();
      el1.CalcShape(ip1, shape1);
      el2.CalcShape(ip2, shape2);
      CalcOrtho(Tr.Jacobian(), normal);

      real_t velocity1[3][3] = {{0.0}};
      real_t velocity2[3][3] = {{0.0}};
      real_t grid1[3] = {0.0, 0.0, 0.0};
      real_t grid2[3] = {0.0, 0.0, 0.0};
      for (int history = 0; history < history_order; history++)
      {
         for (int component = 0; component < dim; component++)
         {
            const int component_offset1 = (dim * history + component) * dof1;
            const int component_offset2 = offset2 + (dim * history + component) * dof2;
            for (int j = 0; j < dof1; j++)
            {
               velocity1[history][component] +=
                  elfun(component_offset1 + j) * shape1(j);
            }
            for (int j = 0; j < dof2; j++)
            {
               velocity2[history][component] +=
                  elfun(component_offset2 + j) * shape2(j);
            }
         }
      }
      for (int component = 0; component < dim; component++)
      {
         const int component_offset1 = (dim * history_order + component) * dof1;
         const int component_offset2 = offset2 + (dim * history_order + component) * dof2;
         for (int j = 0; j < dof1; j++)
         {
            grid1[component] += elfun(component_offset1 + j) * shape1(j);
         }
         for (int j = 0; j < dof2; j++)
         {
            grid2[component] += elfun(component_offset2 + j) * shape2(j);
         }
      }

      real_t flux1[3] = {0.0, 0.0, 0.0};
      real_t flux2[3] = {0.0, 0.0, 0.0};
      for (int history = 0; history < history_order; history++)
      {
         real_t normal_speed = 0.0;
         for (int component = 0; component < dim; component++)
         {
            normal_speed +=
               0.5 * (velocity1[history][component] +
                      velocity2[history][component] - grid1[component] -
                      grid2[component]) * normal(component);
         }
         const real_t dissipation = upwind * std::abs(normal_speed);
         const real_t coefficient1 = 0.5 * (-normal_speed + dissipation);
         const real_t coefficient2 = 0.5 * (-normal_speed - dissipation);
         for (int component = 0; component < dim; component++)
         {
            const real_t jump =
               velocity1[history][component] - velocity2[history][component];
            flux1[component] += (*beta)(history) * coefficient1 * jump;
            flux2[component] += (*beta)(history) * coefficient2 * jump;
         }
      }

      const real_t weight = face_ip.weight;
      for (int component = 0; component < dim; component++)
      {
         const int component_offset1 = component * dof1;
         const int component_offset2 = offset2 + component * dof2;
         for (int j = 0; j < dof1; j++)
         {
            elvect(component_offset1 + j) +=
               weight * flux1[component] * shape1(j);
         }
         for (int j = 0; j < dof2; j++)
         {
            elvect(component_offset2 + j) +=
               weight * flux2[component] * shape2(j);
         }
      }
   }
}

void ALEConvectionInteriorIntegrator::AssemblePAInteriorFaces(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ALEConvectionInteriorIntegrator supports 2D and 3D meshes");
   MFEM_VERIFY(fes.GetVDim() == dim * (history_order + 1),
               "packed ALE interior space has the wrong vector dimension");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "packed ALE interior space requires byNODES ordering");
   const FiniteElement &trace = *fes.GetTypicalTraceElement();
   const IntegrationRule *ir = IntRule ? IntRule :
      &IntRules.Get(trace.GetGeomType(), 2 * trace.GetOrder() + 2);
   if (!IntRule) { SetIntRule(ir); }
   FaceQuadratureSpace quadrature(*mesh, *ir, FaceType::Interior);
   nf = quadrature.GetNumFaces();
   if (nf == 0) { return; }
   const MemoryType memory =
      pa_mt == MemoryType::DEFAULT ? Device::GetDeviceMemoryType() : pa_mt;
   geom = mesh->GetFaceGeometricFactors(
             *ir,
             FaceGeometricFactors::DETERMINANTS |
             FaceGeometricFactors::NORMALS,
             FaceType::Interior,
             memory);
   maps = &trace.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   MFEM_VERIFY(ir->GetNPoints() == (dim == 2 ? quad1D : quad1D * quad1D),
               "ALE interior PA requires a tensor-product face rule");
}

void ALEConvectionInteriorIntegrator::AddMultPA(
   const Vector &x, Vector &y) const
{
   if (nf == 0) { return; }
   MFEM_VERIFY(maps && geom,
               "assemble the ALE interior PA kernel before applying it");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE history weights changed size after construction");
   const IntegrationRule &ir = *IntRule;
   if (dim == 2)
   {
      PAALEConvectionInteriorApply2D(
         history_order, upwind, dofs1D, quad1D, nf, ir.GetWeights(),
         *maps, geom->detJ, geom->normal, *beta, x, y);
   }
   else
   {
      PAALEConvectionInteriorApply3D(
         history_order, upwind, dofs1D, quad1D, nf, ir.GetWeights(),
         *maps, geom->detJ, geom->normal, *beta, x, y);
   }
}

ALEConvectionBoundaryIntegrator::ALEConvectionBoundaryIntegrator(
   int order, real_t upwind_factor, const Vector &beta_weights,
   const Vector &delta_weights, bool convection, bool pressure_delta,
   bool continuity_enabled)
   : history_order(order),
     upwind(upwind_factor),
     beta(&beta_weights),
     delta(&delta_weights),
     include_convection(convection),
     include_pressure_delta(pressure_delta),
     include_continuity_scratch(continuity_enabled)
{
   MFEM_VERIFY(history_order > 0 && history_order <= 3,
               "native ALE boundary integrator supports BDF/EX order one to three");
   MFEM_VERIFY(upwind >= 0.0, "ALE upwind factor must be non-negative");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE beta vector must match the history order");
   MFEM_VERIFY(delta->Size() == history_order,
               "ALE delta vector must match the history order");
   MFEM_VERIFY(include_convection || include_pressure_delta,
               "ALE boundary integrator has no enabled output");
}

void ALEConvectionBoundaryIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   MFEM_VERIFY(Tr.Elem2No < 0,
               "ALEConvectionBoundaryIntegrator requires a boundary face");
   const int dim = Tr.GetSpaceDim();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ALEConvectionBoundaryIntegrator supports 2D and 3D meshes");
   const int vdim = dim * (history_order + 1 +
                           (include_continuity_scratch ? 1 : 0));
   MFEM_VERIFY(!include_convection || datum,
               "Dirichlet ALE convection requires a boundary datum");
   MFEM_VERIFY(!datum || datum->GetVDim() == dim,
               "ALE boundary datum dimension must match the mesh");
   MFEM_VERIFY(beta->Size() == history_order && delta->Size() == history_order,
               "ALE history weights changed size after construction");

   const int dof = el1.GetDof();
   MFEM_VERIFY(elfun.Size() == vdim * dof,
               "packed ALE boundary state has the wrong size");
   elvect.SetSize(vdim * dof);
   elvect = 0.0;
   shape.SetSize(dof);
   normal.SetSize(dim);
   if (include_pressure_delta) { dshape.SetSize(dof, dim); }

   const IntegrationRule *ir = IntRule;
   if (!ir)
   {
      ir = &IntRules.Get(Tr.GetGeometryType(), 2 * el1.GetOrder() + 2);
   }

   for (int point = 0; point < ir->GetNPoints(); point++)
   {
      const IntegrationPoint &face_ip = ir->IntPoint(point);
      Tr.SetAllIntPoints(&face_ip);
      const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
      el1.CalcShape(ip1, shape);
      CalcOrtho(Tr.Jacobian(), normal);

      real_t velocity[3][3] = {{0.0}};
      real_t grid[3] = {0.0, 0.0, 0.0};
      for (int history = 0; history < history_order; history++)
      {
         for (int component = 0; component < dim; component++)
         {
            const int offset = (dim * history + component) * dof;
            for (int j = 0; j < dof; j++)
            {
               velocity[history][component] += elfun(offset + j) * shape(j);
            }
         }
      }
      for (int component = 0; component < dim; component++)
      {
         const int offset = (dim * history_order + component) * dof;
         for (int j = 0; j < dof; j++)
         {
            grid[component] += elfun(offset + j) * shape(j);
         }
      }

      const real_t weight = face_ip.weight;
      if (include_convection)
      {
         datum->Eval(datum_value, *Tr.Elem1, ip1);
         real_t normal_speed = 0.0;
         for (int component = 0; component < dim; component++)
         {
            normal_speed +=
               (datum_value(component) - grid[component]) * normal(component);
         }
         const real_t coefficient =
            upwind * std::abs(normal_speed) - normal_speed;
         for (int component = 0; component < dim; component++)
         {
            real_t correction = 0.0;
            for (int history = 0; history < history_order; history++)
            {
               correction += (*beta)(history) *
                             (velocity[history][component] - datum_value(component));
            }
            correction *= coefficient * weight;
            const int offset = component * dof;
            for (int j = 0; j < dof; j++)
            {
               elvect(offset + j) += correction * shape(j);
            }
         }
      }

      if (include_pressure_delta)
      {
         Tr.Elem1->SetIntPoint(&ip1);
         el1.CalcPhysDShape(*Tr.Elem1, dshape);
         real_t pressure_load = 0.0;
         for (int history = 0; history < history_order; history++)
         {
            real_t acceleration[3] = {0.0, 0.0, 0.0};
            real_t relative[3] = {0.0, 0.0, 0.0};
            for (int component = 0; component < dim; component++)
            {
               relative[component] =
                  velocity[history][component] - grid[component];
            }
            for (int component = 0; component < dim; component++)
            {
               for (int direction = 0; direction < dim; direction++)
               {
                  real_t gradient = 0.0;
                  const int offset = (dim * history + component) * dof;
                  for (int j = 0; j < dof; j++)
                  {
                     gradient += elfun(offset + j) * dshape(j, direction);
                  }
                  acceleration[component] += gradient * relative[direction];
               }
            }
            real_t normal_acceleration = 0.0;
            for (int component = 0; component < dim; component++)
            {
               normal_acceleration += acceleration[component] * normal(component);
            }
            pressure_load += (*delta)(history) * normal_acceleration;
         }
         pressure_load *= weight;
         const int offset = dim * dof;
         for (int j = 0; j < dof; j++)
         {
            elvect(offset + j) += pressure_load * shape(j);
         }
      }
   }
}

void ALEConvectionBoundaryIntegrator::AssemblePABoundaryFaces(
   const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "ALEConvectionBoundaryIntegrator supports 2D and 3D meshes");
   const int expected_vdim = dim *
      (history_order + 1 + (include_continuity_scratch ? 1 : 0));
   MFEM_VERIFY(fes.GetVDim() == expected_vdim,
               "packed ALE boundary space has the wrong vector dimension");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "packed ALE boundary space requires byNODES ordering");
   MFEM_VERIFY(!include_convection || datum,
               "Dirichlet ALE convection requires a boundary datum");

   const FiniteElement &trace = *fes.GetTypicalTraceElement();
   const IntegrationRule *ir = IntRule ? IntRule :
      &IntRules.Get(trace.GetGeomType(), 2 * trace.GetOrder() + 2);
   if (!IntRule) { SetIntRule(ir); }
   FaceQuadratureSpace quadrature(*mesh, *ir, FaceType::Boundary);
   nf = quadrature.GetNumFaces();
   nq = ir->GetNPoints();
   ne = fes.GetNE();
   element_dofs = fes.GetTypicalFE()->GetDof();
   if (nf == 0) { return; }
   const MemoryType memory =
      pa_mt == MemoryType::DEFAULT ? Device::GetDeviceMemoryType() : pa_mt;
   geom = mesh->GetFaceGeometricFactors(
             *ir,
             FaceGeometricFactors::DETERMINANTS |
             FaceGeometricFactors::NORMALS,
             FaceType::Boundary,
             memory);
   maps = &trace.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   MFEM_VERIFY(ir->GetNPoints() == (dim == 2 ? quad1D : quad1D * quad1D),
               "ALE boundary PA requires a tensor-product face rule");

   if (include_convection)
   {
      CoefficientVector sampled_datum(
         *datum, quadrature, CoefficientStorage::COMPRESSED);
      pa_datum.SetSize(sampled_datum.Size(), memory);
      pa_datum = sampled_datum;
   }

   if (include_pressure_delta)
   {
      MFEM_VERIFY(element_dofs ==
                  (dim == 2 ? dofs1D * dofs1D :
                   dofs1D * dofs1D * dofs1D),
                  "ALE pressure-boundary PA requires tensor-product elements");
      pa_boundary_elements.SetSize(nf, memory);
      pa_basis.SetSize(dofs1D * dim * nq * nf, memory);
      pa_derivative.SetSize(dofs1D * dim * nq * nf, memory);
      pa_inverse_jacobian.SetSize(dim * dim * nq * nf, memory);

      auto boundary_elements = pa_boundary_elements.HostWrite();
      auto basis_data = Reshape(pa_basis.HostWrite(), dofs1D, dim, nq, nf);
      auto derivative_data =
         Reshape(pa_derivative.HostWrite(), dofs1D, dim, nq, nf);
      auto inverse_data =
         Reshape(pa_inverse_jacobian.HostWrite(), dim, dim, nq, nf);
      Poly_1D::Basis &basis1d =
         poly1d.GetBasis(dofs1D - 1, BasisType::GaussLobatto);
      Vector values(dofs1D), derivatives(dofs1D);

      for (int face = 0; face < nf; ++face)
      {
         const int mesh_face = quadrature.GetMeshFaceIndex(face);
         const Mesh::FaceInformation information =
            mesh->GetFaceInformation(mesh_face);
         boundary_elements[face] = information.element[0].index;
         FaceElementTransformations *transformation =
            mesh->GetFaceElementTransformations(mesh_face);
         MFEM_VERIFY(transformation && transformation->Elem2No < 0,
                     "ALE boundary quadrature contains a non-boundary face");
         for (int point = 0; point < nq; ++point)
         {
            const int lex_point = ToLexOrdering(
               dim, information.element[0].local_face_id, quad1D, point);
            const IntegrationPoint &face_ip = ir->IntPoint(point);
            transformation->SetAllIntPoints(&face_ip);
            const IntegrationPoint &element_ip =
               transformation->GetElement1IntPoint();
            const real_t coordinate[3] =
            {
               element_ip.x, element_ip.y, element_ip.z
            };
            for (int direction = 0; direction < dim; ++direction)
            {
               basis1d.Eval(coordinate[direction], values, derivatives);
               for (int dof = 0; dof < dofs1D; ++dof)
               {
                  basis_data(dof, direction, lex_point, face) = values(dof);
                  derivative_data(dof, direction, lex_point, face) =
                     derivatives(dof);
               }
            }
            transformation->Elem1->SetIntPoint(&element_ip);
            const DenseMatrix &inverse =
               transformation->Elem1->InverseJacobian();
            for (int reference = 0; reference < dim; ++reference)
            {
               for (int physical = 0; physical < dim; ++physical)
               {
                  inverse_data(reference, physical, lex_point, face) =
                     inverse(reference, physical);
               }
            }
         }
      }
   }
}

void ALEConvectionBoundaryIntegrator::AddMultPA(
   const Vector &x, Vector &y) const
{
   if (nf == 0 || !include_convection) { return; }
   MFEM_VERIFY(maps && geom,
               "assemble the ALE boundary PA kernel before applying it");
   MFEM_VERIFY(beta->Size() == history_order,
               "ALE history weights changed size after construction");
   const IntegrationRule &ir = *IntRule;
   if (dim == 2)
   {
      PAALEConvectionBoundaryApply2D(
         history_order, include_continuity_scratch, upwind,
         dofs1D, quad1D, nf, ir.GetWeights(), *maps,
         geom->detJ, geom->normal, pa_datum, *beta, x, y);
   }
   else
   {
      PAALEConvectionBoundaryApply3D(
         history_order, include_continuity_scratch, upwind,
         dofs1D, quad1D, nf, ir.GetWeights(), *maps,
         geom->detJ, geom->normal, pa_datum, *beta, x, y);
   }
}

void ALEConvectionBoundaryIntegrator::AddMultPAFace(
   const Vector &face_x, const Vector &element_x, Vector &face_y) const
{
   AddMultPA(face_x, face_y);
   if (nf == 0 || !include_pressure_delta) { return; }
   MFEM_VERIFY(delta->Size() == history_order,
               "ALE pressure weights changed size after construction");
   MFEM_VERIFY(pa_basis.Size() && pa_derivative.Size() &&
               pa_inverse_jacobian.Size(),
               "assemble the ALE pressure-boundary PA kernel before applying it");
   const IntegrationRule &ir = *IntRule;
   if (dim == 2)
   {
      PAALEPressureBoundaryApply2D(
         history_order, include_continuity_scratch,
         dofs1D, quad1D, nf, ne, ir.GetWeights(), *maps,
         geom->detJ, geom->normal, pa_basis, pa_derivative,
         pa_inverse_jacobian, pa_boundary_elements, *delta,
         face_x, element_x, face_y);
   }
   else
   {
      PAALEPressureBoundaryApply3D(
         history_order, include_continuity_scratch,
         dofs1D, quad1D, nf, ne, ir.GetWeights(), *maps,
         geom->detJ, geom->normal, pa_basis, pa_derivative,
         pa_inverse_jacobian, pa_boundary_elements, *delta,
         face_x, element_x, face_y);
   }
}

real_t NonlinearFormIntegrator::GetLocalStateEnergyPA(const Vector &x) const
{
   mfem_error ("NonlinearFormIntegrator::GetLocalStateEnergyPA(...)\n"
               "   is not implemented for this class.");
   return 0.0;
}

void NonlinearFormIntegrator::AssemblePA(const FiniteElementSpace&)
{
   mfem_error ("NonlinearFormIntegrator::AssemblePA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssemblePA(const FiniteElementSpace &,
                                         const FiniteElementSpace &)
{
   mfem_error ("NonlinearFormIntegrator::AssemblePA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssemblePAInteriorFaces(
   const FiniteElementSpace &)
{
   mfem_error("NonlinearFormIntegrator::AssemblePAInteriorFaces(...)\n"
              "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssemblePABoundaryFaces(
   const FiniteElementSpace &)
{
   mfem_error("NonlinearFormIntegrator::AssemblePABoundaryFaces(...)\n"
              "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssembleGradPA(const Vector &x,
                                             const FiniteElementSpace &fes)
{
   mfem_error ("NonlinearFormIntegrator::AssembleGradPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultPA(const Vector &, Vector &) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultPAFace(
   const Vector &face_x, const Vector &, Vector &face_y) const
{
   AddMultPA(face_x, face_y);
}

void NonlinearFormIntegrator::AddMultGradPA(const Vector&, Vector&) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultGradPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssembleGradDiagonalPA(Vector &diag) const
{
   mfem_error ("NonlinearFormIntegrator::AssembleGradDiagonalPA(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssembleMF(const FiniteElementSpace &fes)
{
   mfem_error ("NonlinearFormIntegrator::AssembleMF(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AddMultMF(const Vector &, Vector &) const
{
   mfem_error ("NonlinearFormIntegrator::AddMultMF(...)\n"
               "   is not implemented for this class.");
}

void NonlinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void NonlinearFormIntegrator::AssembleFaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   mfem_error("NonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

real_t NonlinearFormIntegrator::GetElementEnergy(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun)
{
   mfem_error("NonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


void BlockNonlinearFormIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceVector"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleElementGrad"
              " is not overloaded!");
}

void BlockNonlinearFormIntegrator::AssembleFaceGrad(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   mfem_error("BlockNonlinearFormIntegrator::AssembleFaceGrad"
              " is not overloaded!");
}

real_t BlockNonlinearFormIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   mfem_error("BlockNonlinearFormIntegrator::GetElementEnergy"
              " is not overloaded!");
   return 0.0;
}


real_t InverseHarmonicModel::EvalW(const DenseMatrix &J) const
{
   Z.SetSize(J.Width());
   CalcAdjugateTranspose(J, Z);
   return 0.5*(Z*Z)/J.Det();
}

void InverseHarmonicModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();
   real_t t;

   Z.SetSize(dim);
   S.SetSize(dim);
   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);
   t = 0.5*S.Trace();
   for (int i = 0; i < dim; i++)
   {
      S(i,i) -= t;
   }
   t = J.Det();
   S *= -1.0/(t*t);
   Mult(S, Z, P);
}

void InverseHarmonicModel::AssembleH(
   const DenseMatrix &J, const DenseMatrix &DS, const real_t weight,
   DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();
   real_t t;

   Z.SetSize(dim);
   S.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   CalcAdjugateTranspose(J, Z);
   MultAAt(Z, S);

   t = 1.0/J.Det();
   Z *= t;  // Z = J^{-t}
   S *= t;  // S = |J| (J.J^t)^{-1}
   t = 0.5*S.Trace();

   MultABt(DS, Z, G);  // G = DS.J^{-1}
   Mult(G, S, C);

   // 1.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j <= i; j++)
      {
         real_t a = 0.0;
         for (int d = 0; d < dim; d++)
         {
            a += G(i,d)*G(j,d);
         }
         a *= weight;
         for (int k = 0; k < dim; k++)
            for (int l = 0; l <= k; l++)
            {
               real_t b = a*S(k,l);
               A(i+k*dof,j+l*dof) += b;
               if (i != j)
               {
                  A(j+k*dof,i+l*dof) += b;
               }
               if (k != l)
               {
                  A(i+l*dof,j+k*dof) += b;
                  if (i != j)
                  {
                     A(j+l*dof,i+k*dof) += b;
                  }
               }
            }
      }

   // 2.
   for (int i = 1; i < dof; i++)
      for (int j = 0; j < i; j++)
      {
         for (int k = 1; k < dim; k++)
            for (int l = 0; l < k; l++)
            {
               real_t a =
                  weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                          C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                          t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));

               A(i+k*dof,j+l*dof) += a;
               A(j+l*dof,i+k*dof) += a;

               A(i+l*dof,j+k*dof) -= a;
               A(j+k*dof,i+l*dof) -= a;
            }
      }
}


inline void NeoHookeanModel::EvalCoeffs() const
{
   mu = c_mu->Eval(*Ttr, Ttr->GetIntPoint());
   K = c_K->Eval(*Ttr, Ttr->GetIntPoint());
   if (c_g)
   {
      g = c_g->Eval(*Ttr, Ttr->GetIntPoint());
   }
}

real_t NeoHookeanModel::EvalW(const DenseMatrix &J) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   real_t dJ = J.Det();
   real_t sJ = dJ/g;
   real_t bI1 = pow(dJ, -2.0/dim)*(J*J); // \bar{I}_1

   return 0.5*(mu*(bI1 - dim) + K*(sJ - 1.0)*(sJ - 1.0));
}

void NeoHookeanModel::EvalP(const DenseMatrix &J, DenseMatrix &P) const
{
   int dim = J.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   CalcAdjugateTranspose(J, Z);

   real_t dJ = J.Det();
   real_t a  = mu*pow(dJ, -2.0/dim);
   real_t b  = K*(dJ/g - 1.0)/g - a*(J*J)/(dim*dJ);

   P = 0.0;
   P.Add(a, J);
   P.Add(b, Z);
}

void NeoHookeanModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                const real_t weight, DenseMatrix &A) const
{
   int dof = DS.Height(), dim = DS.Width();

   if (have_coeffs)
   {
      EvalCoeffs();
   }

   Z.SetSize(dim);
   G.SetSize(dof, dim);
   C.SetSize(dof, dim);

   real_t dJ = J.Det();
   real_t sJ = dJ/g;
   real_t a  = mu*pow(dJ, -2.0/dim);
   real_t bc = a*(J*J)/dim;
   real_t b  = bc - K*sJ*(sJ - 1.0);
   real_t c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);

   CalcAdjugateTranspose(J, Z);
   Z *= (1.0/dJ); // Z = J^{-t}

   MultABt(DS, J, C); // C = DS J^t
   MultABt(DS, Z, G); // G = DS J^{-1}

   a *= weight;
   b *= weight;
   c *= weight;

   // 1.
   for (int i = 0; i < dof; i++)
      for (int k = 0; k <= i; k++)
      {
         real_t s = 0.0;
         for (int d = 0; d < dim; d++)
         {
            s += DS(i,d)*DS(k,d);
         }
         s *= a;

         for (int d = 0; d < dim; d++)
         {
            A(i+d*dof,k+d*dof) += s;
         }

         if (k != i)
            for (int d = 0; d < dim; d++)
            {
               A(k+d*dof,i+d*dof) += s;
            }
      }

   a *= (-2.0/dim);

   // 2.
   for (int i = 0; i < dof; i++)
      for (int j = 0; j < dim; j++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dim; l++)
            {
               A(i+j*dof,k+l*dof) +=
                  a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                  b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
            }
}

const IntegrationRule* HyperelasticNLFIntegrator::GetDefaultIntegrationRule(
   const FiniteElement& trial_fe, const FiniteElement& test_fe,
   const ElementTransformation& trans) const
{
   return &(IntRules.Get(test_fe.GetGeomType(), 2*test_fe.GetOrder() + 3));
}

real_t HyperelasticNLFIntegrator::GetElementEnergy(const FiniteElement &el,
                                                   ElementTransformation &Ttr,
                                                   const Vector &elfun)
{
   int dof = el.GetDof(), dim = el.GetDim();
   real_t energy;

   DSh.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpr.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);

   const IntegrationRule *ir = GetIntegrationRule(el, Ttr);

   energy = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      MultAtB(PMatI, DSh, Jpr);
      Mult(Jpr, Jrt, Jpt);

      energy += ip.weight * Ttr.Weight() * model->EvalW(Jpt);
   }

   return energy;
}

void HyperelasticNLFIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = GetIntegrationRule(el, Ttr);
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elvect = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->EvalP(Jpt, P);

      P *= ip.weight * Ttr.Weight();
      AddMultABt(DS, P, PMatO);
   }
}

void HyperelasticNLFIntegrator::AssembleElementGrad(const FiniteElement &el,
                                                    ElementTransformation &Ttr,
                                                    const Vector &elfun,
                                                    DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elmat.SetSize(dof*dim);

   const IntegrationRule *ir = GetIntegrationRule(el, Ttr);
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
   }

   elmat = 0.0;
   model->SetTransformation(Ttr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
   }
}

real_t IncompressibleNeoHookeanIntegrator::GetElementEnergy(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::GetElementEnergy"
                 " has incorrect block finite element space size!");
   }

   int dof_u = el[0]->GetDof();
   int dim = el[0]->GetDim();

   DSh_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   J1.SetSize(dim);
   J.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   real_t energy = 0.0;
   real_t mu = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      MultAtB(PMatI_u, DSh_u, J1);
      Mult(J1, J0i, J);

      mu = c_mu->Eval(Tr, ip);

      energy += ip.weight*Tr.Weight()*(mu/2.0)*(J*J - 3);
   }

   return energy;
}

void IncompressibleNeoHookeanIntegrator::AssembleElementVector(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array<Vector *> &elvec)
{
   if (el.Size() != 2)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " has finite element space of incorrect block number");
   }

   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();
   int spaceDim = Tr.GetSpaceDim();

   if (dim != spaceDim)
   {
      mfem_error("IncompressibleNeoHookeanIntegrator::AssembleElementVector"
                 " is not defined on manifold meshes");
   }


   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   elvec[0]->SetSize(dof_u*dim);
   PMatO_u.UseExternalData(elvec[0]->GetData(), dof_u, dim);

   Sh_p.SetSize(dof_p);
   elvec[1]->SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   *elvec[0] = 0.0;
   *elvec[1] = 0.0;

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);

      real_t pres = Sh_p * *elfun[1];
      real_t mu = c_mu->Eval(Tr, ip);
      real_t dJ = F.Det();

      CalcInverseTranspose(F, FinvT);

      P = 0.0;
      P.Add(mu * dJ, F);
      P.Add(-1.0 * pres * dJ, FinvT);
      P *= ip.weight*Tr.Weight();

      AddMultABt(DS_u, P, PMatO_u);

      elvec[1]->Add(ip.weight * Tr.Weight() * (dJ - 1.0), Sh_p);
   }

}

void IncompressibleNeoHookeanIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   int dof_u = el[0]->GetDof();
   int dof_p = el[1]->GetDof();

   int dim = el[0]->GetDim();

   elmats(0,0)->SetSize(dof_u*dim, dof_u*dim);
   elmats(0,1)->SetSize(dof_u*dim, dof_p);
   elmats(1,0)->SetSize(dof_p, dof_u*dim);
   elmats(1,1)->SetSize(dof_p, dof_p);

   *elmats(0,0) = 0.0;
   *elmats(0,1) = 0.0;
   *elmats(1,0) = 0.0;
   *elmats(1,1) = 0.0;

   DSh_u.SetSize(dof_u, dim);
   DS_u.SetSize(dof_u, dim);
   J0i.SetSize(dim);
   F.SetSize(dim);
   FinvT.SetSize(dim);
   Finv.SetSize(dim);
   P.SetSize(dim);
   PMatI_u.UseExternalData(elfun[0]->GetData(), dof_u, dim);
   Sh_p.SetSize(dof_p);

   int intorder = 2*el[0]->GetOrder() + 3; // <---
   const IntegrationRule &ir = IntRules.Get(el[0]->GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcInverse(Tr.Jacobian(), J0i);

      el[0]->CalcDShape(ip, DSh_u);
      Mult(DSh_u, J0i, DS_u);
      MultAtB(PMatI_u, DS_u, F);

      el[1]->CalcShape(ip, Sh_p);
      real_t pres = Sh_p * *elfun[1];
      real_t mu = c_mu->Eval(Tr, ip);
      real_t dJ = F.Det();
      real_t dJ_FinvT_DS;

      CalcInverseTranspose(F, FinvT);

      // u,u block
      for (int i_u = 0; i_u < dof_u; ++i_u)
      {
         for (int i_dim = 0; i_dim < dim; ++i_dim)
         {
            for (int j_u = 0; j_u < dof_u; ++j_u)
            {
               for (int j_dim = 0; j_dim < dim; ++j_dim)
               {

                  // m = j_dim;
                  // k = i_dim;

                  for (int n=0; n<dim; ++n)
                  {
                     for (int l=0; l<dim; ++l)
                     {
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * (mu * F(i_dim, l) - pres * FinvT(i_dim,l)) *
                           FinvT(j_dim,n) * DS_u(i_u,l) * DS_u(j_u, n) *
                           ip.weight * Tr.Weight();

                        if (j_dim == i_dim && n==l)
                        {
                           (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                              dJ * mu * DS_u(i_u, l) * DS_u(j_u,n) *
                              ip.weight * Tr.Weight();
                        }

                        // a = n;
                        // b = m;
                        (*elmats(0,0))(i_u + i_dim*dof_u, j_u + j_dim*dof_u) +=
                           dJ * pres * FinvT(i_dim, n) *
                           FinvT(j_dim,l) * DS_u(i_u,l) * DS_u(j_u,n) *
                           ip.weight * Tr.Weight();
                     }
                  }
               }
            }
         }
      }

      // u,p and p,u blocks
      for (int i_p = 0; i_p < dof_p; ++i_p)
      {
         for (int j_u = 0; j_u < dof_u; ++j_u)
         {
            for (int dim_u = 0; dim_u < dim; ++dim_u)
            {
               for (int l=0; l<dim; ++l)
               {
                  dJ_FinvT_DS = dJ * FinvT(dim_u,l) * DS_u(j_u, l) * Sh_p(i_p) *
                                ip.weight * Tr.Weight();
                  (*elmats(1,0))(i_p, j_u + dof_u * dim_u) += dJ_FinvT_DS;
                  (*elmats(0,1))(j_u + dof_u * dim_u, i_p) -= dJ_FinvT_DS;

               }
            }
         }
      }
   }

}

const IntegrationRule&
VectorConvectionNLFIntegrator::GetRule(const FiniteElement &fe,
                                       const ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void VectorConvectionNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &T,
   const Vector &elfun,
   Vector &elvect)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim);
   const IntegrationRule *ir = GetIntegrationRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      real_t w = ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }

      MultAtB(EF, dshape, gradEF);
      EF.MultTranspose(shape, vec1);
      gradEF.Mult(vec1, vec2);
      vec2 *= w;
      AddMultVWt(shape, vec2, ELV);
   }
}

void VectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   real_t w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = GetIntegrationRule(el, trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1);

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3);
      MultVWt(shape, vec3, elmat_comp);

      for (int ii = 0; ii < dim; ii++)
      {
         elmat.AddMatrix(elmat_comp, ii * nd, ii * nd);
      }

      MultVVt(shape, elmat_comp);
      w = ip.weight * trans.Weight();
      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }
      for (int ii = 0; ii < dim; ii++)
      {
         for (int jj = 0; jj < dim; jj++)
         {
            elmat.AddMatrix(w * gradEF(ii, jj), elmat_comp, ii * nd, jj * nd);
         }
      }
   }
}


void ConvectiveVectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = GetIntegrationRule(el, trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      const real_t w = Q ? Q->Eval(trans, ip) * ip.weight : ip.weight;

      EF.MultTranspose(shape, vec1); // u^n

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3); // (u^n \cdot grad u^{n+1})
      MultVWt(shape, vec3, elmat_comp); // (u^n \cdot grad u^{n+1},v)

      for (int ii = 0; ii < dim; ii++)
      {
         elmat.AddMatrix(elmat_comp, ii * nd, ii * nd);
      }
   }
}


void SkewSymmetricVectorConvectionNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   gradEF.SetSize(dim);

   DenseMatrix elmat_comp_T(nd);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim), vec3(nd), vec4(dim), vec5(nd);

   const IntegrationRule *ir = GetIntegrationRule(el, trans);

   elmat = 0.0;
   elmat_comp_T = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      const real_t w = Q ? Q->Eval(trans, ip) * ip.weight : ip.weight;

      EF.MultTranspose(shape, vec1); // u^n

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3); // (u^n \cdot grad u^{n+1})
      MultVWt(shape, vec3, elmat_comp); // (u^n \cdot grad u^{n+1},v)
      elmat_comp_T.Transpose(elmat_comp);

      for (int ii = 0; ii < dim; ii++)
      {
         elmat.AddMatrix(.5, elmat_comp, ii * nd, ii * nd);
         elmat.AddMatrix(-.5, elmat_comp_T, ii * nd, ii * nd);
      }
   }
}

}
