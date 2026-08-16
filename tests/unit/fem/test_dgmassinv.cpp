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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

TEST_CASE("DG Mass Inverse", "[GPU]")
{
   auto mesh_filename = GENERATE(
                           "../../data/inline-segment.mesh",
                           "../../data/star.mesh",
                           "../../data/star-q3.mesh",
                           "../../data/fichera.mesh"
                        );
   auto order = GENERATE(2, 3, 4, 5);
   auto btype1 = GENERATE(BasisType::GaussLobatto, BasisType::GaussLegendre,
                          BasisType::Positive);

   // Skip Bernstein for order > 1, too ill-conditioned
   if (btype1 == BasisType::Positive && order > 2) { return; }

   CAPTURE(mesh_filename, order, btype1);

   Mesh mesh = Mesh::LoadFromFile(mesh_filename);
   DG_FECollection fec(order, mesh.Dimension(), btype1);
   FiniteElementSpace fes(&mesh, &fec);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   m.Assemble();

   Array<int> empty;
   OperatorJacobiSmoother jacobi(m, empty);

   int n = fes.GetTrueVSize();
   Vector B(n), X1(n), X2(n), X3(n);
   B.Randomize(1);
   X1.Randomize(2);
   X2 = X1;
   X3 = X1;

   const double tol = 1e-8;

   CGSolver cg;
   cg.SetAbsTol(tol);
   cg.SetRelTol(0.0);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().None());
   cg.SetOperator(m);
   cg.SetPreconditioner(jacobi);
   cg.Mult(B, X1);

   INFO("Global CG iterations: " << cg.GetNumIterations());

   SECTION("Local CG")
   {
      auto btype2 = GENERATE(BasisType::GaussLobatto, BasisType::GaussLegendre);
      CAPTURE(btype2);

      DGMassInverse m_inv(fes, btype2);
      m_inv.SetAbsTol(tol);
      m_inv.SetRelTol(0.0);
      m_inv.Mult(B, X2);

      DGMassInversePreconditioner preconditioner(m_inv);
      preconditioner.SetOperator(m);
      preconditioner.Mult(B, X3);
      X3 -= X2;
      REQUIRE(X3.Normlinf() == MFEM_Approx(0.0, 1e2*tol, 1e2*tol));

      X2 -= X1;
      REQUIRE(X2.Normlinf() == MFEM_Approx(0.0, 1e2*tol, 1e2*tol));
   }
}

TEST_CASE("Vector DG Mass Inverse is componentwise",
          "[GPU][DGMassInverseVector]")
{
   constexpr int order = 3;
   constexpr int vdim = 2;
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   // Exercise the square identity/direct path, a square but noncollocated
   // fallback, and a nonsquare generic fallback.
   const int rule_kind = GENERATE(0, 1, 2);
   CAPTURE(ordering, rule_kind);

   Mesh mesh = Mesh::MakeCartesian2D(
                  3, 2, Element::QUADRILATERAL, true, 1.0, 1.0);
   mesh.SetCurvature(order, false, 2, Ordering::byNODES);
   mesh.Transform([](const Vector &xold, Vector &xnew)
   {
      xnew = xold;
      xnew(0) += 0.08*xold(0)*(1.0 - xold(0))*xold(1);
      xnew(1) += 0.06*xold(0)*xold(1)*(1.0 - xold(1));
   });

   L2_FECollection fec(order, 2, BasisType::GaussLobatto);
   FiniteElementSpace scalar_fes(&mesh, &fec);
   FiniteElementSpace vector_fes(&mesh, &fec, vdim, ordering);
   IntegrationRules lobatto_rules(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir =
      rule_kind == 0 ? IntRules.Get(Geometry::SQUARE, 2*order) :
      rule_kind == 1 ? lobatto_rules.Get(Geometry::SQUARE, order + 1) :
                       IntRules.Get(Geometry::SQUARE, 2*order + 2);
   ConstantCoefficient coefficient(0.37);

   DGMassInverse scalar_inverse(scalar_fes, coefficient, ir,
                                BasisType::GaussLegendre);
   DGMassInverse vector_inverse(vector_fes, coefficient, ir,
                                BasisType::GaussLegendre);
   auto configure = [](DGMassInverse &inverse)
   {
      inverse.iterative_mode = false;
      inverse.SetRelTol(1e-14);
      inverse.SetAbsTol(0.0);
      inverse.SetMaxIter(100);
   };
   configure(scalar_inverse);
   configure(vector_inverse);

   const int scalar_size = scalar_fes.GetTrueVSize();
   auto vdof = [=](int dof, int component)
   {
      return ordering == Ordering::byNODES ?
             Ordering::Map<Ordering::byNODES>(scalar_size, vdim,
                                              dof, component) :
             Ordering::Map<Ordering::byVDIM>(scalar_size, vdim,
                                             dof, component);
   };

   constexpr real_t component_scale = 3.25;
   Vector b0(scalar_size), b1(scalar_size), b(vector_fes.GetTrueVSize());
   b0.Randomize(11);
   b1 = b0;
   b1 *= component_scale;
   for (int i = 0; i < scalar_size; ++i)
   {
      b[vdof(i,0)] = b0[i];
      b[vdof(i,1)] = b1[i];
   }

   Vector u0(scalar_size), u1(scalar_size), u(vector_fes.GetTrueVSize());
   scalar_inverse.Mult(b0, u0);
   scalar_inverse.Mult(b1, u1);
   vector_inverse.Mult(b, u);

   // Verify against an independent forward mass action. In particular, the
   // square Lobatto rule has the same number of points as dofs but a
   // nonidentity map in the internal Legendre basis, so it must retain CG.
   BilinearForm scalar_mass(&scalar_fes);
   scalar_mass.AddDomainIntegrator(new MassIntegrator(coefficient, &ir));
   scalar_mass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   scalar_mass.Assemble();
   Vector recovered(scalar_size);
   scalar_mass.Mult(u0, recovered);
   recovered -= b0;
   REQUIRE(recovered.Norml2() <=
           5e-12*std::max(b0.Norml2(), real_t(1.0)));

   // The packed recurrence executes the identical scalar arithmetic for each
   // component; require exact agreement, including the generic byVDIM gather
   // and scatter path.
   for (int i = 0; i < scalar_size; ++i)
   {
      REQUIRE(u[vdof(i,0)] == u0[i]);
      REQUIRE(u[vdof(i,1)] == u1[i]);
   }
   Vector component_error(u1);
   component_error.Add(-component_scale, u0);
   REQUIRE(component_error.Norml2() <=
           2e-12*std::max(u1.Norml2(), real_t(1.0)));

   // Reassembly must observe coefficient changes and preserve the same
   // componentwise result. Scaling M by s scales its inverse action by 1/s.
   constexpr real_t coefficient_scale = 2.5;
   Vector unscaled(u);
   coefficient.constant *= coefficient_scale;
   scalar_inverse.Update();
   vector_inverse.Update();
   scalar_mass.Assemble();
   scalar_inverse.Mult(b0, u0);
   scalar_inverse.Mult(b1, u1);
   vector_inverse.Mult(b, u);
   for (int i = 0; i < scalar_size; ++i)
   {
      REQUIRE(u[vdof(i,0)] == u0[i]);
      REQUIRE(u[vdof(i,1)] == u1[i]);
   }
   Vector coefficient_error(u);
   coefficient_error.Add(-1.0/coefficient_scale, unscaled);
   REQUIRE(coefficient_error.Norml2() <=
           2e-12*std::max(u.Norml2(), real_t(1.0)));
   scalar_mass.Mult(u0, recovered);
   recovered -= b0;
   REQUIRE(recovered.Norml2() <=
           5e-12*std::max(b0.Norml2(), real_t(1.0)));
}

TEST_CASE("Fehn ALE CFL rate is quadrature-point local",
          "[GPU][FehnALECFLRate]")
{
   const int dim = GENERATE(2, 3);
   constexpr int order = 3;
   Mesh mesh = dim == 2 ?
               Mesh::MakeCartesian2D(
                  2, 1, Element::QUADRILATERAL, true, 2.0, 0.25) :
               Mesh::MakeCartesian3D(
                  2, 1, 1, Element::HEXAHEDRON, 2.0, 0.25, 0.125);
   mesh.SetCurvature(order, false, dim, Ordering::byNODES);
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const IntegrationRule &ir = IntRules.Get(geometry, 2*order + 1);
   FehnALECFLRateOperator rate(fes, ir);

   Vector velocity(fes.GetVSize());
   const int scalar_size = fes.GetNDofs();
   for (int c = 0; c < dim; ++c)
   {
      const real_t value = c == 0 ? 2.0 : (c == 1 ? 0.5 : 0.25);
      for (int i = 0; i < scalar_size; ++i)
      {
         velocity[i + c*scalar_size] = value;
      }
   }
   Vector element_rate;
   rate.Mult(velocity, element_rate);
   const real_t expected = dim == 2 ? sqrt(8.0) : sqrt(12.0);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      REQUIRE(element_rate[e] == MFEM_Approx(expected, 1e-13, 1e-13));
   }
   REQUIRE(rate.ComputeMax(velocity) == MFEM_Approx(expected, 1e-13, 1e-13));
}

#ifdef MFEM_USE_MPI
TEST_CASE("Vector DG Mass Inverse permits empty MPI ranks",
          "[Parallel][DGMassInverseVectorEmpty]")
{
   int comm_size = 1;
   int comm_rank = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
   if (comm_size < 2) { return; }

   Mesh serial = Mesh::MakeCartesian2D(
                    1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   int partitioning[1] = { 0 };
   ParMesh mesh(MPI_COMM_WORLD, serial, partitioning);
   if (comm_rank > 0) { REQUIRE(mesh.GetNE() == 0); }

   constexpr int order = 3;
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   L2_FECollection fec(order, 2, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec, 2, ordering);
   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2*order);
   DGMassInverse inverse(fes, ir, BasisType::GaussLegendre);
   Vector b(fes.GetTrueVSize()), u(fes.GetTrueVSize());
   b.Randomize(19);
   inverse.Mult(b, u);
   inverse.Update();
   inverse.Mult(b, u);
   REQUIRE(u.Size() == fes.GetTrueVSize());
}
#endif
