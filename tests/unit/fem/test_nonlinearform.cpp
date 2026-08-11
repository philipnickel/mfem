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

namespace
{

void SetALEElementComponent(FiniteElementSpace &fes, Vector &state,
                            int element, int component, real_t value)
{
   Array<int> vdofs;
   fes.GetElementVDofs(element, vdofs);
   const int dof = vdofs.Size() / fes.GetVDim();
   for (int j = 0; j < dof; j++)
   {
      state(vdofs[component * dof + j]) = value;
   }
}

real_t SumALEElementComponent(FiniteElementSpace &fes, const Vector &load,
                              int element, int component)
{
   Array<int> vdofs;
   fes.GetElementVDofs(element, vdofs);
   const int dof = vdofs.Size() / fes.GetVDim();
   real_t sum = 0.0;
   for (int j = 0; j < dof; j++)
   {
      sum += load(vdofs[component * dof + j]);
   }
   return sum;
}

} // namespace

TEST_CASE("ALE convection interior integrator", "[NonlinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                                     false, 2.0, 1.0);
   L2_FECollection fec(1, 2, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 4, Ordering::byNODES);
   Vector beta(1);
   beta = 1.0;
   NonlinearForm form(&fes);
   form.AddInteriorFaceIntegrator(
      new ALEConvectionInteriorIntegrator(1, 1.0, beta));

   Vector state(fes.GetVSize());
   Vector load(fes.GetVSize());
   state = 0.0;
   SetALEElementComponent(fes, state, 0, 0, 2.0);
   SetALEElementComponent(fes, state, 1, 0, -1.0);
   form.Mult(state, load);
   REQUIRE(SumALEElementComponent(fes, load, 0, 0) ==
           MFEM_Approx(0.0, 1e-12));
   REQUIRE(SumALEElementComponent(fes, load, 1, 0) ==
           MFEM_Approx(-1.5, 1e-12));

   // A discontinuous grid velocity reverses the upwind side after averaging
   // the two traces: average(u-w).n = -0.5 on this oriented face.
   SetALEElementComponent(fes, state, 1, 2, 2.0);
   form.Mult(state, load);
   REQUIRE(SumALEElementComponent(fes, load, 0, 0) ==
           MFEM_Approx(1.5, 1e-12));
   REQUIRE(SumALEElementComponent(fes, load, 1, 0) ==
           MFEM_Approx(0.0, 1e-12));
}

TEST_CASE("ALE convection interior integrator in 3D", "[NonlinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                                     2.0, 1.0, 1.0);
   L2_FECollection fec(1, 3, BasisType::GaussLobatto);
   // One velocity history followed by the grid velocity: 2 * dim components.
   FiniteElementSpace fes(&mesh, &fec, 6, Ordering::byNODES);
   Vector beta(1);
   beta = 1.0;
   NonlinearForm form(&fes);
   form.AddInteriorFaceIntegrator(
      new ALEConvectionInteriorIntegrator(1, 1.0, beta));

   Vector state(fes.GetVSize());
   Vector load(fes.GetVSize());
   state = 0.0;
   SetALEElementComponent(fes, state, 0, 0, 2.0);
   SetALEElementComponent(fes, state, 1, 0, -1.0);
   form.Mult(state, load);
   REQUIRE(SumALEElementComponent(fes, load, 0, 0) ==
           MFEM_Approx(0.0, 1e-12));
   REQUIRE(SumALEElementComponent(fes, load, 1, 0) ==
           MFEM_Approx(-1.5, 1e-12));

   // In 3D the grid-velocity block starts at component dim == 3.
   SetALEElementComponent(fes, state, 1, 3, 2.0);
   form.Mult(state, load);
   REQUIRE(SumALEElementComponent(fes, load, 0, 0) ==
           MFEM_Approx(1.5, 1e-12));
   REQUIRE(SumALEElementComponent(fes, load, 1, 0) ==
           MFEM_Approx(0.0, 1e-12));
}

TEST_CASE("ALE convection boundary integrator in 3D", "[NonlinearForm]")
{
   Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                     1.0, 1.0, 1.0);
   L2_FECollection fec(1, 3, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 6, Ordering::byNODES);
   Vector beta(1), delta(1), datum_value(3);
   beta = 1.0;
   delta = 1.0;
   datum_value = 0.0;
   datum_value(0) = 1.0;
   VectorConstantCoefficient datum(datum_value);

   NonlinearForm form(&fes);
   auto *integrator = new ALEConvectionBoundaryIntegrator(
      1, 1.0, beta, delta, true, true);
   integrator->SetDatum(datum);
   form.AddBdrFaceIntegrator(integrator);

   Vector state(fes.GetVSize()), load(fes.GetVSize());
   state = 0.0;
   SetALEElementComponent(fes, state, 0, 0, 1.0);
   form.Mult(state, load);
   // Constant velocity equal to the datum has neither a boundary correction
   // nor a convective pressure datum. This also pins the 3D pressure output
   // block at component dim == 3 without relying on a particular face order.
   REQUIRE(load.Normlinf() == MFEM_Approx(0.0, 1e-12));
}

TEST_CASE("Partial nonlinear form supports native face integrators",
          "[NonlinearForm][PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   Mesh mesh = dim == 2 ?
      Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                            false, 2.0, 1.0) :
      Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                            2.0, 1.0, 1.0);
   L2_FECollection fec(2, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   ConstantCoefficient coefficient(1.0);
   Array<int> boundary_marker(mesh.bdr_attributes.Max());
   boundary_marker = 0;
   boundary_marker[0] = 1;

   NonlinearForm nonlinear(&fes);
   nonlinear.AddInteriorFaceIntegrator(
      new VectorNormalJumpIntegrator(coefficient));
   nonlinear.AddBdrFaceIntegrator(
      new VectorNormalJumpIntegrator(coefficient), boundary_marker);
   nonlinear.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nonlinear.Setup();

   BilinearForm bilinear(&fes);
   bilinear.AddInteriorFaceIntegrator(
      new VectorNormalJumpIntegrator(coefficient));
   bilinear.AddBdrFaceIntegrator(
      new VectorNormalJumpIntegrator(coefficient), boundary_marker);
   bilinear.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   bilinear.Assemble();

   Vector state(fes.GetVSize()), nonlinear_load(fes.GetVSize());
   Vector bilinear_load(fes.GetVSize());
   state.Randomize(17);
   nonlinear.Mult(state, nonlinear_load);
   bilinear.Mult(state, bilinear_load);
   nonlinear_load -= bilinear_load;
   REQUIRE(nonlinear_load.Normlinf() == MFEM_Approx(0.0, 1e-12));
}

TEST_CASE("ALE volume partial assembly matches legacy",
          "[NonlinearForm][PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   const int degree = GENERATE(2, 6);
   const int history_order = 2;
   Mesh mesh = dim == 2 ?
      Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                            false, 2.0, 1.0) :
      Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                            2.0, 1.0, 1.0);
   L2_FECollection fec(degree, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim * (history_order + 1),
                          Ordering::byNODES);
   Vector beta(history_order);
   beta(0) = 0.65;
   beta(1) = -0.15;
   const Geometry::Type element_geometry =
      dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const IntegrationRule &rule =
      IntRules.Get(element_geometry, 2 * degree + 2);

   NonlinearForm legacy(&fes);
   auto *legacy_integrator =
      new ALEConvectionVolumeIntegrator(history_order, beta);
   legacy_integrator->SetIntRule(&rule);
   legacy.AddDomainIntegrator(legacy_integrator);

   NonlinearForm partial(&fes);
   auto *partial_integrator =
      new ALEConvectionVolumeIntegrator(history_order, beta);
   partial_integrator->SetIntRule(&rule);
   partial.AddDomainIntegrator(partial_integrator);
   partial.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   partial.Setup();

   Vector state(fes.GetVSize()), legacy_load(fes.GetVSize());
   Vector partial_load(fes.GetVSize());
   state.Randomize(17 + dim);
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));

   beta(0) = -0.45;
   beta(1) = 1.05;
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));
}

TEST_CASE("ALE interior partial assembly matches legacy",
          "[NonlinearForm][PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   const int degree = GENERATE(2, 6);
   const int history_order = 2;
   Mesh mesh = dim == 2 ?
      Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                            false, 2.0, 1.0) :
      Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                            2.0, 1.0, 1.0);
   L2_FECollection fec(degree, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim * (history_order + 1),
                          Ordering::byNODES);
   Vector beta(history_order);
   beta(0) = 0.7;
   beta(1) = -0.2;
   const Geometry::Type face_geometry =
      dim == 2 ? Geometry::SEGMENT : Geometry::SQUARE;
   const IntegrationRule &rule =
      IntRules.Get(face_geometry, 2 * degree + 2);

   NonlinearForm legacy(&fes);
   auto *legacy_integrator =
      new ALEConvectionInteriorIntegrator(history_order, 0.85, beta);
   legacy_integrator->SetIntRule(&rule);
   legacy.AddInteriorFaceIntegrator(legacy_integrator);

   NonlinearForm partial(&fes);
   auto *partial_integrator =
      new ALEConvectionInteriorIntegrator(history_order, 0.85, beta);
   partial_integrator->SetIntRule(&rule);
   partial.AddInteriorFaceIntegrator(partial_integrator);
   partial.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   partial.Setup();

   Vector state(fes.GetVSize()), legacy_load(fes.GetVSize());
   Vector partial_load(fes.GetVSize());
   state.Randomize(23 + dim);
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));

   // History weights are timestep data, not PA setup data.
   beta(0) = -0.35;
   beta(1) = 1.1;
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));
}

TEST_CASE("ALE boundary partial assembly matches legacy",
          "[NonlinearForm][PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   const bool use_constant_datum = GENERATE(false, true);
   INFO("dim=" << dim << ", constant datum=" << use_constant_datum);
   const int degree = GENERATE(2, 6);
   const int history_order = 2;
   Mesh mesh = dim == 2 ?
      Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                            false, 2.0, 1.0) :
      Mesh::MakeCartesian3D(2, 1, 1, Element::HEXAHEDRON,
                            2.0, 1.0, 1.0);
   L2_FECollection fec(degree, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim * (history_order + 1),
                          Ordering::byNODES);
   Vector beta(history_order), delta(history_order), datum_value(dim);
   beta(0) = 0.8;
   beta(1) = -0.3;
   delta(0) = 0.4;
   delta(1) = -0.1;
   for (int component = 0; component < dim; ++component)
   {
      datum_value(component) = 0.15 * (component + 1);
   }
   VectorConstantCoefficient constant_datum(datum_value);
   VectorFunctionCoefficient variable_datum(
      dim, [](const Vector &position, Vector &value)
   {
      for (int component = 0; component < value.Size(); ++component)
      {
         value(component) = 0.1 * (component + 1) +
                            0.05 * position(component);
      }
   });
   VectorCoefficient &datum = use_constant_datum ?
                              static_cast<VectorCoefficient&>(constant_datum) :
                              static_cast<VectorCoefficient&>(variable_datum);
   Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[0] = 1;
   const Geometry::Type face_geometry =
      dim == 2 ? Geometry::SEGMENT : Geometry::SQUARE;
   const IntegrationRule &rule =
      IntRules.Get(face_geometry, 2 * degree + 2);

   NonlinearForm legacy(&fes);
   auto *legacy_integrator = new ALEConvectionBoundaryIntegrator(
      history_order, 0.9, beta, delta, true, true);
   legacy_integrator->SetDatum(datum);
   legacy_integrator->SetIntRule(&rule);
   legacy.AddBdrFaceIntegrator(legacy_integrator, marker);

   NonlinearForm partial(&fes);
   auto *partial_integrator = new ALEConvectionBoundaryIntegrator(
      history_order, 0.9, beta, delta, true, true);
   partial_integrator->SetDatum(datum);
   partial_integrator->SetIntRule(&rule);
   partial.AddBdrFaceIntegrator(partial_integrator, marker);
   partial.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   partial.Setup();

   Vector state(fes.GetVSize()), legacy_load(fes.GetVSize());
   Vector partial_load(fes.GetVSize());
   state.Randomize(31 + dim);
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));

   beta(0) = -0.6;
   beta(1) = 1.2;
   delta(0) = -0.25;
   delta(1) = 0.65;
   legacy.Mult(state, legacy_load);
   partial.Mult(state, partial_load);
   partial_load -= legacy_load;
   REQUIRE(partial_load.Normlinf() <=
           1e-12 * (1.0 + legacy_load.Normlinf()));
}

TEST_CASE("NonlinearForm Boundary Integrator", "[NonlinearForm]")
{
   // See problem description in ex27.

   Mesh mesh("./data/holes.mesh", 1, 1);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);

   Array<int> nbc_bdr(mesh.bdr_attributes.Max());
   Array<int> rbc_bdr(mesh.bdr_attributes.Max());
   Array<int> dbc_bdr(mesh.bdr_attributes.Max());
   nbc_bdr = 0; nbc_bdr[0] = 1;
   rbc_bdr = 0; rbc_bdr[1] = 1;
   dbc_bdr = 0; dbc_bdr[2] = 1;

   Array<int> ess_tdof_list(0);
   fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);

   // See defaults in ex27.
   ConstantCoefficient matCoef(1.0);
   ConstantCoefficient dbcCoef(0.0);
   ConstantCoefficient nbcCoef(1.0);
   ConstantCoefficient rbcACoef(1.0);
   ConstantCoefficient rbcBCoef(1.0);
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   GridFunction u1(&fespace), u2(&fespace);
   u1 = 0.0;
   u2 = 0.0;
   u1.ProjectBdrCoefficient(dbcCoef, dbc_bdr);
   u2.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

   LinearForm b(&fespace);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   b.Assemble();

   // Solve as a linear problem.
   {
      BilinearForm a(&fespace);
      a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, u1, b, A, X, B);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
      a.RecoverFEMSolution(X, b, u1);
   }

   // Solve as a nonlinear problem.
   {
      NonlinearForm a_nf(&fespace);
      a_nf.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
      a_nf.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
      a_nf.SetEssentialTrueDofs(ess_tdof_list);

      IterativeSolver::PrintLevel print;
      print.Iterations();
      CGSolver cg;
      cg.SetPrintLevel(print);
      cg.SetMaxIter(100);
      cg.SetRelTol(1e-12); cg.SetAbsTol(0.0);

      NewtonSolver newton;
      newton.iterative_mode = false;
      newton.SetSolver(cg);
      newton.SetOperator(a_nf);
      newton.SetPrintLevel(print);
      newton.SetRelTol(1e-14); newton.SetAbsTol(0.0);
      newton.SetMaxIter(1);

      newton.Mult(b, u2);
   }

   u2 -= u1;
   REQUIRE(u2.Norml2() == MFEM_Approx(0.0, 1e-5));
}
