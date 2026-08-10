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
