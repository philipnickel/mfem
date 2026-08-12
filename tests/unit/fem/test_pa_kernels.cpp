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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

namespace pa_kernels
{

enum class FECType
{
   H1,
   L2_VALUE,
   L2_INTEGRAL
};

std::unique_ptr<FiniteElementCollection> create_fec(
   FECType fec_type, int order, int dim)
{
   using Ptr = std::unique_ptr<FiniteElementCollection>;
   switch (fec_type)
   {
      case FECType::H1:
         return Ptr(new H1_FECollection(order, dim));
      case FECType::L2_VALUE:
         return Ptr(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                        FiniteElement::VALUE));
      case FECType::L2_INTEGRAL:
         return Ptr(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                        FiniteElement::INTEGRAL));
      default:
         MFEM_ABORT("Invalid FECType");
   }
}

Mesh MakeCartesianNonaligned(const int dim, const int ne)
{
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(ne, ne, ne, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }

   // Remap vertices so that the mesh is not aligned with axes.
   for (int i=0; i<mesh.GetNV(); ++i)
   {
      real_t *vcrd = mesh.GetVertex(i);
      vcrd[1] += 0.2 * vcrd[0];
      if (dim == 3) { vcrd[2] += 0.3 * vcrd[0]; }
   }

   return mesh;
}

real_t zero_field(const Vector &x)
{
   MFEM_CONTRACT_VAR(x);
   return 0.0;
}

void solenoidal_field2d(const Vector &x, Vector &u)
{
   u(0) = x(1);
   u(1) = -x(0);
}

void non_solenoidal_field2d(const Vector &x, Vector &u)
{
   u(0) = x(0) * x(1);
   u(1) = -x(0) + x(1);
}

real_t div_non_solenoidal_field2d(const Vector &x)
{
   return 1.0 + x(1);
}

void solenoidal_field3d(const Vector &x, Vector &u)
{
   u(0) = -x(0)*x(0);
   u(1) = x(0)*x(1);
   u(2) = x(0)*x(2);
}

TEST_CASE("PA mass and convection on embedded tensor surfaces",
          "[PartialAssembly][MassIntegrator][ConvectionIntegrator]")
{
   const int parent_dim = GENERATE(2, 3);
   Mesh parent = parent_dim == 2 ?
                 Mesh::MakeCartesian2D(4, 2, Element::QUADRILATERAL,
                                       false, 2.0, 1.0) :
                 Mesh::MakeCartesian3D(3, 2, 1, Element::HEXAHEDRON,
                                       2.0, 1.5, 1.0);
   Array<int> attributes({parent_dim == 2 ? 3 : 6});
   SubMesh surface = SubMesh::CreateFromBoundary(parent, attributes);
   const int dim = surface.Dimension();
   const int order = 3;
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&surface, &fec);

   VectorFunctionCoefficient velocity(parent_dim,
      [parent_dim](const Vector &, Vector &value)
   {
      value.SetSize(parent_dim);
      value = 0.0;
      value[0] = 0.7;
      if (parent_dim == 3) { value[1] = -0.4; }
   });
   FunctionCoefficient affine([parent_dim](const Vector &x)
   {
      return parent_dim == 2 ? x[0] : 0.17*x[0] - 0.23*x[1];
   });
   const real_t derivative = parent_dim == 2 ? 0.7 :
                             0.7*0.17 + (-0.4)*(-0.23);
   const Geometry::Type geometry = dim == 1 ? Geometry::SEGMENT : Geometry::SQUARE;
   const IntegrationRule &ir = IntRules.Get(geometry, 4*order + 3);

   BilinearForm mass_pa(&fes), mass_fa(&fes), convection_pa(&fes);
   mass_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   convection_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_pa.AddDomainIntegrator(new MassIntegrator(&ir));
   mass_fa.AddDomainIntegrator(new MassIntegrator(&ir));
   auto *convection = new ConvectionIntegrator(velocity, 1.0);
   convection->SetIntRule(&ir);
   convection_pa.AddDomainIntegrator(convection);
   mass_pa.Assemble();
   mass_fa.Assemble(); mass_fa.Finalize();
   convection_pa.Assemble();

   GridFunction field(&fes);
   field.ProjectCoefficient(affine);
   Vector one(fes.GetVSize()), pa(fes.GetVSize()), reference(fes.GetVSize());
   one = 1.0;

   mass_pa.Mult(one, pa);
   mass_fa.Mult(one, reference);
   pa -= reference;
   REQUIRE(pa.Normlinf() <= 5e-13*std::max(real_t(1.0), reference.Normlinf()));

   convection_pa.Mult(field, pa);
   mass_pa.Mult(one, reference);
   reference *= derivative;
   pa -= reference;
   REQUIRE(pa.Normlinf() <= 5e-13*std::max(real_t(1.0), reference.Normlinf()));
}

void non_solenoidal_field3d(const Vector &x, Vector &u)
{
   u(0) = x(0)*x(0);
   u(1) = x(1)*x(1);
   u(2) = x(2)*x(2);
}

real_t div_non_solenoidal_field3d(const Vector &x)
{
   return 2*(x(0) + x(1) + x(2));
}

class ElementP0Coefficient : public Coefficient
{
private:
   const Vector &values;

public:
   explicit ElementP0Coefficient(const Vector &values_) : values(values_) { }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &) override
   { return values[T.ElementNo]; }
};

void pa_divergence_testnd(int dim,
                          void (*f1)(const Vector &, Vector &),
                          real_t (*divf1)(const Vector &))
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Vector valued
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1, dim);

   // Scalar
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2);

   GridFunction field(&fes1), field2(&fes2);

   MixedBilinearForm dform(&fes1, &fes2);
   dform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   dform.AddDomainIntegrator(new VectorDivergenceIntegrator);
   dform.Assemble();

   // Project u = f1
   VectorFunctionCoefficient fcoeff1(dim, f1);
   field.ProjectCoefficient(fcoeff1);

   // Check if div(u) = divf1
   dform.Mult(field, field2);
   FunctionCoefficient fcoeff2(divf1);
   LinearForm lf(&fes2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(fcoeff2));
   lf.Assemble();
   field2 -= lf;

   REQUIRE(field2.Normlinf() == MFEM_Approx(0.0));
}

template <typename INTEGRATOR>
void pa_mixed_transpose_test(FiniteElementSpace &fes1,
                             FiniteElementSpace &fes2)
{
   MixedBilinearForm bform_pa(&fes1, &fes2);
   bform_pa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   bform_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   bform_pa.Assemble();

   MixedBilinearForm bform_fa(&fes1, &fes2);
   bform_fa.AddDomainIntegrator(new TransposeIntegrator(new INTEGRATOR));
   bform_fa.Assemble();
   bform_fa.Finalize();

   GridFunction x(&fes1), y_pa(&fes2), y_fa(&fes2);
   x.Randomize(1);

   bform_pa.Mult(x, y_pa);
   bform_fa.Mult(x, y_fa);

   y_pa -= y_fa;
   REQUIRE(y_pa.Normlinf() == MFEM_Approx(0.0));
}

void pa_divergence_transpose_testnd(int dim)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1);

   // Vector valued
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2, dim);

   pa_mixed_transpose_test<VectorDivergenceIntegrator>(fes1, fes2);
}

TEST_CASE("PA VectorDivergence", "[PartialAssembly], [GPU]")
{
   SECTION("2D")
   {
      // Check if div([y, -x]) == 0
      pa_divergence_testnd(2, solenoidal_field2d, zero_field);
      // Check if div([x*y, -x+y]) == 1 + y
      pa_divergence_testnd(2, non_solenoidal_field2d, div_non_solenoidal_field2d);
      // Check transpose
      pa_divergence_transpose_testnd(2);
   }

   SECTION("3D")
   {
      // Check if div([-x^2, xy, xz]) == 0
      pa_divergence_testnd(3, solenoidal_field3d, zero_field);
      // Check if div([x^2, y^2, z^2]) == 2(x + y + z)
      pa_divergence_testnd(3, non_solenoidal_field3d, div_non_solenoidal_field3d);
      // Check transpose
      pa_divergence_transpose_testnd(3);
   }
}

TEST_CASE("PA VectorDivDiv on curved L2 elements",
          "[PartialAssembly][GPU][VectorDivDiv]")
{
   constexpr int order = 3;
   const int dim = GENERATE(2, 3);
   const int q1d = GENERATE(order + 1, order + 2);
   CAPTURE(dim);
   CAPTURE(q1d);
   Mesh mesh;
   if (dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(
                3, 2, Element::QUADRILATERAL, true, 1.0, 1.0);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(
                2, 2, 2, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   }
   mesh.SetCurvature(order, false, dim, Ordering::byNODES);
   mesh.Transform([](const Vector &xold, Vector &xnew)
   {
      xnew = xold;
      real_t perturbation = 0.08;
      for (int d = 0; d < xold.Size(); ++d)
      {
         perturbation *= sin(M_PI*xold(d));
      }
      xnew(xnew.Size() - 1) += perturbation;
   });

   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, dim, Ordering::byNODES);
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   const IntegrationRule &ir = IntRules.Get(geometry, 2*q1d - 2);

   Vector tau(mesh.GetNE());
   for (int e = 0; e < tau.Size(); ++e) { tau[e] = 0.2 + 0.07*e; }

   BilinearForm pa(&fes);
   pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   auto *pa_integ = new VectorDivDivIntegrator;
   pa_integ->SetElementCoefficient(tau);
   pa_integ->SetIntRule(&ir);
   pa.AddDomainIntegrator(pa_integ);
   pa.Assemble();

   ElementP0Coefficient lambda(tau);
   ConstantCoefficient zero(0.0);
   BilinearForm assembled(&fes);
   auto *reference = new ElasticityIntegrator(lambda, zero);
   reference->SetIntRule(&ir);
   assembled.AddDomainIntegrator(reference);
   assembled.Assemble();
   assembled.Finalize();

   GridFunction x(&fes), y_pa(&fes), y_assembled(&fes);
   x.Randomize(17);
   pa.Mult(x, y_pa);
   assembled.Mult(x, y_assembled);

   Vector error(y_pa);
   error -= y_assembled;
   const real_t reference_norm = y_assembled.Norml2();
   REQUIRE(error.Norml2() <= 2e-12*std::max(reference_norm, real_t(1.0)));

   Vector pa_diagonal(fes.GetVSize());
   pa.AssembleDiagonal(pa_diagonal);
   Vector assembled_diagonal(fes.GetVSize());
   assembled.SpMat().GetDiag(assembled_diagonal);
   pa_diagonal -= assembled_diagonal;
   REQUIRE(pa_diagonal.Normlinf() <=
           2e-12*std::max(assembled_diagonal.Normlinf(), real_t(1.0)));

   // SetElementCoefficient owns a fresh copy. Reassembly must replace, rather
   // than accumulate onto, the old quadrature data.
   constexpr real_t coefficient_scale = 1.75;
   Vector original(y_pa);
   GridFunction scaled(&fes);
   tau *= coefficient_scale;
   pa_integ->SetElementCoefficient(tau);
   pa.Assemble();
   pa.Mult(x, scaled);
   scaled.Add(-coefficient_scale, original);
   REQUIRE(scaled.Norml2() <=
           2e-12*std::max(original.Norml2(), real_t(1.0)));
}

TEST_CASE("Empty L2 lexicographic permutation uses native ordering",
          "[DofToQuad][VectorDivDiv]")
{
   constexpr int order = 3;
   Mesh mesh = Mesh::MakeCartesian2D(
                  1, 1, Element::QUADRILATERAL, true, 1.0, 1.0);
   L2_FECollection fec(order, 2, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   const FiniteElement &fe = *fes.GetTypicalFE();
   const auto &nodal_fe = dynamic_cast<const NodalFiniteElement &>(fe);
   REQUIRE(nodal_fe.GetLexicographicOrdering().Size() == 0);

   const IntegrationRule &ir = IntRules.Get(Geometry::SQUARE, 2*order);
   const DofToQuad &native = fe.GetDofToQuad(ir, DofToQuad::FULL);
   const DofToQuad &lex = fe.GetDofToQuad(ir, DofToQuad::LEXICOGRAPHIC_FULL);

   REQUIRE(lex.B.Size() == native.B.Size());
   REQUIRE(lex.Bt.Size() == native.Bt.Size());
   REQUIRE(lex.G.Size() == native.G.Size());
   REQUIRE(lex.Gt.Size() == native.Gt.Size());
   for (int i = 0; i < native.B.Size(); ++i) { REQUIRE(lex.B[i] == native.B[i]); }
   for (int i = 0; i < native.Bt.Size(); ++i) { REQUIRE(lex.Bt[i] == native.Bt[i]); }
   for (int i = 0; i < native.G.Size(); ++i) { REQUIRE(lex.G[i] == native.G[i]); }
   for (int i = 0; i < native.Gt.Size(); ++i) { REQUIRE(lex.Gt[i] == native.Gt[i]); }
}

real_t f1(const Vector &x)
{
   real_t r = pow(x(0),2);
   if (x.Size() >= 2) { r += pow(x(1), 3); }
   if (x.Size() >= 3) { r += pow(x(2), 4); }
   return r;
}

void gradf1(const Vector &x, Vector &u)
{
   u(0) = 2*x(0);
   if (x.Size() >= 2) { u(1) = 3*pow(x(1), 2); }
   if (x.Size() >= 3) { u(2) = 4*pow(x(2), 3); }
}

void pa_gradient_testnd(int dim, FECType fec_type,
                        real_t (*f1)(const Vector &),
                        void (*gradf1)(const Vector &, Vector &))
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec1(order, dim);
   FiniteElementSpace fes1(&mesh, &fec1);
   GridFunction field(&fes1);

   // Vector valued
   auto fec2 = create_fec(fec_type, order, dim);
   FiniteElementSpace fes2(&mesh, fec2.get(), dim);
   GridFunction field2(&fes2);

   MixedBilinearForm gform(&fes1, &fes2);
   gform.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   gform.AddDomainIntegrator(new GradientIntegrator);
   gform.Assemble();

   // Project u = f1
   FunctionCoefficient fcoeff1(f1);
   field.ProjectCoefficient(fcoeff1);

   // Check if grad(u) = gradf1
   gform.Mult(field, field2);
   VectorFunctionCoefficient fcoeff2(dim, gradf1);
   LinearForm lf(&fes2);
   lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
   lf.Assemble();
   field2 -= lf;

   REQUIRE(field2.Norml2() == MFEM_Approx(0.0));
}

void pa_gradient_transpose_testnd(int dim, FECType fec_type)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 4;

   // Scalar
   H1_FECollection fec2(order, dim);
   FiniteElementSpace fes2(&mesh, &fec2);
   GridFunction y_pa(&fes2), y_fa(&fes2);

   // Vector valued
   auto fec1 = create_fec(fec_type, order, dim);
   FiniteElementSpace fes1(&mesh, fec1.get(), dim);

   pa_mixed_transpose_test<GradientIntegrator>(fes1, fes2);
}

TEST_CASE("PA Gradient", "[PartialAssembly], [GPU]")
{
   auto fec_type = GENERATE(FECType::H1, FECType::L2_VALUE,
                            FECType::L2_INTEGRAL);

   SECTION("2D")
   {
      // Check if grad(x^2 + y^3) == [2x, 3y^2]
      pa_gradient_testnd(2, fec_type, f1, gradf1);
      // Check transpose
      pa_gradient_transpose_testnd(2, fec_type);
   }

   SECTION("3D")
   {
      // Check if grad(x^2 + y^3 + z^4) == [2x, 3y^2, 4z^3]
      pa_gradient_testnd(3, fec_type, f1, gradf1);
      // Check transpose
      pa_gradient_transpose_testnd(3, fec_type);
   }
}

real_t test_nl_convection_nd(int dim)
{
   Mesh mesh = MakeCartesianNonaligned(dim, 2);
   int order = 2;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(3);

   NonlinearForm nlf_fa(&fes);
   nlf_fa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_fa.Mult(x, y_fa);

   NonlinearForm nlf_pa(&fes);
   nlf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   nlf_pa.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
   nlf_pa.Setup();
   nlf_pa.Mult(x, y_pa);

   y_fa -= y_pa;
   real_t difference = y_fa.Norml2();


   return difference;
}

TEST_CASE("Nonlinear Convection", "[PartialAssembly], [NonlinearPA], [GPU]")
{
   SECTION("2D")
   {
      REQUIRE(test_nl_convection_nd(2) == MFEM_Approx(0.0));
   }

   SECTION("3D")
   {
      REQUIRE(test_nl_convection_nd(3) == MFEM_Approx(0.0));
   }
}

template <typename INTEGRATOR>
real_t test_pa_vector_integrator(int dim, int sdim)
{
   const bool all = launch_all_non_regression_tests;
   const auto NE = all ? GENERATE(1, 2, 3) : 2;
   const auto p = all ? GENERATE(1, 2, 3): 2;
   CAPTURE(p, NE);

   Mesh mesh = MakeCartesianNonaligned(dim, NE);
   mesh.SetCurvature(p, false, sdim);

   H1_FECollection fec(p, dim);
   FiniteElementSpace fes(&mesh, &fec, sdim);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   ConstantCoefficient const_coeff(M_PI_2);
   FunctionCoefficient funct_coeff([](const Vector &x) { return M_1_PI + x[0]*x[0]; });

   Vector val(dim); val = 1.0;
   VectorConstantCoefficient v_const_coeff(val);
   VectorFunctionCoefficient v_funct_coeff(dim, [&](const Vector &x, Vector &v)
   {
      v(0) = M_LN2 * x(0);
      if (dim > 1) { v(1) = M_E * x(1); }
      if (dim > 2) { v(2) = M_PI * x(2); }
   });

   MatrixFunctionCoefficient m_funct_coeff(dim, [&](const Vector &x,
                                                    DenseMatrix &f)
   {
      f = 0.0;
      if (dim == 1)
      {
         f(0,0) = 1.1 + sin(M_PI * x[0]);  // 1,1
      }
      else if (dim == 2)
      {
         f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
         f(1,0) = cos(1.3 * M_PI * x[1]);  // 2,1
         f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
         f(1,1) = 1.1 + sin(4.9 * M_PI * x[0]);  // 2,2
      }
      else if (dim == 3)
      {
         f(0,0) = 1.1 + sin(M_PI * x[1]);  // 1,1
         f(0,1) = cos(2.5 * M_PI * x[0]);  // 1,2
         f(0,2) = sin(4.9 * M_PI * x[2]);  // 1,3
         f(1,0) = cos(M_PI * x[0]);  // 2,1
         f(1,1) = 1.1 + sin(6.1 * M_PI * x[1]);  // 2,2
         f(1,2) = cos(6.1 * M_PI * x[2]);  // 2,3
         f(2,0) = sin(1.5 * M_PI * x[1]);  // 3,1
         f(2,1) = cos(2.9 * M_PI * x[0]);  // 3,2
         f(2,2) = 1.1 + sin(6.1 * M_PI * x[2]);  // 3,3
      }
   });
   REQUIRE((sdim > dim || m_funct_coeff.GetVDim() == fes.GetVDim()));

   BilinearForm blf_fa(&fes);
   blf_fa.SetAssemblyLevel(AssemblyLevel::LEGACY);
   // scalar coefficients
   blf_fa.AddDomainIntegrator(new INTEGRATOR);
   blf_fa.AddDomainIntegrator(new INTEGRATOR(const_coeff));
   blf_fa.AddDomainIntegrator(new INTEGRATOR(funct_coeff));
   if (sdim == dim)
   {
      // vector coefficients
      blf_fa.AddDomainIntegrator(new INTEGRATOR(v_const_coeff));
      blf_fa.AddDomainIntegrator(new INTEGRATOR(v_funct_coeff));
      // matrix coefficients
      blf_fa.AddDomainIntegrator(new INTEGRATOR(m_funct_coeff));
   }
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // scalar coefficients
   blf_pa.AddDomainIntegrator(new INTEGRATOR);
   blf_pa.AddDomainIntegrator(new INTEGRATOR(const_coeff));
   blf_pa.AddDomainIntegrator(new INTEGRATOR(funct_coeff));
   if (sdim == dim)
   {
      // vector coefficients
      blf_pa.AddDomainIntegrator(new INTEGRATOR(v_const_coeff));
      blf_pa.AddDomainIntegrator(new INTEGRATOR(v_funct_coeff));
      // matrix coefficients
      blf_pa.AddDomainIntegrator(new INTEGRATOR(m_funct_coeff));
   }
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   return y_fa.Norml2();
}

TEST_CASE("PA Vector Mass",
          "[PartialAssembly][VectorPA][VectorMassPA][GPU]")
{
   const auto DIM = GENERATE(2, 3);
   CAPTURE(DIM);
   REQUIRE(test_pa_vector_integrator<VectorMassIntegrator>(DIM, DIM)
           == MFEM_Approx(0.0));
}

TEST_CASE("PA Vector Diffusion",
          "[PartialAssembly][VectorPA][VectorDiffusionPA][GPU]")
{
   const auto DIM = GENERATE(2, 3);
   CAPTURE(DIM);
   REQUIRE(test_pa_vector_integrator<VectorDiffusionIntegrator>(DIM, DIM)
           == MFEM_Approx(0.0));
}

TEST_CASE("PA Vector Diffusion 2D/3D",
          "[PartialAssembly][VectorPA][VectorDiffusionPA][CUDA]")
{
   const int DIM = 2, SDIM = 3;
   CAPTURE(DIM, SDIM);
   REQUIRE(test_pa_vector_integrator<VectorDiffusionIntegrator>(DIM, SDIM)
           == MFEM_Approx(0.0));
}

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = x(1); v(1) = -x(0); break;
      case 3: v(0) = x(1); v(1) = -x(0); v(2) = x(0); break;
   }
}

void AddConvectionIntegrators(BilinearForm &k, Coefficient &rho,
                              VectorCoefficient &velocity, bool dg)
{
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));

   if (dg)
   {
      k.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
      k.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(rho, velocity, 1.0, -0.5)));
   }
}

void test_pa_convection(const std::string &meshname, int order, int prob,
                        int refinement)
{
   INFO("mesh=" << meshname << ", order=" << order << ", prob=" << prob
        << ", refinement=" << refinement );
   Mesh mesh(meshname.c_str(), 1, 1);
   mesh.EnsureNodes();
   mesh.SetCurvature(mesh.GetNodalFESpace()->GetElementOrder(0));
   for (int r = 0; r < refinement; r++)
   {
      mesh.RandomRefinement(0.6,false,1,4);
   }
   int dim = mesh.Dimension();

   std::unique_ptr<FiniteElementCollection> fec;
   if (prob)
   {
      auto basis = prob==3 ? BasisType::Positive : BasisType::GaussLobatto;
      fec.reset(new L2_FECollection(order, dim, basis));
   }
   else
   {
      fec.reset(new H1_FECollection(order, dim));
   }
   FiniteElementSpace fespace(&mesh, fec.get());

   L2_FECollection vel_fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace vel_fespace(&mesh, &vel_fec, dim);
   GridFunction vel_gf(&vel_fespace);
   GridFunction rho_gf(&fespace);

   BilinearForm k_pa(&fespace);
   BilinearForm k_fa(&fespace);

   std::unique_ptr<VectorCoefficient> vel_coeff;
   std::unique_ptr<Coefficient> rho;

   // prob: 0: CG, 1: DG continuous coeff, 2: DG discontinuous coeff
   if (prob >= 2)
   {
      vel_gf.Randomize(1);
      vel_coeff.reset(new VectorGridFunctionCoefficient(&vel_gf));
      rho_gf.Randomize(1);
      rho.reset(new GridFunctionCoefficient(&rho_gf));
   }
   else
   {
      vel_coeff.reset(new VectorFunctionCoefficient(dim, velocity_function));
      rho.reset(new ConstantCoefficient(1.0));
   }


   AddConvectionIntegrators(k_fa, *rho, *vel_coeff, prob > 0);
   AddConvectionIntegrators(k_pa, *rho, *vel_coeff, prob > 0);

   k_fa.Assemble();
   k_fa.Finalize();

   k_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   k_pa.Assemble();

   GridFunction x(&fespace), y_fa(&fespace), y_pa(&fespace);

   x.Randomize(1);

   // Testing Mult
   k_fa.Mult(x,y_fa);
   k_pa.Mult(x,y_pa);

   y_pa -= y_fa;

   REQUIRE(y_pa.Norml2() < 1.e-12);

   // Testing MultTranspose
   k_fa.MultTranspose(x,y_fa);
   k_pa.MultTranspose(x,y_pa);

   y_pa -= y_fa;

   REQUIRE(y_pa.Norml2() < 1.e-12);
}

// Basic unit tests for convection
TEST_CASE("PA Convection", "[PartialAssembly], [GPU]")
{
   // prob:
   // - 0: CG,
   // - 1: DG continuous coeff,
   // - 2: DG discontinuous coeff,
   // - 3: DG Bernstein discontinuous coeff.
   auto prob = GENERATE(0, 1, 2, 3);
   auto order = GENERATE(2);
   // refinement > 0 => Non-conforming mesh
   auto refinement = GENERATE(0, 1);

   SECTION("2D")
   {
      test_pa_convection("../../data/periodic-square.mesh", order, prob,
                         refinement);
   }

   SECTION("3D")
   {
      test_pa_convection("../../data/periodic-cube.mesh", order, prob,
                         refinement);
   }
} // test case

TEST_CASE("PA DG trace with degree-10 overintegration",
          "[PartialAssembly][DGTrace][CPU]")
{
   constexpr int order = 10;
   constexpr int qpts = 16;

   Mesh mesh("../../data/periodic-square.mesh", 1, 1);
   L2_FECollection fec(order, 2, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   Vector velocity_value({1.0, -0.25});
   VectorConstantCoefficient velocity(velocity_value);
   ConstantCoefficient rho(1.0);
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2*qpts - 1);
   REQUIRE(ir.GetNPoints() == qpts);

   BilinearForm full(&fes);
   BilinearForm partial(&fes);
   auto add_face_integrator = [&](BilinearForm &form)
   {
      auto *trace = new TransposeIntegrator(
         new DGTraceIntegrator(rho, velocity, 1.0, -0.5));
      trace->SetIntegrationRule(ir);
      form.AddInteriorFaceIntegrator(trace);
   };
   add_face_integrator(full);
   add_face_integrator(partial);

   full.Assemble();
   full.Finalize();
   partial.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   partial.Assemble();

   GridFunction x(&fes), y_full(&fes), y_partial(&fes);
   x.Randomize(1);

   full.Mult(x, y_full);
   partial.Mult(x, y_partial);
   y_partial -= y_full;
   REQUIRE(y_partial.Norml2() <= 1e-11);

   full.MultTranspose(x, y_full);
   partial.MultTranspose(x, y_partial);
   y_partial -= y_full;
   REQUIRE(y_partial.Norml2() <= 1e-11);
}

// Advanced unit tests for convection
TEST_CASE("PA Convection advanced", "[PartialAssembly], [MFEMData], [GPU]")
{
   if (launch_all_non_regression_tests)
   {
      // prob:
      // - 0: CG,
      // - 1: DG continuous coeff,
      // - 2: DG discontinuous coeff,
      // - 3: DG Bernstein discontinuous coeff.
      auto prob = GENERATE(0, 1, 2, 3);
      auto order = GENERATE(2);
      // refinement > 0 => Non-conforming mesh
      auto refinement = GENERATE(0,1);

      SECTION("2D")
      {
         test_pa_convection("../../data/periodic-hexagon.mesh", order, prob,
                            refinement);
         test_pa_convection("../../data/star-q3.mesh", order, prob,
                            refinement);
         test_pa_convection(mfem_data_dir+"/gmsh/v22/unstructured_quad.v22.msh",
                            order, prob, refinement);
      }

      SECTION("3D")
      {
         test_pa_convection("../../data/fichera-q3.mesh", order, prob,
                            refinement);
         test_pa_convection(mfem_data_dir+"/gmsh/v22/unstructured_hex.v22.msh",
                            order, prob, refinement);
      }
   }
} // PA Convection test case

template <typename INTEGRATOR>
static void test_pa_integrator()
{
   const bool all_tests = launch_all_non_regression_tests;

   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto map_type = GENERATE(FiniteElement::VALUE, FiniteElement::INTEGRAL);

   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 3);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto, map_type);
   FiniteElementSpace fes(&mesh, &fec);

   const int q_order = 2*order + q_order_inc;
   // Don't use a special integration rule if q_order_inc == 0
   const bool use_ir = q_order_inc > 0;
   const IntegrationRule *ir =
      use_ir ? &IntRules.Get(mesh.GetTypicalElementGeometry(), q_order) : nullptr;

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   FunctionCoefficient coeff(f1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new INTEGRATOR(coeff,ir));
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new INTEGRATOR(coeff,ir));
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("PA Mass", "[PartialAssembly], [GPU]")
{
   test_pa_integrator<MassIntegrator>();
} // PA Mass test case

TEST_CASE("PA Diffusion", "[PartialAssembly], [GPU]")
{
   test_pa_integrator<DiffusionIntegrator>();
} // PA Diffusion test case

TEST_CASE("PA Markers", "[PartialAssembly], [GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;
   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   auto dg = GENERATE(false, true);
   CAPTURE(fname, order, dg);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   std::unique_ptr<FiniteElementCollection> fec;
   if (dg) { fec.reset(new L2_FECollection(order, dim, BasisType::GaussLobatto)); }
   else { fec.reset(new H1_FECollection(order, dim)); }
   FiniteElementSpace fes(&mesh, fec.get());

   for (int i = 0; i < mesh.GetNE(); ++i) { mesh.SetAttribute(i, 1 + i%2); }
   for (int i = 0; i < mesh.GetNBE(); ++i) { mesh.SetBdrAttribute(i, 1 + i%2); }
   mesh.SetAttributes();

   Array<int> marker(2);
   marker[0] = 0;
   marker[1] = 1;

   Vector vel_vec(dim);
   vel_vec.Randomize(1);
   VectorConstantCoefficient vel(vel_vec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddDomainIntegrator(new MassIntegrator, marker);
   if (dg) { blf_fa.AddBdrFaceIntegrator(new DGTraceIntegrator(vel, 1.0)); }
   else { blf_fa.AddBoundaryIntegrator(new MassIntegrator, marker); }
   blf_fa.Assemble();
   blf_fa.Finalize();

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator, marker);
   if (dg) { blf_pa.AddBdrFaceIntegrator(new DGTraceIntegrator(vel, 1.0)); }
   else { blf_pa.AddBoundaryIntegrator(new MassIntegrator, marker); }
   blf_pa.Assemble();

   blf_fa.Mult(x, y_fa);
   blf_pa.Mult(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));

   blf_fa.MultTranspose(x, y_fa);
   blf_pa.MultTranspose(x, y_pa);
   y_fa -= y_pa;
   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

TEST_CASE("PA Boundary Mass", "[PartialAssembly], [GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;

   auto fname = GENERATE("../../data/star.mesh", "../../data/star-q3.mesh",
                         "../../data/fichera.mesh", "../../data/fichera-q3.mesh");
   auto order = !all_tests ? 2 : GENERATE(1, 2, 3);

   Mesh mesh(fname);
   int dim = mesh.Dimension();
   RT_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   FunctionCoefficient coeff(f1);

   BilinearForm blf_fa(&fes);
   blf_fa.AddBoundaryIntegrator(new MassIntegrator(coeff));
   blf_fa.Assemble();
   blf_fa.Finalize();
   blf_fa.Mult(x, y_fa);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddBoundaryIntegrator(new MassIntegrator(coeff));
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

namespace
{
template <typename T> struct ParTypeHelper { };
template <> struct ParTypeHelper<FiniteElementSpace>
{
   using GF_t = GridFunction;
   using BLF_t = BilinearForm;
};
#ifdef MFEM_USE_MPI
template <> struct ParTypeHelper<ParFiniteElementSpace>
{
   using GF_t = ParGridFunction;
   using BLF_t = ParBilinearForm;
};
#endif
}

template <typename CoeffType>
std::unique_ptr<CoeffType> MakeCoeff(int);

template <>
std::unique_ptr<ConstantCoefficient> MakeCoeff<ConstantCoefficient>(int)
{
   return std::make_unique<ConstantCoefficient>(3.14159);
}

template <>
std::unique_ptr<MatrixConstantCoefficient> MakeCoeff<MatrixConstantCoefficient>
(int dim)
{
   DenseMatrix A(dim);
   for (int i = 0; i < dim*dim; ++i)
   {
      A.GetData()[i] = 1.0 / (i + 3.0);
   }
   for (int i = 0; i < dim; ++i)
   {
      A(i,i) += 2.0 + i;
   }
   return std::make_unique<MatrixConstantCoefficient>(A);
}

template <> std::unique_ptr<SymmetricMatrixConstantCoefficient>
MakeCoeff<SymmetricMatrixConstantCoefficient>(int dim)
{
   DenseSymmetricMatrix A(dim);
   for (int i = 0; i < A.GetStoredSize(); ++i)
   {
      A.GetData()[i] = 1.0 / (i + 3.0);
   }
   for (int i = 0; i < dim; ++i)
   {
      A(i,i) += 2.0 + i;
   }
   return std::make_unique<SymmetricMatrixConstantCoefficient>(A);
}

template <typename CoeffType = ConstantCoefficient,
          typename FES = FiniteElementSpace>
void test_dg_diffusion(FES &fes)
{
   using GF_t = typename ParTypeHelper<FES>::GF_t;
   using BLF_t = typename ParTypeHelper<FES>::BLF_t;

   GF_t x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   const int dim = fes.GetMesh()->Dimension();
   auto coeff = MakeCoeff<CoeffType>(dim);

   const real_t sigma = -1.0;
   const real_t kappa = 10.0;

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = irs.Get(fes.GetMesh()->GetTypicalFaceGeometry(),
                                       2*fes.GetMaxElementOrder());

   BLF_t blf_fa(&fes);
   blf_fa.AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*coeff, sigma, kappa));
   blf_fa.AddBdrFaceIntegrator(new DGDiffusionIntegrator(*coeff, sigma, kappa));
   (*blf_fa.GetFBFI())[0]->SetIntegrationRule(ir);
   (*blf_fa.GetBFBFI())[0]->SetIntegrationRule(ir);
   blf_fa.Assemble();
   blf_fa.Finalize();
   OperatorHandle A_fa;
   Array<int> empty;
   blf_fa.FormSystemMatrix(empty, A_fa);
   A_fa->Mult(x, y_fa);

   BLF_t blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddInteriorFaceIntegrator(
      new DGDiffusionIntegrator(*coeff, sigma, kappa));
   blf_pa.AddBdrFaceIntegrator(new DGDiffusionIntegrator(*coeff, sigma, kappa));
   (*blf_pa.GetFBFI())[0]->SetIntegrationRule(ir);
   (*blf_pa.GetBFBFI())[0]->SetIntegrationRule(ir);
   blf_pa.Assemble();
   blf_pa.Mult(x, y_pa);

   y_fa -= y_pa;

   REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
}

std::vector<std::string> get_dg_test_meshes()
{
   std::vector<std::string> mesh_filenames =
   {
      "../../data/star.mesh",
      "../../data/star-q3.mesh",
      "../../data/fichera.mesh",
      "../../data/fichera-q3.mesh",
   };
   const bool have_data_dir = mfem_data_dir != "";
   if (have_data_dir)
   {
      mesh_filenames.push_back(mfem_data_dir + "/gmsh/v22/unstructured_quad.v22.msh");
      mesh_filenames.push_back(mfem_data_dir + "/gmsh/v22/unstructured_hex.v22.msh");
   }
   return mesh_filenames;
}

TEST_CASE("PA DG Diffusion", "[PartialAssembly], [GPU]")
{
   const auto mesh_fname = GENERATE_COPY(from_range(get_dg_test_meshes()));
   const int order = GENERATE(1, 2);
   CAPTURE(order, mesh_fname);

   Mesh mesh = Mesh::LoadFromFile(mesh_fname.c_str());
   const int dim = mesh.Dimension();

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   test_dg_diffusion<ConstantCoefficient>(fes);
   test_dg_diffusion<MatrixConstantCoefficient>(fes);
   test_dg_diffusion<SymmetricMatrixConstantCoefficient>(fes);
}

#ifdef MFEM_USE_MPI

TEST_CASE("Parallel PA DG Diffusion", "[PartialAssembly][Parallel][GPU]")
{
   const auto mesh_fname = GENERATE_COPY(from_range(get_dg_test_meshes()));
   const int order = GENERATE(1, 2);
   CAPTURE(order, mesh_fname);

   Mesh serial_mesh = Mesh::LoadFromFile(mesh_fname.c_str());
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   const int dim = mesh.Dimension();

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   test_dg_diffusion<ConstantCoefficient>(fes);
   test_dg_diffusion<MatrixConstantCoefficient>(fes);
}

#endif

} // namespace pa_kernels

TEST_CASE("Dispatch Map Specializations")
{
   // The kernel specializations are registered the first time the associated
   // object is created (in the constructor of a static local variable in the
   // object's constructor). We create a dummy objects here to ensure that the
   // kernels are registered before testing.

   MassIntegrator{};
   REQUIRE_FALSE(MassIntegrator::ApplyPAKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(MassIntegrator::DiagonalPAKernels::GetDispatchTable().empty());

   DiffusionIntegrator{};
   REQUIRE_FALSE(
      DiffusionIntegrator::ApplyPAKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(
      DiffusionIntegrator::DiagonalPAKernels::GetDispatchTable().empty());

   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   H1_FECollection fec(1, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);
   fes.GetQuadratureInterpolator(IntRules.Get(mesh.GetElementGeometry(0), 1));

   using QI = QuadratureInterpolator;
   REQUIRE_FALSE(QI::TensorEvalKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::GradKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::DetKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::EvalKernels::GetDispatchTable().empty());
   REQUIRE_FALSE(QI::CollocatedGradKernels::GetDispatchTable().empty());
}
