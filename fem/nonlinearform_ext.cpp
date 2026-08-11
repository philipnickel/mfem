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

// Implementations of classes FABilinearFormExtension, EABilinearFormExtension,
// PABilinearFormExtension and MFBilinearFormExtension.

#include "nonlinearform.hpp"
#include "ceed/interface/util.hpp"
#include "../general/forall.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#endif

namespace mfem
{

NonlinearFormExtension::NonlinearFormExtension(const NonlinearForm *nlf)
   : Operator(nlf->FESpace()->GetVSize()), nlf(nlf) { }

PANonlinearFormExtension::PANonlinearFormExtension(const NonlinearForm *nlf):
   NonlinearFormExtension(nlf),
   fes(*nlf->FESpace()),
   dnfi(*nlf->GetDNFI()),
   fnfi(nlf->GetInteriorFaceIntegrators()),
   bfnfi(nlf->GetBdrFaceIntegrators()),
   bfnfi_marker(nlf->GetBdrFaceIntegratorMarkers()),
   elemR(nullptr),
   int_face_restriction(nullptr),
   bdr_face_restriction(nullptr),
   bdr_face_attributes(nullptr),
   Grad(*this)
{
   elemR = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   // Face kernels can require adjacent full-element data even when a CEED
   // backend owns the domain action, so retain this native restriction in all
   // backend configurations.
   xe.SetSize(elemR->Height(), Device::GetMemoryType());
   ye.SetSize(elemR->Height(), Device::GetMemoryType());
   ye.UseDevice(true);
   int_face_y.UseDevice(true);
   bdr_face_y.UseDevice(true);
   bdr_face_work.UseDevice(true);
}

real_t PANonlinearFormExtension::GetGridFunctionEnergy(const Vector &x) const
{
   real_t energy = 0.0;

   elemR->Mult(x, xe);
   for (int i = 0; i < dnfi.Size(); i++)
   {
      energy += dnfi[i]->GetLocalStateEnergyPA(xe);
   }
   return energy;
}

void PANonlinearFormExtension::Assemble()
{
   for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AssemblePA(fes); }

   if (fnfi.Size())
   {
      int_face_restriction = fes.GetFaceRestriction(
                                ElementDofOrdering::LEXICOGRAPHIC,
                                FaceType::Interior);
      int_face_x.SetSize(int_face_restriction->Height(), Device::GetMemoryType());
      int_face_y.SetSize(int_face_restriction->Height(), Device::GetMemoryType());
      for (int i = 0; i < fnfi.Size(); ++i)
      {
         fnfi[i]->AssemblePAInteriorFaces(fes);
      }
   }

   if (bfnfi.Size())
   {
      bdr_face_restriction = fes.GetFaceRestriction(
                                ElementDofOrdering::LEXICOGRAPHIC,
                                FaceType::Boundary,
                                L2FaceValues::DoubleValued);
      bdr_face_x.SetSize(bdr_face_restriction->Height(), Device::GetMemoryType());
      bdr_face_y.SetSize(bdr_face_restriction->Height(), Device::GetMemoryType());
      bdr_face_work.SetSize(bdr_face_restriction->Height(), Device::GetMemoryType());
      bdr_face_attributes = &fes.GetMesh()->GetBdrFaceAttributes();
      for (int i = 0; i < bfnfi.Size(); ++i)
      {
         bfnfi[i]->AssemblePABoundaryFaces(fes);
      }
   }
}

void PANonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   const bool use_ceed = DeviceCanUseCeed();
   const bool need_element_x = !use_ceed ||
      (int_face_restriction && fnfi.Size()) ||
      (bdr_face_restriction && bfnfi.Size());
   if (need_element_x) { elemR->Mult(x, xe); }

   if (!use_ceed)
   {
      ye = 0.0;
      for (int i = 0; i < dnfi.Size(); ++i) { dnfi[i]->AddMultPA(xe, ye); }
      elemR->MultTranspose(ye, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < dnfi.Size(); ++i)
      {
         dnfi[i]->AddMultPA(x, y);
      }
   }

   if (int_face_restriction && fnfi.Size())
   {
      const Vector *face_source = &x;
#ifdef MFEM_USE_MPI
      ParGridFunction parallel_source;
      if (auto *parallel_fes = dynamic_cast<ParFiniteElementSpace*>(
                                  const_cast<FiniteElementSpace*>(&fes)))
      {
         parallel_source.MakeRef(parallel_fes, const_cast<Vector&>(x), 0);
         face_source = &parallel_source;
      }
#endif
      int_face_restriction->Mult(*face_source, int_face_x);
      int_face_y = 0.0;
      for (int i = 0; i < fnfi.Size(); ++i)
      {
         fnfi[i]->AddMultPAFace(int_face_x, xe, int_face_y);
      }
      int_face_restriction->AddMultTransposeInPlace(int_face_y, y);
   }

   if (bdr_face_restriction && bfnfi.Size())
   {
      bdr_face_restriction->Mult(x, bdr_face_x);
      bdr_face_y = 0.0;
      for (int i = 0; i < bfnfi.Size(); ++i)
      {
         const Array<int> *marker = bfnfi_marker[i];
         if (!marker)
         {
            bfnfi[i]->AddMultPAFace(bdr_face_x, xe, bdr_face_y);
            continue;
         }
         bdr_face_work = 0.0;
         bfnfi[i]->AddMultPAFace(bdr_face_x, xe, *bdr_face_attributes,
                                 *marker, bdr_face_work);
         bdr_face_y += bdr_face_work;
      }
      bdr_face_restriction->AddMultTransposeInPlace(bdr_face_y, y);
   }
}

Operator &PANonlinearFormExtension::GetGradient(const Vector &x) const
{
   Grad.AssembleGrad(x);
   return Grad;
}

void PANonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   elemR = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   int_face_restriction = nullptr;
   bdr_face_restriction = nullptr;
   bdr_face_attributes = nullptr;
   xe.SetSize(elemR->Height());
   ye.SetSize(elemR->Height());
   int_face_x.SetSize(0);
   int_face_y.SetSize(0);
   bdr_face_x.SetSize(0);
   bdr_face_y.SetSize(0);
   bdr_face_work.SetSize(0);
   Grad.Update();
}

PANonlinearFormExtension::Gradient::Gradient(const PANonlinearFormExtension &e):
   Operator(e.Height()), ext(e)
{ }

void PANonlinearFormExtension::Gradient::AssembleGrad(const Vector &g)
{
   ext.elemR->Mult(g, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradPA(ext.xe, ext.fes);
   }
}

void PANonlinearFormExtension::Gradient::Mult(const Vector &x, Vector &y) const
{
   ext.ye = 0.0;
   ext.elemR->Mult(x, ext.xe);
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AddMultGradPA(ext.xe, ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, y);
}

void PANonlinearFormExtension::Gradient::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == Height(),
               "Vector for holding diagonal has wrong size!");
   ext.ye = 0.0;
   for (int i = 0; i < ext.dnfi.Size(); ++i)
   {
      ext.dnfi[i]->AssembleGradDiagonalPA(ext.ye);
   }
   ext.elemR->MultTranspose(ext.ye, diag);
}

void PANonlinearFormExtension::Gradient::Update()
{
   height = width = ext.Height();
}


MFNonlinearFormExtension::MFNonlinearFormExtension(const NonlinearForm *form):
   NonlinearFormExtension(form), fes(*form->FESpace())
{
   if (!DeviceCanUseCeed())
   {
      const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      elem_restrict_lex = fes.GetElementRestriction(ordering);
      if (elem_restrict_lex) // replace with a check for not identity
      {
         localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
         localY.UseDevice(true); // ensure 'localY = 0.0' is done on device
      }
   }
}

void MFNonlinearFormExtension::Assemble()
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int Ni = integrators.Size();
   for (int i = 0; i < Ni; ++i)
   {
      integrators[i]->AssembleMF(fes);
   }
}

void MFNonlinearFormExtension::Mult(const Vector &x, Vector &y) const
{
   const Array<NonlinearFormIntegrator*> &integrators = *nlf->GetDNFI();
   const int iSz = integrators.Size();
   // replace the check 'elem_restrict_lex' with a check for not identity
   if (elem_restrict_lex && !DeviceCanUseCeed())
   {
      elem_restrict_lex->Mult(x, localX);
      localY = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(localX, localY);
      }
      elem_restrict_lex->MultTranspose(localY, y);
   }
   else
   {
      y.UseDevice(true); // typically this is a large vector, so store on device
      y = 0.0;
      for (int i = 0; i < iSz; ++i)
      {
         integrators[i]->AddMultMF(x, y);
      }
   }
}

void MFNonlinearFormExtension::Update()
{
   height = width = fes.GetVSize();
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fes.GetElementRestriction(ordering);
   if (elem_restrict_lex) // replace with a check for not identity
   {
      localX.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      localY.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   }
}

} // namespace mfem
