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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/forall.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

SIPGeometryPenaltyOperator::SIPGeometryPenaltyOperator(
   ParMesh &mesh_, int degree_, const IntegrationRule &face_ir_,
   const IntegrationRule &element_ir_)
   : mesh(mesh_), degree(degree_), face_ir(face_ir_), element_ir(element_ir_)
{
   MFEM_VERIFY(degree >= 0, "SIP degree must be non-negative");
   FaceQuadratureSpace interior_space(mesh, face_ir, FaceType::Interior);
   FaceQuadratureSpace boundary_space(mesh, face_ir, FaceType::Boundary);
   interior_faces.SetSize(interior_space.GetNumFaces());
   boundary_faces.SetSize(boundary_space.GetNumFaces());
   interior_e1.SetSize(interior_faces.Size());
   interior_e2.SetSize(interior_faces.Size());
   boundary_e1.SetSize(boundary_faces.Size());
   for (int i = 0; i < interior_faces.Size(); ++i)
   {
      interior_faces[i] = interior_space.GetMeshFaceIndex(i);
      mesh.GetFaceElements(interior_faces[i], &interior_e1[i], &interior_e2[i]);
      MFEM_VERIFY(interior_e1[i] >= 0 && interior_e1[i] < mesh.GetNE(),
                  "interior SIP face has invalid local adjacency");
   }
   for (int i = 0; i < boundary_faces.Size(); ++i)
   {
      boundary_faces[i] = boundary_space.GetMeshFaceIndex(i);
      int e2;
      mesh.GetFaceElements(boundary_faces[i], &boundary_e1[i], &e2);
      MFEM_VERIFY(boundary_e1[i] >= 0 && boundary_e1[i] < mesh.GetNE() && e2 < 0,
                  "physical boundary SIP face has invalid adjacency");
   }

   p0_fec = std::make_unique<L2_FECollection>(
               0, mesh.Dimension(), BasisType::GaussLobatto);
   p0_fes = std::make_unique<ParFiniteElementSpace>(&mesh, p0_fec.get(), 1);
   MFEM_VERIFY(p0_fes->GetVSize() == mesh.GetNE(),
               "P0 SIP exchange space must have one dof per element");
   if (interior_faces.Size() > 0)
   {
      face_restriction = p0_fes->GetFaceRestriction(
                            ElementDofOrdering::NATIVE, FaceType::Interior);
      MFEM_VERIFY(face_restriction != nullptr,
                  "SIP geometry update requires an interior face restriction");
      MFEM_VERIFY(face_restriction->Height() == 2 * interior_faces.Size(),
                  "P0 face restriction ordering does not match interior faces");
   }
   local_p0.SetSize(p0_fes->GetVSize());
   restricted_p0.SetSize(2 * interior_faces.Size());
   element_penalties.SetSize(mesh.GetNE());
   face_penalties.SetSize(mesh.GetNumFaces());
   interior_quadrature_penalties.SetSize(
      interior_faces.Size() * face_ir.GetNPoints());
   boundary_quadrature_penalties.SetSize(
      boundary_faces.Size() * face_ir.GetNPoints());
   weighted_boundary.SetSize(mesh.GetNE());
   local_p0.UseDevice(true);
   restricted_p0.UseDevice(true);
   element_penalties.UseDevice(true);
   face_penalties.UseDevice(true);
   interior_quadrature_penalties.UseDevice(true);
   boundary_quadrature_penalties.UseDevice(true);
   weighted_boundary.UseDevice(true);
   Update();
}

void SIPGeometryPenaltyOperator::Update()
{
   const int ne = mesh.GetNE();
   weighted_boundary = 0.0;
   const MemoryType mt = Device::GetDeviceMemoryType();
   const int nqf = face_ir.GetNPoints();
   const real_t *Wf = face_ir.GetWeights().Read();
   real_t *A = weighted_boundary.ReadWrite();
   const int nif = interior_faces.Size();
   const FaceGeometricFactors *interior_geom = nullptr;
   const real_t *Ji = nullptr;
   if (nif > 0)
   {
      interior_geom = mesh.GetFaceGeometricFactors(
         face_ir, FaceGeometricFactors::DETERMINANTS, FaceType::Interior, mt);
      MFEM_VERIFY(interior_geom != nullptr && interior_geom->detJ.Size() == nqf * nif,
                  "interior SIP face geometry has the wrong size");
      Ji = interior_geom->detJ.Read();
   }
   const int *E1 = interior_e1.Read();
   const int *E2 = interior_e2.Read();
   mfem::forall(nif, [=] MFEM_HOST_DEVICE(int i)
   {
      real_t measure = 0.0;
      for (int q = 0; q < nqf; ++q)
      {
         measure += Wf[q] * Ji[q + nqf * i];
      }
      AtomicAdd(A[E1[i]], 0.5 * measure);
      if (E2[i] >= 0 && E2[i] < ne)
      {
         AtomicAdd(A[E2[i]], 0.5 * measure);
      }
   });

   const int nbf = boundary_faces.Size();
   const FaceGeometricFactors *boundary_geom = mesh.GetFaceGeometricFactors(
      face_ir, FaceGeometricFactors::DETERMINANTS, FaceType::Boundary, mt);
   MFEM_VERIFY(boundary_geom != nullptr && boundary_geom->detJ.Size() == nqf * nbf,
               "boundary SIP face geometry has the wrong size");
   const real_t *Jb = boundary_geom->detJ.Read();
   const int *BE1 = boundary_e1.Read();
   mfem::forall(nbf, [=] MFEM_HOST_DEVICE(int i)
   {
      real_t measure = 0.0;
      for (int q = 0; q < nqf; ++q)
      {
         measure += Wf[q] * Jb[q + nqf * i];
      }
      AtomicAdd(A[BE1[i]], measure);
   });

   const int nqe = element_ir.GetNPoints();
   const GeometricFactors *element_geom = mesh.GetGeometricFactors(
      element_ir, GeometricFactors::DETERMINANTS, mt);
   const real_t *We = element_ir.GetWeights().Read();
   const real_t *Je = element_geom->detJ.Read();
   const real_t degree_scale = (degree + 1) * (degree + 1);
   real_t *P = element_penalties.Write();
   real_t *P0 = local_p0.Write();
   mfem::forall(ne, [=] MFEM_HOST_DEVICE(int e)
   {
      real_t volume = 0.0;
      for (int q = 0; q < nqe; ++q)
      {
         volume += We[q] * fabs(Je[q + nqe * e]);
      }
      P[e] = degree_scale * A[e] / volume;
      P0[e] = P[e];
   });

   if (nif > 0) { face_restriction->Mult(local_p0, restricted_p0); }
   face_penalties = 0.0;
   const int *IF = interior_faces.Read();
   const real_t *RP0 = restricted_p0.Read();
   real_t *FP = face_penalties.ReadWrite();
   real_t *IQP = interior_quadrature_penalties.Write();
   mfem::forall(nif, [=] MFEM_HOST_DEVICE(int i)
   {
      const real_t penalty = fmax(RP0[2*i], RP0[2*i + 1]);
      FP[IF[i]] = penalty;
      for (int q = 0; q < nqf; ++q)
      {
         IQP[q + nqf * i] = penalty;
      }
   });
   const int *BF = boundary_faces.Read();
   real_t *BQP = boundary_quadrature_penalties.Write();
   mfem::forall(nbf, [=] MFEM_HOST_DEVICE(int i)
   {
      const real_t penalty = P[BE1[i]];
      FP[BF[i]] = penalty;
      for (int q = 0; q < nqf; ++q)
      {
         BQP[q + nqf * i] = penalty;
      }
   });
}

void SIPGeometryPenaltyOperator::FillInteriorQuadraturePenalties(
   Vector &values) const
{
   const int nf = interior_faces.Size();
   if (nf == 0)
   {
      MFEM_VERIFY(values.Size() == 0,
                  "empty interior SIP quadrature vector has the wrong size");
      return;
   }
   MFEM_VERIFY(values.Size() % nf == 0,
               "interior SIP quadrature vector has the wrong size");
   const int nq = values.Size() / nf;
   const int *F = interior_faces.Read();
   const real_t *P = face_penalties.Read();
   real_t *V = values.Write();
   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int f)
   {
      const real_t penalty = P[F[f]];
      for (int q = 0; q < nq; ++q)
      {
         V[q + nq * f] = penalty;
      }
   });
}

void SIPGeometryPenaltyOperator::FillBoundaryQuadraturePenalties(
   Vector &values) const
{
   const int nf = boundary_faces.Size();
   if (nf == 0)
   {
      MFEM_VERIFY(values.Size() == 0,
                  "empty boundary SIP quadrature vector has the wrong size");
      return;
   }
   MFEM_VERIFY(values.Size() % nf == 0,
               "boundary SIP quadrature vector has the wrong size");
   const int nq = values.Size() / nf;
   const int *F = boundary_faces.Read();
   const real_t *P = face_penalties.Read();
   real_t *V = values.Write();
   mfem::forall(nf, [=] MFEM_HOST_DEVICE(int f)
   {
      const real_t penalty = P[F[f]];
      for (int q = 0; q < nq; ++q)
      {
         V[q + nq * f] = penalty;
      }
   });
}

void ParBilinearForm::pAllocMat()
{
   int nbr_size = pfes->GetFaceNbrVSize();

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      if (keep_nbr_block)
      {
         mat = new SparseMatrix(height + nbr_size, width + nbr_size);
      }
      else
      {
         mat = new SparseMatrix(height, width + nbr_size);
      }
      return;
   }

   // the sparsity pattern is defined from the map: face->element->dof
   const Table &lelem_ldof = fes->GetElementToDofTable(); // <-- dofs
   const Table &nelem_ndof = pfes->face_nbr_element_dof; // <-- vdofs
   Table elem_dof; // element + nbr-element <---> dof
   if (nbr_size > 0)
   {
      // merge lelem_ldof and nelem_ndof into elem_dof
      int s1 = lelem_ldof.Size(), s2 = nelem_ndof.Size();
      const int *I1 = lelem_ldof.GetI(), *J1 = lelem_ldof.GetJ();
      const int *I2 = nelem_ndof.GetI(), *J2 = nelem_ndof.GetJ();
      const int nnz1 = I1[s1], nnz2 = I2[s2];

      elem_dof.SetDims(s1 + s2, nnz1 + nnz2);

      int *I = elem_dof.GetI(), *J = elem_dof.GetJ();
      for (int i = 0; i <= s1; i++)
      {
         I[i] = I1[i];
      }
      for (int j = 0; j < nnz1; j++)
      {
         J[j] = J1[j];
      }
      for (int i = 0; i <= s2; i++)
      {
         I[s1+i] = I2[i] + nnz1;
      }
      for (int j = 0; j < nnz2; j++)
      {
         J[nnz1+j] = J2[j] + height;
      }
   }
   //   dof_elem x  elem_face x face_elem x elem_dof  (keep_nbr_block = true)
   // ldof_lelem x lelem_face x face_elem x elem_dof  (keep_nbr_block = false)
   Table dof_dof;
   {
      Table face_dof; // face_elem x elem_dof
      {
         Table *face_elem = pfes->GetParMesh()->GetFaceToAllElementTable();
         if (nbr_size > 0)
         {
            mfem::Mult(*face_elem, elem_dof, face_dof);
         }
         else
         {
            mfem::Mult(*face_elem, lelem_ldof, face_dof);
         }
         delete face_elem;
         if (nbr_size > 0)
         {
            elem_dof.Clear();
         }
      }

      if (keep_nbr_block)
      {
         Table dof_face;
         Transpose(face_dof, dof_face, height + nbr_size);
         mfem::Mult(dof_face, face_dof, dof_dof);
      }
      else
      {
         Table ldof_face;
         {
            Table face_ldof;
            Table *face_lelem = fes->GetMesh()->GetFaceToElementTable();
            mfem::Mult(*face_lelem, lelem_ldof, face_ldof);
            delete face_lelem;
            Transpose(face_ldof, ldof_face, height);
         }
         mfem::Mult(ldof_face, face_dof, dof_dof);
      }
   }

   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();
   int nrows = dof_dof.Size();
   real_t *data = Memory<real_t>(I[nrows]);

   mat = new SparseMatrix(I, J, data, nrows, height + nbr_size);
   *mat = 0.0;

   dof_dof.LoseData();
}

void ParBilinearForm::ParallelRAP(SparseMatrix &loc_A, OperatorHandle &A,
                                  bool steal_loc_A)
{
   ParFiniteElementSpace &pfespace = *ParFESpace();

   // Create a block diagonal parallel matrix
   OperatorHandle A_diag(Operator::Hypre_ParCSR);
   A_diag.MakeSquareBlockDiag(pfespace.GetComm(),
                              pfespace.GlobalVSize(),
                              pfespace.GetDofOffsets(),
                              &loc_A);

   // Parallel matrix assembly using P^t A P (if needed)
   if (IsIdentityProlongation(pfespace.GetProlongationMatrix()))
   {
      A_diag.SetOperatorOwner(false);
      A.Reset(A_diag.As<HypreParMatrix>());
      if (steal_loc_A)
      {
         HypreStealOwnership(*A.As<HypreParMatrix>(), loc_A);
      }
   }
   else
   {
      OperatorHandle P(Operator::Hypre_ParCSR);
      P.ConvertFrom(pfespace.Dof_TrueDof_Matrix());
      A.MakePtAP(A_diag, P);
   }
}

HypreParMatrix *ParBilinearForm::ParallelAssembleInternalMatrix()
{
   if (p_mat.Ptr() == NULL)
   {
      ParallelAssemble(p_mat, mat);
   }
   return p_mat.As<HypreParMatrix>();
}

void ParBilinearForm::ParallelAssemble(OperatorHandle &A, SparseMatrix *A_local)
{
   A.Clear();

   if (A_local == NULL) { return; }
   MFEM_VERIFY(A_local->Finalized(), "the local matrix must be finalized");

   OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;

   if (interior_face_integs.Size() == 0)
   {
      // construct a parallel block-diagonal matrix 'A' based on 'a'
      dA.MakeSquareBlockDiag(pfes->GetComm(), pfes->GlobalVSize(),
                             pfes->GetDofOffsets(), A_local);
   }
   else
   {
      // handle the case when 'a' contains off-diagonal
      int lvsize = pfes->GetVSize();
      const HYPRE_BigInt *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
      HYPRE_BigInt ldof_offset = pfes->GetMyDofOffset();

      Array<HYPRE_BigInt> glob_J(A_local->NumNonZeroElems());
      int *J = A_local->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
      {
         if (J[i] < lvsize)
         {
            glob_J[i] = J[i] + ldof_offset;
         }
         else
         {
            glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
         }
      }

      // TODO - construct dA directly in the A format
      hdA.Reset(
         new HypreParMatrix(pfes->GetComm(), lvsize, pfes->GlobalVSize(),
                            pfes->GlobalVSize(), A_local->GetI(), glob_J,
                            A_local->GetData(), pfes->GetDofOffsets(),
                            pfes->GetDofOffsets()));
      // - hdA owns the new HypreParMatrix
      // - the above constructor copies all input arrays
      glob_J.DeleteAll();
      dA.ConvertFrom(hdA);
   }

   // TODO - assemble the Dof_TrueDof_Matrix directly in the required format?
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   // TODO: When Ph.Type() == Operator::ANY_TYPE we want to use the Operator
   // returned by pfes->GetProlongationMatrix(), however that Operator is a
   // const Operator, so we cannot store it in OperatorHandle. We need a const
   // version of class OperatorHandle, e.g. ConstOperatorHandle.

   A.MakePtAP(dA, Ph);
}

HypreParMatrix *ParBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   OperatorHandle Mh(Operator::Hypre_ParCSR);
   ParallelAssemble(Mh, m);
   Mh.SetOperatorOwner(false);
   return Mh.As<HypreParMatrix>();
}

void ParBilinearForm::AssembleSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   DenseMatrix elemmat;

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      int Elem2NbrNo = T->Elem2No - pmesh->GetNE();
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs2);
      vdofs1.Copy(vdofs_all);
      for (int j = 0; j < vdofs2.Size(); j++)
      {
         if (vdofs2[j] >= 0)
         {
            vdofs2[j] += height;
         }
         else
         {
            vdofs2[j] -= height;
         }
      }
      vdofs_all.Append(vdofs2);
      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k]->
         AssembleFaceMatrix(*pfes->GetFE(T->Elem1No),
                            *pfes->GetFaceNbrFE(Elem2NbrNo),
                            *T, elemmat);
         if (keep_nbr_block)
         {
            mat->AddSubMatrix(vdofs_all, vdofs_all, elemmat, skip_zeros);
         }
         else
         {
            mat->AddSubMatrix(vdofs1, vdofs_all, elemmat, skip_zeros);
         }
      }
   }
}

void ParBilinearForm::Assemble(int skip_zeros)
{
   if (interior_face_integs.Size())
   {
      pfes->ExchangeFaceNbrData();
      if (!ext && mat == NULL)
      {
         pAllocMat();
      }
   }

   BilinearForm::Assemble(skip_zeros);

   if (!ext && interior_face_integs.Size() > 0)
   {
      AssembleSharedFaces(skip_zeros);
   }
}

void ParBilinearForm::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == fes->GetTrueVSize(),
               "Vector for holding diagonal has wrong size!");
   const Operator *P = fes->GetProlongationMatrix();
   if (!ext)
   {
      MFEM_ASSERT(p_mat.Ptr(), "the ParBilinearForm is not assembled!");
      p_mat->AssembleDiagonal(diag); // TODO: add support for PETSc matrices
      return;
   }
   // Here, we have extension, ext.
   if (IsIdentityProlongation(P))
   {
      ext->AssembleDiagonal(diag);
      return;
   }
   // Here, we have extension, ext, and parallel/conforming prolongation, P.
   Vector local_diag(P->Height());
   ext->AssembleDiagonal(local_diag);
   if (fes->Conforming())
   {
      P->MultTranspose(local_diag, diag);
      return;
   }
   // For an AMR mesh, a convergent diagonal is assembled with |P^T| d_l,
   // where |P^T| has the entry-wise absolute values of the conforming
   // prolongation transpose operator.
   const HypreParMatrix *HP = dynamic_cast<const HypreParMatrix*>(P);
   if (HP)
   {
      HP->AbsMultTranspose(1.0, local_diag, 0.0, diag);
   }
   else
   {
      MFEM_ABORT("unsupported prolongation matrix type.");
   }
}

void ParBilinearForm
::ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                               HypreParMatrix &A, const HypreParVector &X,
                               HypreParVector &B) const
{
   Array<int> dof_list;

   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, dof_list);

   // do the parallel elimination
   A.EliminateRowsCols(dof_list, X, B);
}

void ParBilinearForm::ParallelEliminateEssentialBC(
   const Array<int> &bdr_attr_is_ess, const HypreParVector &X, HypreParVector &B)
{
   Array<int> dof_list;
   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, dof_list);

   p_mat.As<HypreParMatrix>()->EliminateRowsCols(dof_list, X, B);
}

HypreParMatrix *ParBilinearForm::
ParallelEliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                             HypreParMatrix &A) const
{
   Array<int> dof_list;

   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, dof_list);

   return A.EliminateRowsCols(dof_list);
}

void ParBilinearForm::ParallelEliminateEssentialBC(const Array<int>
                                                   &bdr_attr_is_ess)
{
   Array<int> tdofs_list;
   pfes->GetEssentialTrueDofs(bdr_attr_is_ess, tdofs_list);

   ParallelEliminateTDofs(tdofs_list);
}

void ParBilinearForm::ParallelEliminateTDofs(const Array<int> &tdofs_list)
{
   p_mat_e.EliminateRowsCols(p_mat, tdofs_list);
}

void ParBilinearForm::ParallelEliminateTDofsInRHS(
   const Array<int> &tdofs_list, const Vector &x, Vector &b)
{
   p_mat.EliminateBC(p_mat_e, tdofs_list, x, b);
}

void ParBilinearForm::TrueAddMult(const Vector &x, Vector &y, const real_t a)
const
{
   const Operator *P = pfes->GetProlongationMatrix();
   Xaux.SetSize(P->Height());
   Yaux.SetSize(P->Height());
   Ytmp.SetSize(P->Width());

   P->Mult(x, Xaux);
   if (ext)
   {
      ext->Mult(Xaux, Yaux);
   }
   else
   {
      MFEM_VERIFY(interior_face_integs.Size() == 0,
                  "the case of interior face integrators is not"
                  " implemented");
      mat->Mult(Xaux, Yaux);
   }
   P->MultTranspose(Yaux, Ytmp);
   y.Add(a, Ytmp);
}

real_t ParBilinearForm::ParInnerProduct(const ParGridFunction &x,
                                        const ParGridFunction &y) const
{
   MFEM_ASSERT(mat != NULL, "local matrix must be assembled");

   real_t loc = InnerProduct(x, y);
   real_t glob = 0.;

   MPI_Allreduce(&loc, &glob, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 pfes->GetComm());

   return glob;
}

real_t ParBilinearForm::TrueInnerProduct(const ParGridFunction &x,
                                         const ParGridFunction &y) const
{
   MFEM_ASSERT(x.ParFESpace() == pfes, "the parallel spaces must match");
   MFEM_ASSERT(y.ParFESpace() == pfes, "the parallel spaces must match");

   HypreParVector *x_p = x.ParallelProject();
   HypreParVector *y_p = y.ParallelProject();

   real_t res = TrueInnerProduct(*x_p, *y_p);

   delete x_p;
   delete y_p;

   return res;
}

real_t ParBilinearForm::TrueInnerProduct(HypreParVector &x,
                                         HypreParVector &y) const
{
   MFEM_VERIFY(p_mat.Ptr() != NULL, "parallel matrix must be assembled");

   if (p_mat->GetType() != Operator::Hypre_ParCSR)
   {
      return TrueInnerProduct((const Vector&)x, (const Vector&)y);
   }

   HypreParVector *Ax = new HypreParVector(pfes);
   HypreParMatrix *A = p_mat.As<HypreParMatrix>();

   A->Mult(x, *Ax);

   real_t res = mfem::InnerProduct(y, *Ax);

   delete Ax;

   return res;
}

real_t ParBilinearForm::TrueInnerProduct(const Vector &x,
                                         const Vector &y) const
{
   MFEM_VERIFY(p_mat.Ptr() != NULL, "parallel matrix must be assembled");

   Vector Ax(pfes->GetTrueVSize());
   p_mat->Mult(x, Ax);

   real_t res = mfem::InnerProduct(pfes->GetComm(), y, Ax);

   return res;
}

void ParBilinearForm::FormLinearSystem(
   const Array<int> &ess_tdof_list, Vector &x, Vector &b,
   OperatorHandle &A, Vector &X, Vector &B, int copy_interior)
{
   const Operator &P = *pfes->GetProlongationMatrix();
   const SparseMatrix &R = *pfes->GetRestrictionMatrix();

   if (ext)
   {
      if (hybridization)
      {
         HypreParVector true_X(pfes), true_B(pfes);
         P.MultTranspose(b, true_B);
         R.Mult(x, true_X);

         FormSystemMatrix(ess_tdof_list, A);
         ConstrainedOperator *A_constrained;
         Operator::FormConstrainedSystemOperator(ess_tdof_list, A_constrained);
         A_constrained->EliminateRHS(true_X, true_B);
         delete A_constrained;
         R.MultTranspose(true_B, b);
         hybridization->ReduceRHS(true_B, B);
         X.SetSize(B.Size());
         X = 0.0;
      }
      else
      {
         ext->FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      }
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   FormSystemMatrix(ess_tdof_list, A);

   // Transform the system and perform the elimination in B, based on the
   // essential BC values from x. Restrict the BC part of x in X, and set the
   // non-BC part to zero. Since there is no good initial guess for the Lagrange
   // multipliers, set X = 0.0 for hybridization.
   if (static_cond)
   {
      // Schur complement reduction to the exposed dofs
      static_cond->ReduceSystem(x, b, X, B, copy_interior);
   }
   else if (hybridization)
   {
      // Reduction to the Lagrange multipliers system
      HypreParVector true_X(pfes), true_B(pfes);
      P.MultTranspose(b, true_B);
      R.Mult(x, true_X);
      ParallelEliminateTDofsInRHS(ess_tdof_list, true_X, true_B);
      R.MultTranspose(true_B, b);
      hybridization->ReduceRHS(true_B, B);
      X.SetSize(B.Size());
      X = 0.0;
   }
   else
   {
      // Variational restriction with P
      X.SetSize(P.Width());
      B.SetSize(X.Size());
      P.MultTranspose(b, B);
      R.Mult(x, X);
      ParallelEliminateTDofsInRHS(ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
}

void ParBilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                       OperatorHandle &A)
{
   if (ext)
   {
      if (hybridization)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         hybridization->GetParallelMatrix(A);
      }
      else
      {
         ext->FormSystemMatrix(ess_tdof_list, A);
      }
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   if (static_cond)
   {
      if (!static_cond->HasEliminatedBC())
      {
         static_cond->SetEssentialTrueDofs(ess_tdof_list);
         static_cond->Finalize();
         static_cond->EliminateReducedTrueDofs(Matrix::DIAG_ONE);
      }
      static_cond->GetParallelMatrix(A);
   }
   else
   {
      if (mat)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         MFEM_VERIFY(p_mat.Ptr() == NULL && p_mat_e.Ptr() == NULL,
                     "The ParBilinearForm must be updated with Update() before "
                     "re-assembling the ParBilinearForm.");
         ParallelAssemble(p_mat, mat);
         delete mat;
         mat = NULL;
         delete mat_e;
         mat_e = NULL;
         ParallelEliminateTDofs(ess_tdof_list);
      }
      if (hybridization)
      {
         hybridization->GetParallelMatrix(A);
      }
      else
      {
         A = p_mat;
      }
   }
}

void ParBilinearForm::RecoverFEMSolution(
   const Vector &X, const Vector &b, Vector &x)
{
   if (ext && !hybridization)
   {
      ext->RecoverFEMSolution(X, b, x);
      return;
   }

   const Operator &P = *pfes->GetProlongationMatrix();

   if (static_cond)
   {
      // Private dofs back solve
      static_cond->ComputeSolution(b, X, x);
   }
   else if (hybridization)
   {
      // Primal unknowns recovery
      HypreParVector true_X(pfes), true_B(pfes);
      P.MultTranspose(b, true_B);
      const SparseMatrix &R = *pfes->GetRestrictionMatrix();
      R.Mult(x, true_X); // get essential b.c. from x
      hybridization->ComputeSolution(true_B, X, true_X);
      x.SetSize(P.Height());
      P.Mult(true_X, x);
   }
   else
   {
      // Apply conforming prolongation
      x.SetSize(P.Height(), GetHypreMemoryType());
      P.Mult(X, x);
   }
}

void ParBilinearForm::Update(FiniteElementSpace *nfes)
{
   BilinearForm::Update(nfes);

   if (nfes)
   {
      pfes = dynamic_cast<ParFiniteElementSpace *>(nfes);
      MFEM_VERIFY(pfes != NULL, "nfes must be a ParFiniteElementSpace!");
   }

   p_mat.Clear();
   p_mat_e.Clear();
}

void ParMixedBilinearForm::pAllocMat()
{
   const int trial_nbr_size = trial_pfes->GetFaceNbrVSize();
   const int test_nbr_size = test_pfes->GetFaceNbrVSize();

   if (keep_nbr_block)
   {
      mat = new SparseMatrix(height + test_nbr_size, width + trial_nbr_size);
   }
   else
   {
      mat = new SparseMatrix(height, width + trial_nbr_size);
   }
}

void ParMixedBilinearForm::AssembleSharedFaces(int skip_zeros)
{
   ParMesh *pmesh = trial_pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> tr_vdofs1, tr_vdofs2, tr_vdofs_all;
   Array<int> te_vdofs1, te_vdofs2, te_vdofs_all;
   DenseMatrix elemmat;

   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      int Elem2NbrNo = T->Elem2No - pmesh->GetNE();
      trial_pfes->GetElementVDofs(T->Elem1No, tr_vdofs1);
      test_pfes->GetElementVDofs(T->Elem1No, te_vdofs1);
      trial_pfes->GetFaceNbrElementVDofs(Elem2NbrNo, tr_vdofs2);
      test_pfes->GetFaceNbrElementVDofs(Elem2NbrNo, te_vdofs2);

      tr_vdofs1.Copy(tr_vdofs_all);
      for (int j = 0; j < tr_vdofs2.Size(); j++)
      {
         if (tr_vdofs2[j] >= 0)
         {
            tr_vdofs2[j] += width;
         }
         else
         {
            tr_vdofs2[j] -= width;
         }
      }
      tr_vdofs_all.Append(tr_vdofs2);

      if (keep_nbr_block)
      {
         te_vdofs1.Copy(te_vdofs_all);
         for (int j = 0; j < te_vdofs2.Size(); j++)
         {
            if (te_vdofs2[j] >= 0)
            {
               te_vdofs2[j] += height;
            }
            else
            {
               te_vdofs2[j] -= height;
            }
         }
         te_vdofs_all.Append(te_vdofs2);
      }

      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k]->
         AssembleFaceMatrix(*trial_pfes->GetFE(T->Elem1No),
                            *test_pfes->GetFE(T->Elem1No),
                            *trial_pfes->GetFaceNbrFE(Elem2NbrNo),
                            *test_pfes->GetFaceNbrFE(Elem2NbrNo),
                            *T, elemmat);
         if (keep_nbr_block)
         {
            mat->AddSubMatrix(te_vdofs_all, tr_vdofs_all, elemmat, skip_zeros);
         }
         else
         {
            mat->AddSubMatrix(te_vdofs1, tr_vdofs_all, elemmat, skip_zeros);
         }
      }
   }
}

void ParMixedBilinearForm::Assemble(int skip_zeros)
{
   if (interior_face_integs.Size())
   {
      trial_pfes->ExchangeFaceNbrData();
      test_pfes->ExchangeFaceNbrData();
      if (!ext && mat == NULL)
      {
         pAllocMat();
      }
   }

   MixedBilinearForm::Assemble(skip_zeros);

   if (!ext && interior_face_integs.Size() > 0)
   {
      AssembleSharedFaces(skip_zeros);
   }
}

HypreParMatrix *ParMixedBilinearForm::ParallelAssembleInternalMatrix()
{
   if (p_mat.Ptr() == NULL)
   {
      ParallelAssemble(p_mat, mat);
   }
   return p_mat.As<HypreParMatrix>();
}

HypreParMatrix *ParMixedBilinearForm::ParallelAssemble(SparseMatrix *m)
{
   OperatorHandle Mh(Operator::Hypre_ParCSR);
   ParallelAssemble(Mh, m);
   Mh.SetOperatorOwner(false);
   return Mh.As<HypreParMatrix>();
}

void ParMixedBilinearForm::ParallelAssemble(OperatorHandle &A,
                                            SparseMatrix *A_local)
{
   A.Clear();

   if (A_local == NULL) { return; }
   MFEM_VERIFY(A_local->Finalized(), "the local matrix must be finalized");

   OperatorHandle dA(A.Type()), hdA;

   if (interior_face_integs.Size() == 0)
   {
      // construct the rectangular block-diagonal matrix dA
      dA.MakeRectangularBlockDiag(trial_pfes->GetComm(),
                                  test_pfes->GlobalVSize(),
                                  trial_pfes->GlobalVSize(),
                                  test_pfes->GetDofOffsets(),
                                  trial_pfes->GetDofOffsets(),
                                  A_local);
   }
   else
   {
      // handle the case when 'a' contains off-diagonal
      const int lvrows = test_pfes->GetVSize();
      const int lvcols = trial_pfes->GetVSize();
      const HYPRE_BigInt *face_nbr_glob_lcol = trial_pfes->GetFaceNbrGlobalDofMap();
      const HYPRE_BigInt lcol_offset = trial_pfes->GetMyDofOffset();

      Array<HYPRE_BigInt> glob_J(A_local->NumNonZeroElems());
      const int *J = A_local->GetJ();
      for (int i = 0; i < glob_J.Size(); i++)
      {
         if (J[i] < lvcols)
         {
            glob_J[i] = J[i] + lcol_offset;
         }
         else
         {
            glob_J[i] = face_nbr_glob_lcol[J[i] - lvcols];
         }
      }

      // TODO - construct dA directly in the A format
      hdA.Reset(
         new HypreParMatrix(trial_pfes->GetComm(), lvrows, test_pfes->GlobalVSize(),
                            trial_pfes->GlobalVSize(), A_local->GetI(), glob_J,
                            A_local->GetData(), test_pfes->GetDofOffsets(),
                            trial_pfes->GetDofOffsets()));
      // - hdA owns the new HypreParMatrix
      // - the above constructor copies all input arrays
      glob_J.DeleteAll();
      dA.ConvertFrom(hdA);
   }

   OperatorHandle P_test(A.Type()), P_trial(A.Type());

   // TODO - construct the Dof_TrueDof_Matrix directly in the required format.
   P_test.ConvertFrom(test_pfes->Dof_TrueDof_Matrix());
   P_trial.ConvertFrom(trial_pfes->Dof_TrueDof_Matrix());

   A.MakeRAP(P_test, dA, P_trial);
}

/// Compute y += a (P^t A P) x, where x and y are vectors on the true dofs
void ParMixedBilinearForm::TrueAddMult(const Vector &x, Vector &y,
                                       const real_t a) const
{
   if (Xaux.ParFESpace() != trial_pfes)
   {
      Xaux.SetSpace(trial_pfes);
      Yaux.SetSpace(test_pfes);
   }

   Xaux.Distribute(&x);
   mat->Mult(Xaux, Yaux);
   test_pfes->Dof_TrueDof_Matrix()->MultTranspose(a, Yaux, 1.0, y);
}

void ParMixedBilinearForm::ParallelEliminateTrialEssentialBC(
   const Array<int> &bdr_attr_is_ess)
{
   Array<int> trial_tdof_list;
   trial_pfes->GetEssentialTrueDofs(bdr_attr_is_ess, trial_tdof_list);

   ParallelEliminateTrialTDofs(trial_tdof_list);
}

void ParMixedBilinearForm::ParallelEliminateTrialTDofs(
   const Array<int> &trial_tdof_list)
{
   HypreParMatrix *temp = p_mat.As<HypreParMatrix>()->EliminateCols(
                             trial_tdof_list);
   p_mat_e.Reset(temp, true);
}

void ParMixedBilinearForm::ParallelEliminateTrialTDofsInRHS(
   const Array<int> &trial_tdof_list, const Vector &x, Vector &b)
{
   p_mat_e.As<HypreParMatrix>()->Mult(-1.0, x, 1.0, b);
}

void ParMixedBilinearForm::ParallelEliminateTestEssentialBC(
   const Array<int> &bdr_attr_is_ess)
{
   Array<int> test_tdof_list;
   test_pfes->GetEssentialTrueDofs(bdr_attr_is_ess, test_tdof_list);

   ParallelEliminateTestTDofs(test_tdof_list);
}

void ParMixedBilinearForm::ParallelEliminateTestTDofs(
   const Array<int> &test_tdof_list)
{
   p_mat.As<HypreParMatrix>()->EliminateRows(test_tdof_list);
}

void ParMixedBilinearForm::FormRectangularSystemMatrix(
   const Array<int>
   &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   if (ext)
   {
      ext->FormRectangularSystemOperator(trial_tdof_list, test_tdof_list, A);
      return;
   }

   if (mat)
   {
      Finalize();
      ParallelAssemble(p_mat);
      delete mat;
      mat = NULL;
      delete mat_e;
      mat_e = NULL;
      ParallelEliminateTrialTDofs(trial_tdof_list);
      ParallelEliminateTestTDofs(test_tdof_list);
   }

   A = p_mat;
}

void ParMixedBilinearForm::FormRectangularLinearSystem(
   const Array<int>
   &trial_tdof_list,
   const Array<int> &test_tdof_list, Vector &x,
   Vector &b, OperatorHandle &A, Vector &X,
   Vector &B)
{
   if (ext)
   {
      ext->FormRectangularLinearSystem(trial_tdof_list, test_tdof_list,
                                       x, b, A, X, B);
      return;
   }

   FormRectangularSystemMatrix(trial_tdof_list, test_tdof_list, A);

   const Operator *test_P = test_pfes->GetProlongationMatrix();
   const SparseMatrix *trial_R = trial_pfes->GetRestrictionMatrix();

   X.SetSize(trial_pfes->TrueVSize());
   B.SetSize(test_pfes->TrueVSize());
   test_P->MultTranspose(b, B);
   trial_R->Mult(x, X);

   ParallelEliminateTrialTDofsInRHS(trial_tdof_list, X, B);
   B.SetSubVector(test_tdof_list, 0.0);
}

HypreParMatrix* ParDiscreteLinearOperator::ParallelAssemble() const
{
   MFEM_ASSERT(mat, "Matrix is not assembled");
   MFEM_ASSERT(mat->Finalized(), "Matrix is not finalized");
   SparseMatrix* RA = mfem::Mult(*range_fes->GetRestrictionMatrix(), *mat);
   HypreParMatrix* P = domain_fes->Dof_TrueDof_Matrix();
   HypreParMatrix* RAP = P->LeftDiagMult(*RA, range_fes->GetTrueDofOffsets());
   delete RA;
   return RAP;
}

void ParDiscreteLinearOperator::ParallelAssemble(OperatorHandle &A)
{
   // construct the rectangular block-diagonal matrix dA
   OperatorHandle dA(A.Type());
   dA.MakeRectangularBlockDiag(domain_fes->GetComm(),
                               range_fes->GlobalVSize(),
                               domain_fes->GlobalVSize(),
                               range_fes->GetDofOffsets(),
                               domain_fes->GetDofOffsets(),
                               mat);

   SparseMatrix *Rt = Transpose(*range_fes->GetRestrictionMatrix());
   OperatorHandle R_test_transpose(A.Type());
   R_test_transpose.MakeRectangularBlockDiag(range_fes->GetComm(),
                                             range_fes->GlobalVSize(),
                                             range_fes->GlobalTrueVSize(),
                                             range_fes->GetDofOffsets(),
                                             range_fes->GetTrueDofOffsets(),
                                             Rt);

   // TODO - construct the Dof_TrueDof_Matrix directly in the required format.
   OperatorHandle P_trial(A.Type());
   P_trial.ConvertFrom(domain_fes->Dof_TrueDof_Matrix());

   A.MakeRAP(R_test_transpose, dA, P_trial);
   delete Rt;
}

void ParDiscreteLinearOperator::FormRectangularSystemMatrix(OperatorHandle &A)
{
   if (ext)
   {
      Array<int> empty;
      ext->FormRectangularSystemOperator(empty, empty, A);
      return;
   }

   mfem_error("not implemented!");
}

void ParDiscreteLinearOperator::GetParBlocks(Array2D<HypreParMatrix *> &blocks)
const
{
   MFEM_VERIFY(mat->Finalized(), "Local matrix needs to be finalized for "
               "GetParBlocks");

   HypreParMatrix* RLP = ParallelAssemble();

   blocks.SetSize(range_fes->GetVDim(), domain_fes->GetVDim());

   RLP->GetBlocks(blocks,
                  range_fes->GetOrdering() == Ordering::byVDIM,
                  domain_fes->GetOrdering() == Ordering::byVDIM);

   delete RLP;
}

}

#endif
