#include "adios2Routines.h"
#include "couplingTypes.h"
#include "dataprocess.h"
#include "importpart3mesh.h"
#include "commpart1.h"
#include "BoundaryDescr3D.h"
#include "testutilities.h"
#include <string>
#include <fstream>

int main(int argc, char **argv){
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::string dir = "../coupling";
  const int time_step = atoi(argv[1]), RK_count = 4;

  fprintf(stderr, "%d number of time steps: %d\n", rank, time_step);

  const std::string xmlfile = "adios2cfg.xml";
  adios2::ADIOS adios(MPI_COMM_WORLD, adios2::DebugON);
  adios2::Variable<double> senddensity;
  adios2::Variable<coupler::CV> sendfield;

  coupler::adios2_handler gDens(adios,"density");
  coupler::adios2_handler cDens(adios,"cpl_density");
  coupler::adios2_handler xFld(adios,"xgc_field");
  coupler::adios2_handler cFld(adios,"cpl_field");
  coupler::adios2_handler gMesh(adios,"gem_mesh");
  coupler::adios2_handler gThf(adios,"gem_thfl");
  coupler::adios2_handler gGrd(adios,"gem_grid");
  coupler::adios2_handler gQprof(adios,"gem_qprf");
  coupler::adios2_handler xCouple(adios,"xgc_mesh");
  coupler::adios2_handler xRzcoords(adios,"xgc_rzcoords");  

  std::string model="global";
  const int m =0;
  coupler::GO start[2] = {0,0};
  const coupler::Array1d<int>* gmesh=coupler::receive_pproc<int>(dir,gMesh,model);
  coupler::GO ntheta = (coupler::GO)gmesh->val(4);
  coupler::GO nr = (coupler::GO)gmesh->val(1);
/*
  coupler::GO count[2] = {2,(coupler::GO)gmesh->val(5)};
  coupler::GO count2[2] = {0,10};
*/
  const coupler::Array1d<double>* thfl_qprof=coupler::receive_pproc<double>(dir,gQprof,model);
  for(coupler::LO i=0; i<2; i++) if(!rank) fprintf(stderr,"array[%d]: %f\n",i,thfl_qprof->val(i));

  coupler::Array1d<double>* rzcoords = coupler::receive_pproc<double>(dir,gGrd,model);
  coupler::Array1d<coupler::LO>* xcouple = coupler::receive_pproc<coupler::LO>(dir, xCouple, model);
  
  //intialize GEM class
  const bool preproc = true;
  const bool ypar = false;
  coupler::TestCase test_case = coupler::TestCase::off;
  coupler::CouplingCase ccase = coupler::CouplingCase::gemxgc; 
  
  coupler::Part1ParalPar3D p1pp3d(gmesh, thfl_qprof, test_case, preproc);
/*
  coupler::Part3Mesh3D p3m3d(xcouple, rzcoords, preproc, test_case);
  coupler::BoundaryDescr3D bdesc(p3m3d, p1pp3d, ccase, test_case, preproc);  
 
  coupler::Part1ParalPar3D* mesh1 = &p1pp3d;
  coupler::Part3Mesh3D*     mesh3 = &p3m3d;
  coupler::BoundaryDescr3D* bound = &bdesc;
  coupler::gemXgcDatasProc3D  gxdp3d(mesh1,mesh3,bound,preproc,test_case,ypar);
*/
/*
  int m;
  coupler::GO start[3];
  coupler::GO count[3];
*/
  for (int i = 0; i < time_step; i++) {
    for (int j = 0; j < RK_count; j++) {
//      m = i*RK_count+j;
       
 
    }
  }

/*  
  gDens.close();
  cDens.close();
  xFld.close();
  cFld.close();
*/
  gMesh.close();
//  gThf.close();
  gGrd.close();
  gQprof.close();
  xCouple.close();
//  xRzcoords.close();

  MPI_Finalize();
  printf("MPI is finalized.\n");
  return 0;
}
