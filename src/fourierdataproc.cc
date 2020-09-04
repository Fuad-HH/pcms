#include "dataprocess.h"
#include "commpart1.h"
#include "importpart3mesh.h"
#include "couplingTypes.h"
#include <iostream>

namespace coupler {

// read densin and potentin routines will be written here. 

// Beforing invoking this routine, the FFtw plan routine must be invoked. 
void DatasProc3D::CmplxdataToRealdata3D()
{
  if(densintmp==NULL){
    std::cout<<"ERROR: before invoking CmplxdataToRealdata3D, DatasProc3.densintmp must be allocated"<<'\n';
    std::exit(EXIT_FAILURE);
  }
  if(densouttmp==NULL){
    std::cout<<"ERROR: before invoking CmplxdataToRealdata3D, DatasProc3.densouttmp must be allocated"<<'\n';
    std::exit(EXIT_FAILURE);
  }
  if(yparal==false){
    for(LO i=0; i<p1->li0;i++){
      for(LO k=0;k<p3->mylk0[i];k++){
	for(LO j=0;j<p1->lj0;j++){
          densintmp[j]=densinterpo[i][j][k];
	}      
        ExecuteCmplxToReal();
        for(LO l=0;l<p1->lj0*2;l++){ 
          denspart3[i][l][k]=densouttmp[l]/double(p1->lj0*2);             }
      }
    }
  }
}

void DatasProc3D::RealdataToCmplxdata3D()
{
  if(potentintmp==NULL){
    std::cout<<"ERROR: before invoking CmplxdataToRealdata3, DatasProc3D.potentintmp must be allocated"<<'\n';
    std::exit(EXIT_FAILURE);
  }
  if(potentouttmp==NULL){
    std::cout<<"ERROR: before invoking CmplxdataToRealdata3,DatasProc3D.potentouttmp must be allocated"<<'\n';
    std::exit(EXIT_FAILURE);
  }
  if(yparal==false){
    for(LO i=0;i<p3->li0;i++){
      for(LO k=0;k<p1->lk0;k++){
	for(LO j=0;j<p1->y_res_back;j++){
	  potentintmp[j]=potentin[i][j][k];  
        }
        ExecuteRealToCmplx();
        for(LO l=0;l<p3->lj0/2;l++){
          potentpart1[i][l][k]=potentouttmp[l]/double(p1->y_res_back);       
        }
      }
    }
  }
}

//TODO Incomplete, Not used, put into the class it supports, I assume DatasProc3D
// This routine is not required in first verion of coupler, but would be modifed 
// for in the 2nd version. Here, the indexes may need exchange. 
void TransposeComplex(CV** InMatrix,CV** OutMatrix, DatasProc3D& dp3d,
     Part1ParalPar3D& p1pp3d)
{
  LO ny0 = dp3d.getP1ny0();
  LO npy = dp3d.getP1npy();

  CV*** sbuf;
  CV*** rbuf;
  sbuf=new CV**[ny0];
  rbuf=new CV**[ny0];
  for(int i=0;i<ny0;i++){
    sbuf[i]=new CV*[dp3d.part1li0];
    rbuf[i]=new CV*[dp3d.part1li0];
    for(int j=0;j<dp3d.part1li0;j++){
      sbuf[i][j]=new CV[npy];
      rbuf[i][j]=new CV[npy];
    }
  }
}
// It's not finished for yparal==true here.
void DatasProc3D::ExecuteCmplxToReal()
{
   if(yparal==true){
     CV** tmp_cmplx;
     tmp_cmplx=new CV*[p1->ny0];
     double** tmp_re; 
     tmp_re=new double*[2*p1->ny0*p1->res_fact]; 
     for(LO i=0;i<p1->ny0;i++)
       tmp_cmplx[i]=new CV[myli0]; 
     for(LO j=0;j<2*p1->ny0*p1->res_fact;j++)
       tmp_re[j]=new double[myli0];
     // Here is not finished for yparal=true

    }else{
     fftw_execute(plan_backward);
    } 
 }

void DatasProc3D::ExecuteRealToCmplx()
{
   if(yparal==true){
     // Here is not finished for yparal=true
   } else{
     fftw_execute(plan_forward);
   }

 } 

void DatasProc3D::InitFourierPlan3D()
{ 
  if(yparal==false){
    plan_backward=fftw_plan_dft_c2r_1d(p1->lj0*2,
                       reinterpret_cast<fftw_complex*>(densintmp),densouttmp,FFTW_ESTIMATE); 
    plan_forward=fftw_plan_dft_r2c_1d(p1->y_res_back,potentintmp,
                      reinterpret_cast<fftw_complex*>(potentouttmp),FFTW_ESTIMATE);
  }
}

void DatasProc3D::FreeFourierPlan3D()
{
  if(plan_forward)
    fftw_destroy_plan(plan_forward);
  if(plan_backward)
    fftw_destroy_plan(plan_backward);  
// This commeted part will be  implememted to releaste the memory during the process.
//  delete[] dp3d.densintmp;
/*  dp3d.densintmp=NULL;
  delete[] dp3d.densouttmp;
  dp3d.densintmp=NULL;
  delete[] dp3d.potentintmp;
  dp3d.potentintmp=NULL;
  delete[] dp3d.potentouttmp;
  dp3d.potentouttmp=NULL;
*/
}

}  
