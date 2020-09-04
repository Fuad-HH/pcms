#include "couplingTypes.h"
#include "dataprocess.h"
#include "importpart3mesh.h"
#include "commpart1.h"
#include "BoundaryDescr3D.h"

namespace coupler {

// routines for interpolation
template<class T>
T Lag3dInterpo1D(const T yin[4],const double xin[4],const double x)
{
  double l0,l1,l2,l3;
  T yout;
  l0=(x-xin[1])*(x-xin[2])*(x-xin[3])/(xin[0]-xin[1])/(xin[0]-xin[2])/(xin[0]-xin[3]);
  l1=(x-xin[0])*(x-xin[2])*(x-xin[3])/(xin[1]-xin[0])/(xin[1]-xin[2])/(xin[1]-xin[3]);
  l2=(x-xin[0])*(x-xin[1])*(x-xin[3])/(xin[2]-xin[0])/(xin[2]-xin[1])/(xin[2]-xin[3]);
  l3=(x-xin[0])*(x-xin[1])*(x-xin[2])/(xin[3]-xin[0])/(xin[3]-xin[1])/(xin[3]-xin[2]);
  yout=yin[0]*l0+yin[1]*l1+yin[2]*l2+yin[3]*l3;
  return yout;
}

//central 3rd order Lagrange polynormal interpolation
template<class T>
void Lag3dArray(const T* yin,const double* xin,const LO nin,T* yout,const double* xout,const LO nout){
  LO jstart=2;
  LO j1=jstart;
  LO j2,j0,jm;
  double x;
  T func[4];
  double coords[4];
  for(LO j=0;j<nout;j++){
    x=xout[j];
    while(x>=xin[j1] && j1<nin-2 && j1>1){
      j1=j1+1;
    }
    j2=j1+1;
    j0=j1-1;
    jm=j1-2;

    coords[0]=xin[jm];
    coords[1]=xin[j0];
    coords[2]=xin[j1];
    coords[3]=xin[j2];
    func[0]=yin[jm];
    func[1]=yin[j0];
    func[2]=yin[j1];
    func[3]=yin[j2];
    yout[j]=Lag3dInterpo1D(func,coords,x);
  }
}

void DatasProc3D::InterpoDensity3D(const BoundaryDescr3D &bdesc)
{
  CV* yin;
  double* xin;
  CV* yout;
  double* xout;
  LO nzb=bdesc.nzb;
  xin=new double[p1->lk0+2*nzb];
  yin=new CV[p1->lk0+2*nzb];  
  if(preproc==true){
    if(p1->periods[2]==1){
      if(p1->mype_z==0){
        for(LO l=0;l<nzb;l++){
          xin[l]=p1->pzcoords[p1->nz0-nzb+l]-2.0*cplPI;
          xin[p1->lk0+nzb+l]=p1->pzcoords[p1->lk2+l+1];          
        }
      } else if(p1->mype_z==p1->npz-1){
          for(LO l=0;l<nzb;l++){
            xin[l]=p1->pzcoords[p1->lk1-nzb+l];
            xin[p1->lk0+nzb+l]=p1->pzcoords[l]+2.0*cplPI;
          }        
      }else{
          for(LO l=0;l<nzb;l++){
            xin[l]=p1->pzcoords[p1->lk1-nzb+l];
            xin[p1->lk0+nzb+l]=p1->pzcoords[p1->lk2+l+1];
          }
      }
      for(LO k=0;k<p1->lk0;k++){  
        xin[nzb+k]=p1->pzcoords[p1->lk1+k];
      }      
    }

   for(LO i=0;i<p1->li0;i++){
      for(LO j=0;j<p1->lj0;j++){
        for(LO l=0;l<nzb;l++){
          yin[l]=bdesc.lowdenz[i][j][l];
          yin[p1->lk0+nzb+l]=bdesc.updenz[i][j][l];
        }
        for(LO k=0;k<p1->lk0;k++){  
          yin[nzb+k]=densin[i][j][k];
        }
        xout=p3->pzcoords[i];
        yout=densinterpo[i][j];
        Lag3dArray(yin,xin,p1->lk0+2*nzb,yout,xout,p3->mylk0[i]);
      }
    }   
  }
  delete[] xin,yin;
  xin=NULL;
  yin=NULL;
}


void DatasProc3D::InterpoPotential3D(const BoundaryDescr3D &bdesc)
{
  CV* yin;
  CV* yout;
  double* xin;
  LO nzb=bdesc.nzb;
  if(preproc==true){
    for(LO i=0;i<p3->li0;i++){
      yin=new CV[p3->mylk0[i]+2*nzb];
      xin=new double[p3->mylk0[i]+2*nzb];

      for(LO l=0;l<nzb;l++){  
        xin[l]=bdesc.lowzpart3[i][l];
        xin[p3->mylk0[i]+nzb+l]=bdesc.upzpart3[i][l];      
      }

      for(LO j=0;j<p3->lj0/2;j++){
        for(LO l=0;l<nzb;l++){
          yin[l]=bdesc.lowpotentz[i][j][l];
          yin[p3->mylk0[i]+nzb+l]=bdesc.uppotentz[i][j][l];
        }
        for(LO k=0;k<p3->mylk0[i];k++){
          xin[k+nzb]=p3->pzcoords[i][k];
          yin[k+nzb]=potentinterpo[i][j][k];
        }
        yout=potentpart1[i][j];
        Lag3dArray(yin,xin,p3->mylk0[i]+2*nzb,yout,p1->pzp,p1->lk0);
     }

     delete[] xin;
     delete[] yin; 
    }
  }

}

}
