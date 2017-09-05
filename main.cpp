#include <iostream>
#include "readBMP.h"
#include <math.h>
#include <stdlib.h>
#include <boost/thread.hpp>
#include <GL/glut.h>
#include "sep_reader.h"

#define TRAIN

SEPReader seismic_reader("/media/antonk/FreeAgent Drive/OpendTectData/Data/oxy/oxy.hdr");
SEPReader   fault_reader("/home/antonk/SmartAFI/git/SmartAFI/out_pick");

int o_x = seismic_reader.o3;
int o_y = seismic_reader.o2;
int o_z = seismic_reader.o1;

int n_x = seismic_reader.n3;
int n_y = seismic_reader.n2;
int n_z = seismic_reader.n1;

float * seismic_arr = new float[n_x*n_y*n_z];
float *   fault_arr = new float[n_x*n_y*n_z];

std::vector<float> errs;

int WIN=16;

float * vis_preview = new float[2*WIN*WIN];
float * vis0_preview = new float[2*WIN*WIN];

void vis2hid_worker(const float * X,float * H,int h,int v,float * c,float * W,std::vector<int> const & vrtx)
{
  for(int t=0;t<vrtx.size();t++)
  {
    int k = vrtx[t];
    for(int j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(int i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

void hid2vis_worker(const float * H,float * V,int h,int v,float * b,float * W,std::vector<int> const & vrtx)
{
  for(int t=0;t<vrtx.size();t++)
  {
    int k = vrtx[t];
    for(int i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(int j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}



struct RBM
{
  int h; // number hidden elements
  int v; // number visible elements
  int n; // number of samples
  float * c; // bias term for hidden state, R^h
  float * b; // bias term for visible state, R^v
  float * W; // weight matrix R^h*v
  float * X; // input data, binary [0,1], v*n
  RBM(int _h,int _v,float * _W,float * _b,float * _c,int _n,float * _X)
  {
    for(int k=0;k<100;k++)
      std::cout << _X[k] << "\t";
    std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = _c;
    b = _b;
    W = _W;
  }
  RBM(int _h,int _v,int _n,float* _X)
  {
    for(int k=0;k<100;k++)
      std::cout << _X[k] << "\t";
    std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new float[h];
    b = new float[v];
    W = new float[h*v];
    constant(c,3.0f,h);
    constant(b,3.0f,v);
    constant(W,3.0f,v*h);
  }

  float norm(float * dat,int size)
  {
    float ret = 0;
    for(int i=0;i<size;i++)
    {
      ret += dat[i]*dat[i];
    }
    return sqrt(ret);
  }

  void zero(float * dat,int size)
  {
    for(int i=0;i<size;i++)
    {
      dat[i] = 0;
    }
  }

  void constant(float * dat,float val,int size)
  {
    for(int i=0;i<size;i++)
    {
      dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
    }
  }

  void add(float * A, float * dA, float epsilon, int size)
  {
    for(int i=0;i<size;i++)
    {
      A[i] += epsilon * dA[i];
    }
  }

  void cd(int nGS,float epsilon)
  {

    // CD Contrastive divergence (Hinton's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hinton's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.
   
    float * vis0 = new float[n*v];
    float * hid0 = new float[n*h];
    float * vis = new float[n*v];
    float * hid = new float[n*h];
    float * dW = new float[h*v];
    float * dc = new float[h];
    float * db = new float[v];

    for(int i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i];
    }
    float S;
    int N = 1;
    for(int k=0;k<n;k++)
    {
      float max_S = 1e-5;
      for(int y=0,i=0;y<WIN;y++)
      {
        for(int x=0;x<WIN;x++,i++)
        {
          if(i+N<WIN*WIN && x+N<WIN)
          {
            S = (pow(
                  vis0[2*WIN*WIN*k+i]
                 +vis0[2*WIN*WIN*k+i+1]
                 ,2))/((N+1)*(
                  pow(vis0[2*WIN*WIN*k+i],2)
                 +pow(vis0[2*WIN*WIN*k+i+1],2)
                  )+1e-5);
          }
          else
          {
            S = 1.0f;
          }
          //S *= S;
          //S *= S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          vis0[2*WIN*WIN*k+i] = 1.0f - S;
          if(vis0[2*WIN*WIN*k+i]<0)vis0[2*WIN*WIN*k+i]=0;
          if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
          //vis0[2*WIN*WIN*k+i] = vis0[2*WIN*WIN*k+i]>0.5f;
          if(vis0[2*WIN*WIN*k+i]>max_S)max_S=vis0[2*WIN*WIN*k+i];
        }
      }
      for(int y=0,i=0;y<WIN;y++)
      {
        for(int x=0;x<WIN;x++,i++)
        {
          vis0[2*WIN*WIN*k+i] /= max_S;
          if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
        }
      }
    }

    //for(int k=0;k<n;k++)
    //{
    //  float max_S = 1e-5;
    //  float min_S = 1;
    //  for(int y=0,i=0;y<WIN;y++)
    //  {
    //    for(int x=0;x<WIN;x++,i++)
    //    {
    //      if(vis0[2*WIN*WIN*k+i]>max_S)max_S=vis0[2*WIN*WIN*k+i];
    //      if(vis0[2*WIN*WIN*k+i]<min_S)min_S=vis0[2*WIN*WIN*k+i];
    //    }
    //  }
    //  for(int y=0,i=0;y<WIN;y++)
    //  {
    //    for(int x=0;x<WIN;x++,i++)
    //    {
    //      vis0[2*WIN*WIN*k+i] = (vis0[2*WIN*WIN*k+i]-min_S)/(max_S-min_S);
    //      if(vis0[2*WIN*WIN*k+i]<0)vis0[2*WIN*WIN*k+i]=0;
    //      if(vis0[2*WIN*WIN*k+i]>1)vis0[2*WIN*WIN*k+i]=1;
    //      vis0[2*WIN*WIN*k+i] -= 0.5f;
    //    }
    //  }
    //}

    vis2hid(vis0,hid0);

    for(int i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }

    for (int iter = 1;iter<=nGS;iter++)
    {
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

      int off = rand()%(n);
      off *= v;
      for(int x=0,k=0;x<2*WIN;x++)
      {
        for(int y=0;y<WIN;y++,k++)
        {
          vis_preview[k] = vis[off+k];
        }
      }
      for(int x=0,k=0;x<2*WIN;x++)
      {
        for(int y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = 0.5f+0.5f*X[off+k];
          //vis0_preview[k] = vis0[off+k];
        }
      }

    }
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    float err = 0;
    for(int k=0;k<n;k++)
    {
      for(int i=0;i<v;i++)
      {
        for(int j=0;j<h;j++)
        {
          dW[i*h+j] -= (vis0[k*v+i]*hid0[k*h+j] - vis[k*v+i]*hid[k*h+j]) / n;
        }
      }

      for(int j=0;j<h;j++)
      {
        dc[j] -= (hid0[k*h+j]*hid0[k*h+j] - hid[k*h+j]*hid[k*h+j]) / n;
      }

      for(int i=0;i<v;i++)
      {
        db[i] -= (vis0[k*v+i]*vis0[k*v+i] - vis[k*v+i]*vis[k*v+i]) / n;
      }

      for(int i=0;i<v;i++)
      {
        err += (vis0[k*v+i]-vis[k*v+i])*(vis0[k*v+i]-vis[k*v+i]);
      }
    }
    err = sqrt(err);
    errs.push_back(err);
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

    std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    std::cout << "dc norm = " << norm(dc,h) << std::endl;
    std::cout << "db norm = " << norm(db,v) << std::endl;
    std::cout << "err = " << err << std::endl;

    delete [] vis0;
    delete [] hid0;
    delete [] vis;
    delete [] hid;
    delete [] dW;
    delete [] dc;
    delete [] db;

  }

  void sigmoid(float * p,float * X,int n)
  {
    for(int i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid(const float * X,float * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<int> > vrtx(8);
    for(int i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker,X,H,h,v,c,W,vrtx[thread]));
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }
  
  void hid2vis(const float * H,float * V)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<int> > vrtx(8);
    for(int i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker,H,V,h,v,b,W,vrtx[thread]));
    }
    for(int thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
  }

};

struct DataUnit
{
  DataUnit *   hidden;
  DataUnit *  visible;
  DataUnit * visible0;
  int h,v;
  float * W;
  float * b;
  float * c;
  RBM * rbm;
  DataUnit(int _v,int _h)
  {
    v = _v;
    h = _h;
    W = new float[v*h];
    b = new float[v];
    c = new float[h];
      hidden = NULL;
     visible = NULL;
    visible0 = NULL;
  }

  void train(float * dat, int n,int num_iters)
  {
    rbm = new RBM(h,v,W,b,c,n,dat);
    for(int i=0;i<num_iters;i++)
    {
      rbm->cd(1,.1);
    }
  }

  void transform(float* X,float* Y)
  {
    rbm->vis2hid(X,Y);
  }
};

// Multi Layer RBM
//
//  Auto-encoder
//
//          [***]
//         /     \
//     [*****] [*****]
//       /         \
// [********]   [********]
//   inputs      outputs
//
struct mRBM
{
  std::vector<DataUnit*>  input_branch;
  std::vector<DataUnit*> output_branch;
  DataUnit* bottle_neck;
  void addInputDatUnit(int v,int h)
  {
    DataUnit * unit = new DataUnit(v,h);
    input_branch.push_back(unit);
  }
  void addOutputDatUnit(int v,int h)
  {
    output_branch.push_back(new DataUnit(v,h));
  }
  void addBottleNeckDatUnit(int v,int h)
  {
    bottle_neck = new DataUnit(v,h);
  }
  void construct(std::vector<int> input_num,std::vector<int> output_num,int bottle_neck_num)
  {
    for(int i=0;i+1<input_num.size();i++)
    {
      input_branch.push_back(new DataUnit(input_num[i],input_num[i+1]));
    }
    for(int i=0;i+1<output_num.size();i++)
    {
      output_branch.push_back(new DataUnit(output_num[i],output_num[i+1]));
    }
    bottle_neck = new DataUnit(input_num[input_num.size()-1]+output_num[output_num.size()-1],bottle_neck_num);
  }
  mRBM()
  {

  }
  void copy(float * X,float * Y,int num)
  {
    for(int i=0;i<num;i++)
    {
      Y[i] = X[i];
    }
  }
  void train(int in_num,int out_num,int n_samp,float * in,float * out)
  {
    int n_iter = 100;
    float * X = NULL;
    float * Y = NULL;
    X = new float[in_num*n_samp];
    for(int i=0;i<input_branch.size();i++)
    {
      input_branch[i]->train(X,n_samp,n_iter);
      Y = new float[input_branch[i]->h*n_samp];
      input_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new float[input_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new float[out_num*n_samp];
    for(int i=0;i<output_branch.size();i++)
    {
      output_branch[i]->train(X,n_samp,n_iter);
      Y = new float[output_branch[i]->h*n_samp];
      output_branch[i]->transform(X,Y);
      delete [] X;
      X = NULL;
      X = new float[output_branch[i]->h*n_samp];
      copy(Y,X,output_branch[i]->h*n_samp);
      delete [] Y;
      Y = NULL;
    }
    delete [] X;
    X = NULL;
    X = new float[(input_branch[input_branch.size()-1]->h + output_branch[output_branch.size()-1]->h)*n_samp];
    bottle_neck->train(X,n_samp,n_iter);
    delete [] X;
    X = NULL;
  }
};

RBM * rbm = NULL;

void drawBox(void)
{
  std::cout << "drawBox" << std::endl;
  float max_err = 0;
  for(int k=0;k<errs.size();k++)
  {
    if(max_err<errs[k])max_err=errs[k];
  }
  glColor3f(1,1,1);
  glBegin(GL_LINES);
  for(int k=0;k+1<errs.size();k++)
  {
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , errs[k] / max_err
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , errs[k+1] / max_err
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*k / ((float)errs.size()-1)
              , errs[k] / max_err
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , 0
              , 0
              );
    glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
              , errs[k+1] / max_err
              , 0
              );
  }
  glEnd();
  if(rbm)
  {
    float max_W = -1000;
    float min_W =  1000;
    for(int i=0,k=0;i<rbm->v;i++)
      for(int j=0;j<rbm->h;j++,k++)
      {
        if(rbm->W[k]>max_W)max_W=rbm->W[k];
        if(rbm->W[k]<min_W)min_W=rbm->W[k];
      }
    float fact_W = 1.0 / (max_W - min_W);
    float col;
    glBegin(GL_QUADS);
    float d=3e-3;
    for(int x=0;x<WIN;x++)
    {
      for(int y=0;y<WIN;y++)
      {
        for(int i=0;i<rbm->v/WIN;i++)
        {
          for(int j=0;j<rbm->h/WIN;j++)
          {
            col = 0.5f + 0.5f*(rbm->W[(i+x)*rbm->h+j+y]-min_W)*fact_W;
            glColor3f(col,col,col);
            glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
            glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
          }
        }
      }
    }
    glEnd();
  }
  {
    float d = 5e-1;
    float col;
    glBegin(GL_QUADS);
    for(int y=0,k=0;y<2*WIN;y++)
    {
      for(int x=0;x<WIN;x++,k++)
      {
        col = vis_preview[k];
        glColor3f(col,col,col);
        glVertex3f(      (x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(      (x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
      }
    }
    for(int y=0,k=0;y<2*WIN;y++)
    {
      for(int x=0;x<WIN;x++,k++)
      {
        col = vis0_preview[k];
        glColor3f(col,col,col);
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
        glVertex3f(      0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
      }
    }
    glEnd();
  }
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(100000);
  glutPostRedisplay();
}

void init(void)
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 3,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */
}

void run_rbm(int w,int h,int n,float * dat)
{
  rbm = new RBM(w*h,w*h,n,dat);
  for(int i=0;;i++)
  {
    rbm->cd(10,.5);
  }
}

struct fault
{
  bool init;
  bool preview;
  float cx,cy; // fault center
  float vx,vy; // fault orientation
  float sx,sy; // structure orientation
  int wx,wy; // window width
  std::vector<float> amp; // seismic event amplitudes
  std::vector<float> shift; // seismic event shifts
  float fault_shift; // fault shift
  void randomize()
  {
    int num_events = 10+rand()%10;
    amp = std::vector<float>(num_events);
    shift = std::vector<float>(num_events);
    for(int i=0;i<num_events;i++)
    {
      amp[i] = (-1+2*(rand()%10000)/10000.0f);
      shift[i] = WIN*(-1+2*(rand()%10000)/10000.0f);
    }
    cx = 0.25+0.5*(rand()%10000)/10000.0f;
    cy = 0.25+0.5*(rand()%10000)/10000.0f;
    fault_shift = (rand()%10>=5)?4:-4;//10*(-1+2*(rand()%10000)/10000.0f);
    vx = 1;
    vy = 0.2f*(-1+2*(rand()%10000)/10000.0f);
    sx = 0;//0.2f*(-1+2*(rand()%10000)/10000.0f);
    sy = 1;
    init = true;
    preview = false;
  }
  void generate_data(int _wx,int _wy,float * dat)
  {
    if(init)
    {
      wx = _wx;
      wy = _wy;
      for(int i=0;i<2*wx*wy;i++)
      {
        dat[i] = 0;
      }
      int off = wx*wy;
      // generate data based on parameters
      float dx,dy;
      float vdot,sdot;
      for(int n=0;n<amp.size();n++)
      {
        for(int y=0,i=0;y<wy;y++)
        {
          for(int x=0;x<wx;x++,i++)
          {
            vdot = vx*(x-wx*cx) + vy*(y-wy*cy);
            sdot = sx*(x-wx*cx) + sy*(y-wy*cy);
            if(vdot>0)
            {
              dx = fault_shift + shift[n] - sdot;
              //dat[i] += amp[n]*exp(-dx*dx*WIN);
              dat[i] += amp[n]*sin(dx*2) + 0.05f*(rand()%10000)/10000.0f;
            }
            else
            {
              dx = shift[n] - sdot;
              //dat[i] += amp[n]*exp(-dx*dx*WIN);
              dat[i] += amp[n]*sin(dx*2) + 0.05f*(rand()%10000)/10000.0f;
            }
            dy = vdot;
            dat[i+off] += fabs(fault_shift) * exp(-dy*dy*10);
          }
        }
      }

      for(int y=0,i=0;y<wy;y++)
      {
        for(int x=0;x<wx;x++,i++)
        {
          //dat[i] = 0.5f + 0.15f*dat[i];
          dat[i+off] = dat[i+off]>0.5f?1:0;
        }
      }

      if(false)
      {
        std::cout << "$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
        for(int x=0,i=0;x<wx;x++)
        {
          for(int y=0;y<wy;y++,i++)
          {
            std::cout << dat[i];
          }
          std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
        for(int x=0,i=0;x<wx;x++)
        {
          for(int y=0;y<wy;y++,i++)
          {
            std::cout << dat[i+off];
          }
          std::cout << std::endl;
        }
      }

    }
    else
    {
      std::cout << "fault not initialized" << std::endl;
      exit(1);
    }
  }
  fault()
  {
    init = false;
    randomize();
  }
};

std::vector<long> indices;

struct real_fault
{
  bool init;
  int ox,oy,oz;
  int nx,ny,nz;
  void randomize()
  {
    init = true;
    while(true)
    {
      int id = indices[rand()%indices.size()];
      ox = (id/(n_z*n_y))%n_x-nx/2;
      oy = (id/n_z)%n_y-ny/2;
      oz = id%n_z-nz/2;
      float max_arr = 0;
      float val;
      for(int i=0,x=ox+nx/4;x+nx/4<ox+nx;x++)
      for(int y=oy+ny/4;y+ny/4<oy+ny;y++)
      for(int z=oz+nz/4;z+nz/4<oz+nz;z++,i++)
      {
        val = fault_arr[z+n_z*(y+n_y*x)];
        if(val>max_arr)max_arr=val;
      }
      std::cout << "max_arr=" << max_arr << std::endl;
      if(max_arr>0.9){break;}
    }
  }
  void generate_data(float * dat)
  {
    if(init)
    {
      int off = nx*ny*nz;
      for(int i=0,z=oz;z<oz+nz;z++)
      for(int y=oy;y<oy+ny;y++)
      for(int x=ox;x<ox+nx;x++,i++)
      {
        dat[i    ] = seismic_arr[z+n_z*(y+n_y*x)];
        dat[i+off] =   fault_arr[z+n_z*(y+n_y*x)];
      }
    }
    else
    {
      std::cout << "fault not initialized" << std::endl;
      exit(1);
    }
  }
  real_fault(int _nx,int _ny,int _nz)
  {
    nx = _nx;
    ny = _ny;
    nz = _nz;
    init = false;
    randomize();
  }
};

void faultGenerator2D(int wx,int wy,int n,float* dat)
{
  for(int k=0;k<n;k++)
  {
    // generate sample
    fault f;
    f.generate_data(wx,wy,&dat[k*2*wx*wy]);
  }
}

void realFaultGenerator2D(int wx,int wy,int n,float * dat)
{
  std::cout << "real fault generator" << std::endl;
	seismic_reader.read_sepval  ( &seismic_arr[0]
		                          , seismic_reader.o1
		                          , seismic_reader.o2
		                          , seismic_reader.o3
		                          , seismic_reader.n1
		                          , seismic_reader.n2
		                          , seismic_reader.n3
		                          );
  float max_seismic = 0;
  for(int i=0,size=seismic_reader.n1*seismic_reader.n2*seismic_reader.n3;i<size;i++)
  {
    if(max_seismic<seismic_arr[i])max_seismic=seismic_arr[i];
  }
  for(int i=0,size=seismic_reader.n1*seismic_reader.n2*seismic_reader.n3;i<size;i++)
  {
    seismic_arr[i] /= 1e-5 + max_seismic;
  }
  std::cout << "seismic reader done" << std::endl;
	  fault_reader.read_sepval  ( &  fault_arr[0]
		                          , fault_reader.o1
		                          , fault_reader.o2
		                          , fault_reader.o3
		                          , fault_reader.n1
		                          , fault_reader.n2
		                          , fault_reader.n3
		                          );
  float max_fault = 0;
  float min_fault = 1;
  for(int i=0,size=fault_reader.n1*fault_reader.n2*fault_reader.n3;i<size;i++)
  {
    if(max_fault<fault_arr[i])max_fault=fault_arr[i];
    if(min_fault>fault_arr[i])min_fault=fault_arr[i];
  }
  std::cout << "max-fault:" << max_fault << std::endl;
  std::cout << "min-fault:" << min_fault << std::endl;
  std::cout << "fault reader done" << std::endl;
  for(int k=0,x=0;x<n_x;x++)
    for(int y=0;y<n_y;y++)
      for(int z=0;z<n_z;z++,k++)
      {
        if(x>WIN)
        if(y>WIN)
        if(z>WIN)
        if(x<n_x-WIN)
        if(y<n_y-WIN)
        if(z<n_z-WIN)
        if(fault_arr[k]>0)
          indices.push_back(k);
      }
  for(int k=0;k<n;k++)
  {
    // generate sample
    real_fault f(wx,1,wy);
    f.generate_data(&dat[k*2*wx*wy]);
  }
  std::cout << "generating data done" << std::endl;
}

void keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case ' ':
      {
        int w = WIN;
        int h = WIN;
        int n = 3000;
        float * dat = new float[2*w*h*n];
        realFaultGenerator2D(w,h,n,dat);
        boost::thread * thr ( new boost::thread(run_rbm,2*w,h,n,dat) );
        break;
      }
    case 27:
      {
        exit(1);
        break;
      }
  };
}

int main(int argc,char ** argv)
{
  srand(time(0));
  std::cout << "Press space to start..." << std::endl;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("Boltzmann Machine");
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  init();
  glutMainLoop();
  return 0;
}

