

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void health(int *gh,int H,int *alive,int *gs)
{
    unsigned id=threadIdx.x;
    gh[id]=H;
    alive[id]=1;
    gs[id]=0;
}

__global__ void dkernel(int T,int *gh,int *gs,int *gx,int *gy,int *alive,int *count)
{
    unsigned id=threadIdx.x;
    int currentround=1;
    while(count[0]>1)
    {
       if(currentround%T==0)
       {
        currentround++;
        __syncthreads();
        continue;
       }
       int target=(id+currentround)%T;
        int m1=(gx[target]-gx[id]);
        int m2=(gy[target]-gy[id]);
        int distance=1e9;
        int shootindex=-1;
       if(alive[id]==1)
       {

        for(int i=0;i<T;i++)
        {   int a1=gx[i]-gx[id];
            int a2=gy[i]-gy[id];
            int dis1=a1;
            if(a1<0)
            dis1=-a1;
            int dis2=a2;
            if(a2<0)
            dis2=-a2;
            int d=dis1+dis2;
            if(i!=id && alive[i]==1 && a1*m2==a2*m1)
            {

                    if((m1<0 && a1<0) || (m2<0 &&  a2<0) || (m1>0 && a1>0) || (m2>0 && a2>0)) // if the point is in the direction of target.
                    {
                        int y=d;
                        if(distance>y)
                        {
                            distance=y;
                            shootindex=i;

                        }

                    }

            }
        }
       }
      if(shootindex!=-1)
      {
        gs[id]++;
        atomicSub(&gh[shootindex],1);
      }
      count[0]=0;

      __syncthreads();
      currentround++;
      if(gh[id]<=0)
      {
          alive[id]=0;
      }
      if(gh[id]>0){
        atomicAdd(&count[0],1);

      }
      __syncthreads();



    }

}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *gh;
    cudaMalloc(&gh,T*sizeof(int));
    int *alive;
    cudaMalloc(&alive,T*sizeof(int));
    int *gs;
    cudaMalloc(&gs,T*sizeof(int));
    health<<<1,T>>>(gh,H,alive,gs);
    int *gx;
    cudaMalloc(&gx,T*sizeof(int));
    cudaMemcpy(gx,xcoord,T*sizeof(int),cudaMemcpyHostToDevice);
    int *gy;
    cudaMalloc(&gy,T*sizeof(int));
    cudaMemcpy(gy,ycoord,T*sizeof(int),cudaMemcpyHostToDevice);
    int count=T;
    int *gpu_count;
    cudaMalloc(&gpu_count,1*sizeof(int));
    cudaMemcpy(gpu_count,&count,1*sizeof(int),cudaMemcpyHostToDevice);
    dkernel<<<1,T>>>(T,gh,gs,gx,gy,alive,gpu_count);
    cudaMemcpy(score,gs,T*sizeof(int),cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}