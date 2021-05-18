#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__ void computeColumn(uint8_t* src,float* dest,int pWidth,int height,int radius,int bpp){
    int i;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= pWidth)
        return;
    //initialize the first element of each column
    dest[col]=src[col];
    //start tue sum up to radius*2 by only adding
    for (i=1;i<=radius*2;i++)
        dest[i*pWidth+col]=src[i*pWidth+col]+dest[(i-1)*pWidth+col];
    for (i=radius*2+1;i<height;i++)
        dest[i*pWidth+col]=src[i*pWidth+col]+dest[(i-1)*pWidth+col]-src[(i-2*radius-1)*pWidth+col];
    //now shift everything up by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
    for (i=radius;i<height;i++){
        dest[(i-radius)*pWidth+col]=dest[i*pWidth+col]/(radius*2+1);
    }
    //now the first and last radius values make no sense, so blank them out
    for (i=0;i<radius;i++){
        dest[i*pWidth+col]=0;
        dest[(height-1)*pWidth-i*pWidth+col]=0;
    }

}

__global__ void computeRow(float* src,float* dest,int pWidth, int height, int radius,int bpp){
    int i;
    int bradius=radius*bpp;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= height)
        return;
    //initialize the first bpp elements so that nothing fails
    for (i=0;i<bpp;i++)
        dest[row*pWidth+i]=src[row*pWidth+i];
    //start the sum up to radius*2 by only adding (nothing to subtract yet)
    for (i=bpp;i<bradius*2*bpp;i++)
        dest[row*pWidth+i]=src[row*pWidth+i]+dest[row*pWidth+i-bpp];
     for (i=bradius*2+bpp;i<pWidth;i++)
        dest[row*pWidth+i]=src[row*pWidth+i]+dest[row*pWidth+i-bpp]-src[row*pWidth+i-2*bradius-bpp];
    //now shift everything over by radius spaces and blank out the last radius items to account for sums at the end of the kernel, instead of the middle
    for (i=bradius;i<pWidth;i++){
        dest[row*pWidth+i-bradius]=dest[row*pWidth+i]/(radius*2+1);
    }
    //now the first and last radius values make no sense, so blank them out
    for (i=0;i<bradius;i++){
        dest[row*pWidth+i]=0;
        dest[(row+1)*pWidth-1-i]=0;
    }
}

int Usage(char* name){
    printf("%s: <filename> <blur radius>\n\tblur radius=pixels to average on any side of the current pixel\n",name);
    return -1;
}


int main(int argc,char** argv){
    long t1,t2;
    int radius=0;
    int width,height,bpp,pWidth;
    char* filename;
    uint8_t *img;
    float* dest,*mid;

    if (argc!=3)
        return Usage(argv[0]);
    filename=argv[1];
    sscanf(argv[2],"%d",&radius);
   
    img=stbi_load(filename,&width,&height,&bpp,0);

    pWidth=width*bpp;  //actual width in bytes of an image row

    //allocate memory for images
    cudaMallocManaged(&mid, sizeof(float)*pWidth*height);   
    cudaMallocManaged(&dest, sizeof(float)*pWidth*height);
    
    t1=time(NULL);

    computeColumn<<<1,1>>>(img,mid,pWidth,height,radius,bpp);
  
    stbi_image_free(img); //done with image

    computeRow<<<1,1>>>(mid,dest,pWidth,height,radius,bpp);
    
    t2=time(NULL);
    free(mid); //done with mid




}