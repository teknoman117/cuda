//
//  imgproc.h
//  
//
//  Created by Nathaniel Lewis on 3/8/12.
//  Copyright (c) 2012 E1FTW Games. All rights reserved.
//

#ifndef _imgproc_h
#define _imgproc_h

#include <iostream>
#include <sys/time.h>

void boxfilter(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh);
void sobelfilter(int iw, int ih, unsigned char *source, unsigned char *dest);

unsigned char* createImageBuffer(unsigned int bytes);
void           destroyImageBuffer(unsigned char* bytes);

#endif
