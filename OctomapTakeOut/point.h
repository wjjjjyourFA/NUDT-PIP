#ifndef POINT_H
#define POINT_H

#ifndef __INT64__
#define __INT64__
typedef  signed     long        INT64;
#endif

struct Pointxyzi{
    double x;
    double y;
    double z;
    double i;
};

#endif

float x_min =  10000000000;
float x_max = -10000000000;
float y_min =  10000000000;
float y_max = -10000000000;
float z_min =  10000000000;
float z_max = -10000000000;
