# Library
#g++ -fPIC -c ot_func.cpp -O3 -fno-trapping-math -freciprocal-math \
#    -fassociative-math -fexcess-precision=fast -fno-rounding-math -fno-signaling-nans
g++ -fPIC -c ot_func.cpp -O3 -march=native -ffast-math -fno-finite-math-only
g++ -shared ot_func.o -lm -o libot.so
#nvcc -Xcompiler -fPIC -c remap_if.cpp remap_kernel.cu -O3
#nvcc -shared remap_if.o remap_kernel.o -o libremap.so
#nvcc -Xcompiler -fPIC -c ot_func.cu -O3 -arch=sm_61
#nvcc -shared ot_func.o -o libot.so

# Cleanup
rm *.o
