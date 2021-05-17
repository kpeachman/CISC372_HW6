CFLAGS=
fastblur: obj/fastblur.o
	nvcc $(CFLAGS) obj/fastblur.o -o fastblur -lm


obj/fastblur.o: fastblur.cu
	nvcc -c $(CFLAGS) fastblur.cu -o obj/fastblur.o 


clean:
	rm -f obj/* fastblur output.png
