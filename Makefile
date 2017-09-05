all:
	g++ -O3 -g0 -o digits main.cpp readBMP.c -lboost_thread -lboost_system -lGL -lglut -lGLU;
	./digits
