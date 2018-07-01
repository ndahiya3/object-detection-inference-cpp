CC = g++
CFLAGS = -std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w

DIR = /home/ndahiya/standalone/

INC = -I/usr/local/include/eigen3
INC += -I./include/third_party
INC += -I./include
INC += -I./include/nsync/public/

LDFLAGS =  -lprotobuf -pthread -lpthread
LDFLAGS += -L$(DIR)/lib/ -Wl,-R$(DIR)/lib/ '-Wl,-R$$ORIGIN'
LDFLAGS += -ltensorflow_cc -lopencv_core -lopencv_highgui -lopencv_imgproc

all: 5_test

protoc_middleman: labelmap.proto
	protoc $$PROTO_PATH --cpp_out=. labelmap.proto
	@touch protoc_middleman

5_test:	protoc_middleman
	$(CC) $(CFLAGS) 5_test.cpp labelmap.pb.cc -o 5_test `pkg-config --cflags --libs protobuf` $(INC) $(LDFLAGS)
run:
	./5_test
clean:
	rm -f 5_test
	rm -f protoc_middleman labelmap.pb.cc labelmap.pb.h
