OBJS1 = IBM1_implementation.o AlignmentCorpus.o IBM1.o
OBJS2 = IBM2Convex_implementation.o IBM2Convex.o AlignmentCorpus.o IBM1.o
CC = g++
DEBUG = -g
CFLAGS = -c -pg $(DEBUG)
LFLAGS = -pg $(DEBUG)

all: p1 p2

p1 : $(OBJS1)
	$(CC) $(LFLAGS) $(OBJS1) -o p1

p2 : $(OBJS2)
	$(CC) $(LFLAGS) $(OBJS2) -o p2

IBM1_implementation.o: IBM1_implementation.cpp IBM1.h AlignmentCorpus.h
	$(CC) $(CFLAGS) IBM1_implementation.cpp

AlignmentCorpus.o : AlignmentCorpus.cpp AlignmentCorpus.h
	$(CC) $(CFLAGS) AlignmentCorpus.cpp

IBM1.h : AlignmentCorpus.h

IBM1.o : IBM1.cpp IBM1.h
	$(CC) $(CFLAGS) IBM1.cpp

IBM2Convex.h : AlignmentCorpus.h IBM1.h

IBM2Convex.o : IBM2Convex.cpp IBM2Convex.h IBM1.h AlignmentCorpus.h
	$(CC) $(CFLAGS) IBM2Convex.cpp

IBM2Convex_implementation.o: IBM2Convex_implementation.cpp IBM2Convex.h 
	$(CC) $(CFLAGS) IBM2Convex_implementation.cpp

test:
	nohup ./p2 training_data_file_names_english_to_french_12_pair.txt training_data_file_names_french_to_english_12_pair.txt trial > trial_run &


clean:
	\rm *.o *~ p1 p2

