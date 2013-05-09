OBJS1 = IBM2Convex_implementation.o IBM2Convex.o AlignmentCorpus.o IBM1.o
OBJS2 = IBM1_implementation.o AlignmentCorpus.o IBM1.o
OBJS3 = AlignmentCorpus_implementation.o AlignmentCorpus.o
CC = g++
DEBUG = -g
CFLAGS = -c -pg $(DEBUG)
LFLAGS = -pg $(DEBUG)

all: p1 p2 p3

p1 : $(OBJS1)
	$(CC) $(LFLAGS) $(OBJS1) -o p1

p2 : $(OBJS2)
	$(CC) $(LFLAGS) $(OBJS2) -o p2

p3: $(OBJS3)
	$(CC) $(LFLAGS) $(OBJS3) -o p3

AlignmentCorpus_implementation.o: AlignmentCorpus_implementation.cpp AlignmentCorpus.h
	$(CC) $(CFLAGS) AlignmentCorpus_implementation.cpp

IBM1_implementation.o: IBM1_implementation.cpp IBM1.h AlignmentCorpus.h
	$(CC) $(CFLAGS) IBM1_implementation.cpp

AlignmentCorpus.o : AlignmentCorpus.cpp AlignmentCorpus.h
	$(CC) $(CFLAGS) AlignmentCorpus.cpp

IBM1.h : AlignmentCorpus.h

IBM1.o : IBM1.cpp IBM1.h
	$(CC) $(CFLAGS) IBM1.cpp

IBM2Convex.h : AlignmentCorpus.h IBM1.h UsefulFunctions.h

IBM2Convex.o : IBM2Convex.cpp IBM2Convex.h IBM1.h AlignmentCorpus.h
	$(CC) $(CFLAGS) IBM2Convex.cpp

IBM2Convex_implementation.o: IBM2Convex_implementation.cpp IBM2Convex.h UsefulFunctions.h
	$(CC) $(CFLAGS) IBM2Convex_implementation.cpp

test:
	nohup ./p1 training_data_file_names_english_to_french_1_pair.txt training_data_file_names_french_to_english_1_pair.txt ibm1 .005 .001 .2 test 20 > /proj/mlnlp/andrei/"IBM2Convex C++"/results/1_pair_initialization_type_ibm1_block_percentage_.005_lambda_.001_c_.2_alpha_1_theta_.5_test_T_20 &
#	nohup ./p1 training_data_file_names_english_to_french_12_pair.txt training_data_file_names_french_to_english_12_pair.txt ibm1 .0005 .001 .2 test 30 > /proj/mlnlp/andrei/"IBM2Convex C++"/results/12_pair_initialization_type_ibm1_block_percentage_.0005_lambda_.001_c_.2_alpha_1_theta_.5_test_T_30 &
#	nohup ./p1 training_data_file_names_english_to_french_12_pair.txt training_data_file_names_french_to_english_12_pair.txt ibm1 .05 .001 .2 test 35 > /proj/mlnlp/andrei/"IBM2Convex C++"/results/12_pair_initialization_type_ibm1_block_percentage_.05_lambda_.001_c_.2_alpha_1_theta_.5_test_T_35 &
#	nohup ./p1 training_data_file_names_english_to_french_80_pair.txt training_data_file_names_french_to_english_80_pair.txt ibm1 .0005 .001 .2 test 40 > /proj/mlnlp/andrei/"IBM2Convex C++"/results/80_pair_initialization_type_ibm1_block_percentage_.0005_lambda_.001_c_.2_alpha_1_theta_.5_test_T_40 &
#	nohup ./p1 training_data_file_names_english_to_french_80_pair.txt training_data_file_names_french_to_english_80_pair.txt ibm1 .05 .001 .2 test 45 > /proj/mlnlp/andrei/"IBM2Convex C++"/results/80_pair_initialization_type_ibm1_block_percentage_.05_lambda_.001_c_.2_alpha_1_theta_.5_test_T_45 &

clean:
	\rm *.o *~ p1 p2 p3

