#ifndef IBM1_H
#define IBM1_H

#include<cstdlib>
#include<cmath>
#include<cstdio>
#include "AlignmentCorpus.h"

class IBM2Convex;

class IBM1: public AlignmentCorpus
{
public:
  IBM1(){
    cout << "Default!" << endl;
  }
  IBM1(string file_name);
  unordered_map < int, unordered_map< int, double > > t_var_; // t(f|e)
  unordered_map < int, double  > source_word_counts_; // c(e)
  unordered_map < int, unordered_map < int, double  > > pair_of_word_counts_; // c(e,f)
  void initialize_t_variables();
  void em(int T = 15);
  void expectation();
  void maximization();
  virtual void write_alignments(string evalutation_file_name = "trial");
  virtual void print();
  void write_variables(string);
  void read_variables(string);
  friend class IBM2Convex;
};

#endif
