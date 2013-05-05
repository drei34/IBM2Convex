#ifndef IBM1_H
#define IBM1_H

#include<cstdlib>
#include<cmath>
#include "AlignmentCorpus.h"

class IBM1: public AlignmentCorpus
{
public:
  IBM1(){
    cout << "Default!" << endl;
  }
  IBM1(string file_name);
  unordered_map < string, unordered_map< string, double > > t_var_; // t(f|e)
  unordered_map < string, double  > source_word_counts_; // c(e)
  unordered_map < string, unordered_map < string, double  > > pair_of_word_counts_; // c(e,f)
  void initialize_t_variables();
  void em(int T = 5);
  void expectation();
  void maximization();
  virtual void write_alignments(string evalutation_file_name = "trial");
  virtual void print();
};

#endif
