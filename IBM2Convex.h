#ifndef IBM2CONVEX_H
#define IBM2CONVEX_H

#include <iterator>
#include <functional>
#include <set>
#include <numeric>
#include <utility>
#include <ctime>
#include "IBM1.h"
#include "UsefulFunctions.h"

class IBM2Convex: public AlignmentCorpus
{
public:
  IBM2Convex(string file_name, string initialization_type = "ibm1", double block_percentage = .05, double lambda = .001, double c = .02, int total_iterations = 10, double alpha = 1, double theta = .5);
  void write_alignments(string file_name);
  void print();
  double compute_objective(vector<vector<double> > &, vector<vector<double> > &);
  void optimize_seg();
  void check_for_zero_gradients();
  
  int iteration_number_;
  int total_iterations_;
  int N_;
  string initialization_type_;
  double learning_rate_;
  int number_of_mistakes_;
  
  vector<double> objectives_;
  vector<double> objectives_average_;
  vector<double> objectives_best_overall_;
  
  vector<int> shuffled_training_indicies_;
  double block_percentage_;
  int B_; // The block_size_.
  int to_compute_objective_counter_;

  vector<vector<double> > t_var_;
  vector<vector<double> > d_var_;
  vector<vector<double> > t_var_average_;
  vector<vector<double> > d_var_average_;
  vector<vector<double> > t_var_best_overall_;
  vector<vector<double> > d_var_best_overall_;

  vector<vector<double> > t_var_gradients_;
  vector<vector<double> > d_var_gradients_;

  unordered_map<int, unordered_map<int, int> > t_var_lookup_;
  unordered_set<pair<int,int> > d_var_gradients_changed_;
  unordered_set<pair<int,int> > t_var_gradients_changed_;

  double lambda_;
  double alpha_;
  double c_;
  double theta_;

protected:
  size_t max_source_sentence_length_;
  size_t max_target_sentence_length_;
  void get_max_sentence_length();
  void initialize_d_variables_uniformly();
  void initialize_t_variables_uniformly(){
    cout << "The defualt is via EM for now!" << endl;
  }
  void compute_gradients(int);
};

#endif
