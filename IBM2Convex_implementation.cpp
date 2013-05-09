#include "IBM2Convex.h"
#include "UsefulFunctions.h"
using namespace std;

int main(int argc, char * argv[])
{
  if(argc != 9){
    /*                                                                                                                                       
      Notice that the actual program name is argv[0].                                                                                        
    */
    cerr << "Usage:\tprogram_name training_data_file_english_to_french training_data_file_french_to_english ibm1/uniform block_percentage lambda c trial/test T \n\tYou actually fed in " << argc << " things." << endl;
    exit(1);
  }

  int total_iterations = string_to_number<int>(argv[8]); // Number of full passes over the data.
  string initialization_type(argv[3]);
  double block_percentage = string_to_number<double>(argv[4]);
  double lambda = string_to_number<double>(argv[5]);
  double c = string_to_number<double>(argv[6]);
  string evaluation_file_name(argv[7]);

  
  cout << "Fitting and writing model from English to French!" << endl;
  IBM2Convex ibm2convex_e_f(argv[1],initialization_type,block_percentage,lambda,c,total_iterations);
 
  cout << endl;

  cout << "Fitting and writing model from French to English!" << endl;
  IBM2Convex ibm2convex_f_e(argv[2],initialization_type,block_percentage,lambda,c,total_iterations);

  cout << endl;

  cout << "Data used for this run: " << endl;  
  cout << "The number of iterations model was ran for: " << total_iterations << endl;
  cout << "The initialization type: " << initialization_type << endl;
  cout << "The block percentage: " << block_percentage << endl;
  cout << "The lambda: " << lambda << endl;
  cout << "The c: " << c << endl;
  cout << "The evaluation file type: " << evaluation_file_name << endl;
  cout << "The lambda: " << ibm2convex_e_f.lambda_ << endl;
  cout << "The alpha: " << ibm2convex_e_f.alpha_ << endl;
  cout << "The theta: " << ibm2convex_e_f.theta_ << endl;

  cout << endl;

  cout << "There are a total of " << ibm2convex_e_f.N_ << " sentence pairs!"<<endl;

  cout << endl;

  cout << "Optimizing the English to French model!" << endl;
  cout << endl;
  
  string evaluation_command; 
  string extra_marker = string("_N_") + number_to_string<int>(ibm2convex_e_f.N_) + string("_block_percentage_") + number_to_string<double>(ibm2convex_e_f.block_percentage_) + string("_lambda_") + number_to_string<double>(ibm2convex_e_f.lambda_) + string("_c_") + number_to_string<double>(ibm2convex_e_f.c_) + string("_alpha_") + number_to_string<double>(ibm2convex_e_f.alpha_) + string("_theta_") + number_to_string<double>(ibm2convex_e_f.theta_) + string("_evaluation_file_name_") + string(evaluation_file_name);
  
  while(ibm2convex_e_f.iteration_number_ < ibm2convex_e_f.total_iterations_){
    cout << "Starting EF model iteration number " << ibm2convex_e_f.iteration_number_ + 1<< "!" << endl;
    ibm2convex_e_f.optimize_seg(); // Member variable "iteration_number_ " is updated within this function.
    cout << "Done with EF model iteration number " << ibm2convex_e_f.iteration_number_<< "!" << endl;
    cout << "The current EF objective is: " << ibm2convex_e_f.objectives_.back() << "!" << endl;
    cout << "The current EF objective using averaged parameters is: " << ibm2convex_e_f.objectives_average_.back() << "!" << endl;
    cout << "The current EF objective using best-objective parameters is: " << ibm2convex_e_f.objectives_best_overall_.back() << "!" << endl;

    cout << endl;

    cout << "Starting FE model iteration number " << ibm2convex_f_e.iteration_number_ + 1 << "!" << endl;
    ibm2convex_f_e.optimize_seg(); // Member variable "iteration_number_ " is updated within this function.
    cout << "Done with FE model iteration number " << ibm2convex_f_e.iteration_number_<< "!" << endl; 
    cout << "The current FE objective is: " << ibm2convex_f_e.objectives_.back() << "!" << endl;
    cout << "The current FE objective using averaged parameters is: " << ibm2convex_f_e.objectives_average_.back() << "!" << endl;
    cout << "The current FE objective using best-objective parameters is: " << ibm2convex_f_e.objectives_best_overall_.back() << "!" << endl;

    cout << endl;

    cout << "Writing the alignment files!" << endl;
    ibm2convex_e_f.write_alignments(evaluation_file_name);
    ibm2convex_f_e.write_alignments(evaluation_file_name);
    cout << "Done writing alignemnt files!" << endl;

    evaluation_command = "python alignment_getter.py ibm2convex_gotten_alignments_english_to_french" + extra_marker + "_final.txt ibm2convex_gotten_alignments_french_to_english" + extra_marker + "_final.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french" + extra_marker + "_final.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_name + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french"+extra_marker+"_final.txt ";
    system(evaluation_command.c_str());

    evaluation_command = "python alignment_getter.py ibm2convex_gotten_alignments_english_to_french" + extra_marker + "_average.txt ibm2convex_gotten_alignments_french_to_english" + extra_marker + "_average.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french" + extra_marker + "_average.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_name + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french"+extra_marker+"_average.txt ";
    system(evaluation_command.c_str());

    evaluation_command = "python alignment_getter.py ibm2convex_gotten_alignments_english_to_french" + extra_marker + "_best_overall.txt ibm2convex_gotten_alignments_french_to_english" + extra_marker + "_best_overall.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french" + extra_marker + "_best_overall.txt";
    system(evaluation_command.c_str());
    evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_name + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french"+extra_marker+"_best_overall.txt ";
    system(evaluation_command.c_str());
  }
  
  return 0;
}

