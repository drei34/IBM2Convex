#include "IBM2Convex.h"
using namespace std;

int main(int argc, char * argv[])
{
  if(argc != 4){
    /*                                                                                                                                       
      Notice that the actual program name is argv[0].                                                                                        
    */
    cerr << "Usage:\tprogram_name training_data_file[english to french or reverse] training_data_file[french to english or reverse] \n\tYou \
actually fed in " << argc << " things." << endl;
    exit(1);
  }

  const int number_of_iterations = 5; // Number of full passes over the data.
  string evaluation_file_type(argv[3]);


  cout << "Fitting and writing model from English to French!" << endl;
  IBM2Convex ibm2convex_e_f(argv[1]);
  

  cout << "Fitting and writing model from French to English!" << endl;
  IBM2Convex ibm2convex_f_e(argv[2]);

  cout << "There are a total of " << ibm2convex_e_f.N_ << " sentence pairs!"<<endl;

  cout << "Optimizing the English to French model!" << endl;
  
  while(ibm2convex_e_f.iteration_number_ < number_of_iterations){
    cout << "Starting iteration number " << ibm2convex_e_f.iteration_number_ + 1<< "!" << endl;
    ibm2convex_e_f.optimize_seg(); // Member variable "iteration_number_ " is updated within this function.
    cout << "Done with iteration number " << ibm2convex_e_f.iteration_number_<< "!" << endl;
    cout << "The current objective is: " << ibm2convex_e_f.objectives_.back() << "!" << endl;
    cout << "The current objective using averaged parameters is: " << ibm2convex_e_f.objectives_average_.back() << "!" << endl;
    cout << "The current objective using best-objective parameters is: " << ibm2convex_e_f.objectives_best_overall_.back() << "!" << endl;
  }
  ibm2convex_e_f.write_alignments(evaluation_file_type);

  cout << "Optimizing the French to English model!" << endl;
  while(ibm2convex_f_e.iteration_number_ < number_of_iterations){
    cout << "Starting iteration number " << ibm2convex_f_e.iteration_number_ + 1 << "!" << endl;
    ibm2convex_f_e.optimize_seg(); // Member variable "iteration_number_ " is updated within this function.
    cout << "Done with iteration number " << ibm2convex_f_e.iteration_number_<< "!" << endl; 
    cout << "The current objective is: " << ibm2convex_f_e.objectives_.back() << "!" << endl;
    cout << "The current objective using averaged parameters is: " << ibm2convex_f_e.objectives_average_.back() << "!" << endl;
    cout << "The current objective using best-objective parameters is: " << ibm2convex_f_e.objectives_best_overall_.back() << "!" << endl;
  } 
  ibm2convex_f_e.write_alignments(evaluation_file_type);

  string evaluation_command;
  system("python alignment_getter.py ibm2convex_gotten_alignments_english_to_french_final.txt ibm2convex_gotten_alignments_french_to_english_final.txt ");
  system("perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french_final.txt ");
  evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_type + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french_final.txt ";
  system(evaluation_command.c_str());

  system("python alignment_getter.py ibm2convex_gotten_alignments_english_to_french_average.txt ibm2convex_gotten_alignments_french_to_english_average.txt ");
  system("perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french_average.txt ");
  evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_type + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french_average.txt ";
  system(evaluation_command.c_str());

  system("python alignment_getter.py ibm2convex_gotten_alignments_english_to_french_best_overall.txt ibm2convex_gotten_alignments_french_to_english_best_overall.txt ");
  system("perl word_alignment_check.pl ibm2convex_symmetric_gotten_alignments_english_to_french_best_overall.txt ");
  evaluation_command = "perl word_alignment_evaluation.pl " + evaluation_file_type + ".wa ibm2convex_symmetric_gotten_alignments_english_to_french_best_overall.txt ";
  system(evaluation_command.c_str());

  return 0;
}

