#include<cstdlib>
#include<cmath>
#include "AlignmentCorpus.h"
#include "IBM1.h"
using namespace std;

int main(int argc, char * argv[])
{
  if(argc != 3){
    cerr << "Usage:\tprogram_name training_data_file[english to french or reverse] training_data_file[french to english or reverse] \n\tYou actually fed in " << argc << " things." << endl;
    exit(1);  
  }
  
  cout << "Fitting and Writing model from English to French!" << endl;
  cout << endl;
  IBM1 ibm1_e_f(argv[1]);
    
  cout << "Fitting and Writing model from French to English!" << endl;
  cout << endl;
  IBM1 ibm1_f_e(argv[2]);
  
  ibm1_e_f.em(10);
  ibm1_f_e.em(10);

  ibm1_e_f.write_alignments();
  ibm1_f_e.write_alignments();

  system("python alignment_getter.py ibm1_gotten_alignments_english_to_french.txt ibm1_gotten_alignments_french_to_english.txt ");
  system("perl word_alignment_check.pl ibm1_symmetric_gotten_alignments_english_to_french_.txt ");
  system("perl word_alignment_evaluation.pl trial.wa ibm1_symmetric_gotten_alignments_english_to_french_.txt ");

  return 0;
}
