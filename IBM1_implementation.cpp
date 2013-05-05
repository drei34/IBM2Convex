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
    
  cout << "Fitting and writing model from French to English!" << endl;
  cout << endl;
  IBM1 ibm1_f_e(argv[2]);
  

  ofstream ofs_1("file1.raw", ios::out | ios::binary);
  ofs_1.write((char *) & ibm1_f_e, sizeof(ibm1_f_e));
  ofs_1.close();


  ofstream ofs_2("file2.raw", ios::out | ios::binary);
  ofs_2.write((char *) & ibm1_e_f, sizeof(ibm1_e_f));
  ofs_2.close();

  IBM1 one;
  IBM1 two;

  ifstream ifs1("file1", ios::binary);
  ifs1.read((char *)&one, sizeof(two));

  ifstream ifs2("file2", ios::binary);
  ifs2.read((char *)&two, sizeof(two));

  one.write_alignments();
  two.write_alignments();

  system("python2.7 alignment_getter.py ibm1_gotten_alignments_english_to_french.txt ibm1_gotten_alignments_french_to_english.txt ");
  system("perl word_alignment_check.pl ibm1_symmetric_gotten_alignments_english_to_french.txt ");
  system("perl word_alignment_evaluation.pl trial.wa ibm1_symmetric_gotten_alignments_english_to_french.txt ");

  return 0;
}
