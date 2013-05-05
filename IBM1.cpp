#include<cctype>
#include<cstdlib>
#include<cmath>
#include "AlignmentCorpus.h"
#include "IBM1.h"
using namespace std;
using namespace boost;

IBM1::IBM1(string file_name):AlignmentCorpus(file_name)
{
  initialize_t_variables();
}

void IBM1::initialize_t_variables()
{
  for(unordered_map<string, unordered_set < string > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
    string source_word = i->first;
    unordered_set<string> target_words = i->second;
    size_t source_dictionary_count = target_words.size();
    for(unordered_set<string>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
      string target_word = *j;
      t_var_[target_word][source_word] = static_cast<double>(1)/(source_dictionary_count);
    }
  }
}

void IBM1::em(int T)
{
  int total_counts;
  for(int i = 0; i < T; ++i){
    cout << "Doing EM iteration " << i+1 << " for IBM1." << endl;
    expectation();
    maximization();
  }
}

void IBM1::expectation()
{
  float delta;
  
  for(vector<vector<string> >::iterator i = source_sentences_tokenized_.begin(), j = target_sentences_tokenized_.begin(); i != source_sentences_tokenized_.end() && j != target_sentences_tokenized_.end(); ++i, ++j){
    unordered_map <string, float> t_var__summed;
    for(vector<string>::iterator jj = j->begin(); jj != j->end(); ++jj){
      string target_word = *jj;
      t_var__summed[target_word] = 0;
      for(vector<string>::iterator ii = i->begin(); ii != i->end(); ++ii){
	string source_word = *ii;
	t_var__summed[target_word] += t_var_[target_word][source_word];
      }
    }
    for(vector<string>::iterator jj = j->begin(); jj != j->end(); ++jj){
      string target_word = *jj;
      for(vector<string>::iterator ii = i->begin(); ii != i->end(); ++ii){
	string source_word = *ii;
	delta = t_var_[target_word][source_word]/t_var__summed[target_word];
	pair_of_word_counts_[source_word][target_word] += delta;
	source_word_counts_[source_word] += delta;
      }
    }
  }
}

void IBM1::maximization()
{
  for(vector<vector<string> >::iterator i = source_sentences_tokenized_.begin(), j = target_sentences_tokenized_.begin(); i != source_sentences_tokenized_.end() && j != target_sentences_tokenized_.end(); ++i, ++j){
    for(vector<string>::iterator jj = j->begin(); jj != j->end(); ++jj){
      for(vector<string>::iterator ii = i->begin(); ii != i->end(); ++ii){
	t_var_[*jj][*ii] = pair_of_word_counts_[*ii][*jj]/source_word_counts_[*ii];
      }
    }
  }
  source_word_counts_.clear();
  pair_of_word_counts_.clear();
  if(source_word_counts_.empty()){
    cout << "Reunordered_setting the counts to zero!" << endl;
  }
}

void IBM1::write_alignments(string evaluation_file_name)
{
  ifstream source_file, target_file;
  if(translation_type_ == "english_to_french"){
    source_file.open(string(evaluation_file_name + ".e").c_str());
    target_file.open(string(evaluation_file_name + ".f").c_str());
  }
  else{
    source_file.open(string(evaluation_file_name + ".f").c_str());
    target_file.open(string(evaluation_file_name + ".e").c_str());
  }
  
  ofstream gotten_alignments_file;
  string gotten_alignments_file_name = "ibm1_gotten_alignments_"+translation_type_+".txt";
  gotten_alignments_file.open(gotten_alignments_file_name.c_str());
  
  string source_file_line, target_file_line;
  while(true){
    getline(source_file,source_file_line);
    getline(target_file,target_file_line);
    
    if(source_file_line.size() == 0){
      break;
    }

    istringstream source_file_ss(source_file_line);
    istringstream target_file_ss(target_file_line);

    vector <string> source_sentence_words, target_sentence_words;
    string word;
    while (source_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      source_sentence_words.push_back(word); // Here we tokenize the sentence.                                                                   
    }
    
    while (target_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      target_sentence_words.push_back(word);
    }

    string data_pair_number = source_sentence_words[1].substr(5,4);
    source_sentence_words.pop_back();
    target_sentence_words.pop_back();
    
    source_sentence_words.erase(source_sentence_words.begin(),source_sentence_words.begin()+2);
    target_sentence_words.erase(target_sentence_words.begin(),target_sentence_words.begin()+2);
    
    source_sentence_words.insert(source_sentence_words.begin(),"null");

    for(int target_sentence_index = 0; target_sentence_index < target_sentence_words.size();++target_sentence_index){
      string target_word = target_sentence_words[target_sentence_index];
      int aligned_source_word_index = 0;
      string aligned_source_word = source_sentence_words[aligned_source_word_index];
      for(int source_sentence_index = 0; source_sentence_index < source_sentence_words.size();++source_sentence_index){
	string source_word = source_sentence_words[source_sentence_index];
	if(t_var_[target_word][source_word] > t_var_[target_word][aligned_source_word]){
	  aligned_source_word_index = source_sentence_index;
	  aligned_source_word = source_sentence_words[aligned_source_word_index];
	}
      }
      if(aligned_source_word_index != 0){
	gotten_alignments_file << data_pair_number << " " << aligned_source_word_index << " " << target_sentence_index+1 <<endl; 
      }
    }
  }
  gotten_alignments_file.close();
}

void IBM1::print()
{
  // Why does it complain if we don't use const iterator? Because the object is const!
  for(unordered_map< string, unordered_map<string, double> >::const_iterator j = t_var_.begin(); j != t_var_.end(); ++j){
      for(unordered_map<string, double>::const_iterator i= (j->second).begin(); i != (j->second).end(); ++i){
	cout <<"t_var_[" << j->first << "|" << i->first << "] = " << (t_var_[j->first])[i->first] << endl;
      }
  }
}
