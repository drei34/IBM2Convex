#include<cctype>
#include "AlignmentCorpus.h"
#include "IBM1.h"

IBM1::IBM1(string file_name):AlignmentCorpus(file_name)
{
  initialize_t_variables();
}

void IBM1::initialize_t_variables()
{
  for(unordered_map<int, unordered_set < int > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
    int source_word = i->first;
    unordered_set<int> target_words = i->second;
    size_t source_dictionary_count = target_words.size();
    for(unordered_set<int>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
      int target_word = *j;
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
  double delta;
  
  for(vector<vector<int> >::iterator i = source_sentences_tokenized_.begin(), j = target_sentences_tokenized_.begin(); i != source_sentences_tokenized_.end() && j != target_sentences_tokenized_.end(); ++i, ++j){
    unordered_map <int, double> t_var__summed;
    for(vector<int>::iterator jj = j->begin(); jj != j->end(); ++jj){
      int target_word = *jj;
      t_var__summed[target_word] = 0;
      for(vector<int>::iterator ii = i->begin(); ii != i->end(); ++ii){
	int source_word = *ii;
	t_var__summed[target_word] += t_var_[target_word][source_word];
      }
    }
    for(vector<int>::iterator jj = j->begin(); jj != j->end(); ++jj){
      int target_word = *jj;
      for(vector<int>::iterator ii = i->begin(); ii != i->end(); ++ii){
	int source_word = *ii;
	delta = t_var_[target_word][source_word]/t_var__summed[target_word];
	pair_of_word_counts_[source_word][target_word] += delta;
	source_word_counts_[source_word] += delta;
      }
    }
  }
}

void IBM1::maximization()
{
  for(vector<vector<int> >::iterator i = source_sentences_tokenized_.begin(), j = target_sentences_tokenized_.begin(); i != source_sentences_tokenized_.end() && j != target_sentences_tokenized_.end(); ++i, ++j){
    for(vector<int>::iterator jj = j->begin(); jj != j->end(); ++jj){
      for(vector<int>::iterator ii = i->begin(); ii != i->end(); ++ii){
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

    vector <string> source_sentence_words_original, target_sentence_words_original;
    vector <int> source_sentence_words, target_sentence_words;
    string word;

    while (source_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      source_sentence_words_original.push_back(word); // Here we tokenize the sentence.                                                                   
    }
    
    while (target_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      target_sentence_words_original.push_back(word);
    }

    string data_pair_number = source_sentence_words_original[1].substr(5,4);
    source_sentence_words_original.pop_back();
    target_sentence_words_original.pop_back();
    
    source_sentence_words_original.erase(source_sentence_words_original.begin(),source_sentence_words_original.begin()+2);
    target_sentence_words_original.erase(target_sentence_words_original.begin(),target_sentence_words_original.begin()+2);
    
    source_sentence_words_original.insert(source_sentence_words_original.begin(),"null");

    for(size_t i = 0; i != source_sentence_words_original.size(); ++i){
      source_sentence_words.push_back(source_word_int_tokens_[source_sentence_words_original[i]]);
    }

    for(size_t j = 0; j != target_sentence_words_original.size(); ++j){
      target_sentence_words.push_back(target_word_int_tokens_[target_sentence_words_original[j]]);
    }

    for(size_t j = 0; j < target_sentence_words.size();++j){
      int target_word = target_sentence_words[j];
      int a_best = 0;
      int a_best_source_word = source_sentence_words[a_best];
      for(int i = 0; i < source_sentence_words.size();++i){
	int source_word = source_sentence_words[i];
	if(t_var_[j][i] > t_var_[j][a_best]){
	  a_best = i;
	  a_best_source_word = source_sentence_words[a_best];
	}
      }
      if(a_best != 0){
	gotten_alignments_file << data_pair_number << " " << a_best << " " << j+1 <<endl; 
      }
    }
  }
  gotten_alignments_file.close();
}

void IBM1::print()
{
  // Why does it complain if we don't use const iterator? Because the object is const!
  for(unordered_map< int, unordered_map<int, double> >::const_iterator j = t_var_.begin(); j != t_var_.end(); ++j){
      for(unordered_map<int, double>::const_iterator i= (j->second).begin(); i != (j->second).end(); ++i){
	int target_word = j->first;
	int source_word = i->first;
	double probability = (t_var_[j->first])[i->first];
	cout <<"t_var_[" << target_word << "|" << source_word << "] = " << probability << endl;
      }
  }
}

void IBM1::write_variables(string file_path){
  FILE * ofp;
  ofp = fopen(file_path.c_str(),"w");
  for(unordered_map< int, unordered_map<int, double> >::iterator j = t_var_.begin(); j != t_var_.end(); ++j){
    for(unordered_map<int, double>::iterator i= (j->second).begin(); i != (j->second).end(); ++i){
      int target_word(j->first); 
      int source_word(i->first);
      double probability(t_var_[target_word][source_word]);
      fprintf(ofp,"%d %d %f\n",target_word,source_word,probability);
    }
  }
  fclose(ofp);
}

void IBM1::read_variables(string file_path){
  FILE * ifp;
  ifp = fopen(file_path.c_str(),"r");
  int target_word, source_word;
  float probability;
  while(fscanf(ifp, "%d %d %f", &target_word, &source_word, &probability) == 3){
    t_var_[target_word][source_word] = static_cast<double>(probability);
  }
  fclose(ifp);
}
