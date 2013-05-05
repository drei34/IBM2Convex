#include "IBM2Convex.h"
using namespace std;

IBM2Convex::IBM2Convex(string file_name, string initialization_type, double lambda, double c, double alpha, double theta):IBM1(file_name), initialization_type_(initialization_type), lambda_(lambda), c_(c), alpha_(alpha), theta_(theta), max_source_sentence_length_(1), max_target_sentence_length_(1), number_of_mistakes_(0), block_percentage_(.005), iteration_number_(0)
{
  if(initialization_type_ == "ibm1"){
    cout << "We are initializing with IBM1 via EM!" << endl;
    em(10);
  } // Otherwise, they are uniformly initialized.
  else{
    cout << "We did not initialize with IBM1, so t variables will be initialized uniformly!" << endl;
  }

  for(unordered_map<string, unordered_set < string > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
    string source_word = i->first;
    unordered_set<string> target_words = i->second;
    for(unordered_set<string>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
      string target_word = *j;
      t_var_average_[target_word][source_word] = t_var_[target_word][source_word];
      t_var_best_overall_[target_word][source_word] = t_var_[target_word][source_word];
    }
  }
  
  get_max_sentence_length();
  initialize_d_variables_uniformly();

  N_ = source_sentences_.size();
  for(size_t k = 0; k < N_; ++k){
    shuffled_training_indicies_.push_back(k);
  }
  learning_rate_ = c_/(1 + number_of_mistakes_);
  
  B_ = static_cast<int>(block_percentage_*N_);

  objectives_.push_back(compute_objective(t_var_,d_var_));
  objectives_average_.push_back(compute_objective(t_var_average_,d_var_average_));
  objectives_best_overall_.push_back(compute_objective(t_var_best_overall_,d_var_best_overall_));
}

void IBM2Convex::initialize_d_variables_uniformly()
{
  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      d_var_[i][j+1] = static_cast<double>(1)/max_source_sentence_length_;
      d_var_average_[i][j+1] = static_cast<double>(1)/max_source_sentence_length_;
      d_var_best_overall_[i][j+1] = static_cast<double>(1)/max_source_sentence_length_;
      d_var_gradients_[i][j+1] = 0;
    }
  }
}

void IBM2Convex::get_max_sentence_length()
{
  for(vector < vector<string> >::iterator source_sentence_iterator = source_sentences_tokenized_.begin(); source_sentence_iterator != source_sentences_tokenized_.end(); ++source_sentence_iterator){
    max_source_sentence_length_ = max_source_sentence_length_ < source_sentence_iterator->size() ? source_sentence_iterator->size() : max_source_sentence_length_;
  }
  for(vector < vector<string> >::iterator target_sentence_iterator = target_sentences_tokenized_.begin(); target_sentence_iterator != target_sentences_tokenized_.end(); ++target_sentence_iterator){
    max_target_sentence_length_ = max_target_sentence_length_ < target_sentence_iterator->size() ? target_sentence_iterator->size() : max_target_sentence_length_;
  }
}

void IBM2Convex::check_for_zero_gradients(){
  /*
    This function checks to see if all the gradients are zero after we have optimized a bactch of size B_.
   */
  for(unordered_map<string, unordered_set < string > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
    string source_word = i->first;
    unordered_set<string> target_words = i->second;
    for(unordered_set<string>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
      string target_word = *j;
      if(t_var_gradients_[target_word][source_word] != 0){
	cout << "This is not zero!" << endl;
	cout << "The variables is t[" << target_word << "|" << source_word << "] = " << t_var_gradients_[target_word][source_word] << endl; 
	exit(1);
      }
    }
  }
  
  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      if(d_var_gradients_[i][j+1] != 0){
	cout << "This is not zero!" << endl;
	cout << "The variables is d[" << i << "|" << j+1 << "] = " << d_var_gradients_[i][j+1] << endl; 
	exit(1);
      }
    }
  }

  cout << "We have checked all the gradeitns and they are all reset to zero!" << endl;
}

void IBM2Convex::compute_gradients(int random_index){
  vector<string> source_sentence = source_sentences_tokenized_[random_index];
  vector<string> target_sentence = target_sentences_tokenized_[random_index];

  for(size_t j = 0; j < target_sentence.size(); ++j){
    string target_word = target_sentence[j];
    float Q_sum_1 = 0;
    float Q_sum_2 = 0;
    Q_sum_1 += lambda_;
    Q_sum_2 += lambda_;
    for(size_t i = 0; i < source_sentence.size(); ++i){
      string source_word = source_sentence[i];
      Q_sum_1 += min(alpha_*t_var_[target_word][source_word],d_var_[i][j+1]);
      Q_sum_2 += t_var_[target_word][source_word];
    }
    for(size_t i = 0; i < source_sentence.size(); ++i){
      string source_word = source_sentence[i];
      if(alpha_*t_var_[target_word][source_word] <= d_var_[i][j+1]){
	t_var_gradients_[target_word][source_word] += -(theta_)*(alpha_/Q_sum_1);
      }
      else{
	d_var_gradients_[i][j+1] += -(theta_)*(1/Q_sum_1);
	d_var_gradients_changed_.insert(pair<int,int>(i,j+1));
      }
      t_var_gradients_[target_word][source_word] += -(1-theta_)*(1/Q_sum_2);
      t_var_gradients_changed_.insert(pair<string,string>(target_word,source_word));
    }
  }
}

double IBM2Convex::compute_objective(unordered_map< string, unordered_map <string,double > > & t_var, unordered_map <int, unordered_map<int, double> > & d_var)
{
  double objective = 0;
  for(vector< vector<string> >::iterator source_sentence_iterator = source_sentences_tokenized_.begin(), target_sentence_iterator = target_sentences_tokenized_.begin(); source_sentence_iterator != source_sentences_tokenized_.end() && target_sentence_iterator != target_sentences_tokenized_.end(); ++source_sentence_iterator, ++target_sentence_iterator){
    vector <string> source_sentence =  *source_sentence_iterator;
    vector <string> target_sentence  = *target_sentence_iterator; 
    for(size_t j = 0; j < target_sentence.size(); ++j){
      string target_word = target_sentence[j];
      double Q_sum_1 = 0;
      Q_sum_1 += lambda_;
      double Q_sum_2 = 0;
      Q_sum_2 += lambda_;
      for(size_t i = 0; i < source_sentence.size(); ++i){
	string source_word = source_sentence[i];
	Q_sum_1 += alpha_*t_var[target_word][source_word] < d_var[i][j+1] ? alpha_*t_var[target_word][source_word] : d_var[i][j+1];
	Q_sum_2 += t_var[target_word][source_word];
      }
      objective += -(theta_)*log(Q_sum_1);
      objective += -(1-theta_)*log(Q_sum_2);
    }
  }
  objective *= static_cast<double>(1)/N_;
  return objective;
}

void IBM2Convex::optimize_seg()
{
  vector<int> index_block;
  unordered_set<string> source_words_in_block;
  unordered_set<int> target_lengths_in_block;
  int max_target_length_in_block = 0;

  random_shuffle(shuffled_training_indicies_.begin(),shuffled_training_indicies_.end());

  double old_objective = objectives_.back();
  double old_objective_average = objectives_average_.back();

  for(size_t index = 0; index < shuffled_training_indicies_.size(); ++index){
    size_t random_index = shuffled_training_indicies_[index];
    
    cout << "We are updatiing the " << index << " example which is index " << random_index << "!" << endl;

    index_block.push_back(random_index);

    if(index_block.size() < B_){
      if(index < N_ - 1){
	continue;
      }
    }

    // If we got here, we are actually going to update.                                                                                                      

    cout << "Optimizing on the block! We've seen a block of at most " << B_ << " training examples!" << endl;
    cout << "Updating the gradients on the block indicies!" << endl;
    
    for(size_t index = 0; index < index_block.size(); ++index){
      size_t block_random_index = index_block[index];
      compute_gradients(block_random_index);
      vector<string> source_sentence = source_sentences_tokenized_[block_random_index];
      vector<string> target_sentence = target_sentences_tokenized_[block_random_index];
      for(size_t i = 0; i < source_sentence.size(); ++i){
	string source_word = source_sentence[i];
	source_words_in_block.insert(source_word);
      }
      target_lengths_in_block.insert(target_sentence.size());
    }
    max_target_length_in_block = *(max_element(target_lengths_in_block.begin(),target_lengths_in_block.end()));
    cout << "Updating the gradients!" << endl;

    for(unordered_set< pair <string, string> >::iterator it = t_var_gradients_changed_.begin(); it != t_var_gradients_changed_.end(); ++it){
	string target_word = it->first;
	string source_word = it->second;
	t_var_gradients_[target_word][source_word] *= static_cast<float>(1)/B_;
      }
	  
    for(unordered_set< pair <int, int> >::iterator it = d_var_gradients_changed_.begin(); it != d_var_gradients_changed_.end(); ++it){
	int i = it->first;
	int j = (it->second) - 1;
	d_var_gradients_[i][j+1] *= static_cast<float>(1)/B_;
      }

    bool over_flow = true;
    long double fake_t_var = 0;
    long double fake_d_var = 0;

    while(over_flow){
      bool t_var_exception_thrown = false;
      bool d_var_exception_thrown = false;
      for(unordered_map < string, unordered_set < string > >::iterator source_dictionary_iterator = source_dictionary_.begin(); source_dictionary_iterator != source_dictionary_.end(); ++source_dictionary_iterator){
	string source_word = source_dictionary_iterator->first;
	for(unordered_set<string>::iterator target_word_iterator = (source_dictionary_iterator->second).begin();target_word_iterator != (source_dictionary_iterator->second).end(); ++target_word_iterator){
	  try{
	    string target_word = *target_word_iterator;
	    fake_t_var = t_var_[target_word][source_word]*exp(-learning_rate_*(t_var_gradients_[target_word][source_word]));  
	  }
	  catch(std::overflow_error){
	    learning_rate_ *= .5;
	    cout << "Over Flow Error!" << endl;
	    cout << "Halving the learning rate! The new learning rate is" << learning_rate_ << "!" <<endl;
	    t_var_exception_thrown = true;
	  }
	}
      }
      for(int j = 0; j < max_target_sentence_length_;++j){
	if(t_var_exception_thrown || d_var_exception_thrown){
	  break;
	}
	for(int i = 0; i < max_source_sentence_length_; ++i){
	  try{
	    fake_d_var = (d_var_[i][j+1])*exp(-learning_rate_*(d_var_gradients_[i][j+1]));
	  }
	  catch(std::overflow_error){
	    learning_rate_ *= .5;
	    cout << "OverFlow Error!" << endl;
	    cout << "Halving the learning rate! The new learning rate is" << learning_rate_ << "!" <<endl;
	    d_var_exception_thrown = true;
	  }
	}
      }
      if(!t_var_exception_thrown && !d_var_exception_thrown){
	over_flow = false;
      }
    }

    for(unordered_set < string >::iterator source_words_in_block_iterator = source_words_in_block.begin(); source_words_in_block_iterator != source_words_in_block.end(); ++source_words_in_block_iterator){
      long double Z_t_var_sum(0);
      string source_word = *source_words_in_block_iterator;
      for(unordered_set <string>::iterator target_words_in_block_iterator = source_dictionary_[source_word].begin(); target_words_in_block_iterator != source_dictionary_[source_word].end(); ++target_words_in_block_iterator){
	string target_word = *target_words_in_block_iterator;
	t_var_[target_word][source_word] = t_var_[target_word][source_word]*exp(-learning_rate_*(t_var_gradients_[target_word][source_word]));
	Z_t_var_sum += t_var_[target_word][source_word];
      }
      for(unordered_set <string>::iterator target_words_in_block_iterator = source_dictionary_[source_word].begin(); target_words_in_block_iterator != source_dictionary_[source_word].end(); ++target_words_in_block_iterator){
	string target_word = *target_words_in_block_iterator;
	t_var_[target_word][source_word] /= Z_t_var_sum;
      }
    }

    for(int j = 0; j < max_target_sentence_length_;++j){
      long double Z_d_var_sum(0);
      for(int i = 0; i < max_source_sentence_length_; ++i){
	d_var_[i][j+1] = (d_var_[i][j+1])*exp(-learning_rate_*(d_var_gradients_[i][j+1]));
	Z_d_var_sum += d_var_[i][j+1];
      }
      for(int i = 0; i < max_source_sentence_length_; ++i){
	d_var_[i][j+1] /= Z_d_var_sum;
      }
    }

    for(unordered_set< pair <string, string> >::iterator it = t_var_gradients_changed_.begin(); it != t_var_gradients_changed_.end(); ++it){
	string target_word = it->first;
	string source_word = it->second;
	t_var_gradients_[target_word][source_word] = 0;
      }
	  
    for(unordered_set< pair <int, int> >::iterator it = d_var_gradients_changed_.begin(); it != d_var_gradients_changed_.end(); ++it){
	int i = it->first;
	int j = it->second - 1;
	d_var_gradients_[i][j+1] = 0;
      }

    cout << "We've gone a block of data!" << endl;
    index_block.clear();
    source_words_in_block.clear();
    target_lengths_in_block.clear();
    max_target_length_in_block = 0;
    t_var_gradients_changed_.clear();
    d_var_gradients_changed_.clear();
  }
  
  iteration_number_++;


  for(unordered_set < string >::iterator source_words_it = source_words_.begin(); source_words_it != source_words_.end(); ++source_words_it){
      string source_word = *source_words_it;
      for(unordered_set <string>::iterator target_words_it = source_dictionary_[source_word].begin(); target_words_it != source_dictionary_[source_word].end(); ++target_words_it){
	string target_word = *target_words_it;
	t_var_average_[target_word][source_word] = (iteration_number_)*(t_var_average_[target_word][source_word]) + t_var_[target_word][source_word];
	t_var_average_[target_word][source_word] *= static_cast<double>(1)/(iteration_number_+1);
      }      
  }

  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      d_var_average_[i][j+1] = (iteration_number_)*(d_var_average_[i][j+1]) + d_var_[i][j+1];
      d_var_average_[i][j+1] *= static_cast<double>(1)/(iteration_number_+1);
    }
  }

  double new_objective = compute_objective(t_var_, d_var_);
  double new_objective_average = compute_objective(t_var_average_, d_var_average_);

  // Here we are changing the learning rate depending on how the averaged solutions does; should we use th current solution?                                 

  if(new_objective_average > old_objective_average){
    number_of_mistakes_++;
    learning_rate_ = c_/(1+number_of_mistakes_);
  }
  
  // This is where we update the running best solution.                                                                                                      
  // Notice that the appending is done here and not at the end of the function call, line the other calls.                                                   

  if(objectives_best_overall_.back() > new_objective){
    for(unordered_map < string, unordered_map < string, double> >::iterator t_var_iterator = t_var_.begin(); t_var_iterator != t_var_.end(); ++t_var_iterator){
      string target_word = t_var_iterator->first;
      for(unordered_map < string, double >::iterator source_words_iterator = (t_var_iterator->second).begin(); source_words_iterator != (t_var_iterator->second).end(); ++source_words_iterator){
	string source_word = source_words_iterator->first;
	t_var_best_overall_[target_word][source_word] = t_var_[target_word][source_word]; 
      }  
    }
    for(int i = 0; i < max_source_sentence_length_; ++i){
      for(int j = 0; j < max_target_sentence_length_; ++j){
	d_var_best_overall_[i][j+1] = d_var_[i][j+1];
      }
    }
    objectives_best_overall_.push_back(new_objective);
  }
  else{
    //If the current solution is not any better, well then we just leave it alone.                                                                          
    objectives_best_overall_.push_back(objectives_best_overall_.back());
  }
  
  objectives_.push_back(new_objective);
  objectives_average_.push_back(new_objective_average);

}

void IBM2Convex::write_alignments(string evaluation_file_name = "trial")
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

  ofstream gotten_alignments_file, gotten_alignments_file_average, gotten_alignments_file_best_overall;
  string gotten_alignments_file_name = "ibm2convex_gotten_alignments_"+translation_type_+"_final.txt";
  string gotten_alignments_file_name_average = "ibm2convex_gotten_alignments_"+translation_type_+"_average.txt";
  string gotten_alignments_file_name_best_overall = "ibm2convex_gotten_alignments_"+translation_type_+"_best_overall.txt";
  gotten_alignments_file.open(gotten_alignments_file_name.c_str());
  gotten_alignments_file_average.open(gotten_alignments_file_name_average.c_str());
  gotten_alignments_file_best_overall.open(gotten_alignments_file_name_best_overall.c_str());

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

    for(size_t j = 0; j < target_sentence_words.size();++j){
      string target_word = target_sentence_words[j];
      size_t a_best = 0;
      size_t a_best_average = 0;
      size_t a_best_overall = 0;
      string source_word_best = source_sentence_words[a_best];
      string source_word_best_average = source_sentence_words[a_best_average];
      string source_word_best_overall = source_sentence_words[a_best_overall];
      for(size_t i = 0; i < source_sentence_words.size();++i){
        string source_word = source_sentence_words[i];
        if(t_var_[target_word][source_word]*d_var_[i][j+1] > t_var_[target_word][source_word_best]*d_var_[a_best][j+1]){
          a_best = i;
          source_word_best = source_sentence_words[a_best];
	}
	if(t_var_average_[target_word][source_word]*d_var_average_[i][j+1] > t_var_average_[target_word][source_word_best_average]*d_var_average_[a_best_average][j+1]){
          a_best_average = i;
          source_word_best_average = source_sentence_words[a_best_average];
	}
	if(t_var_best_overall_[target_word][source_word]*d_var_best_overall_[i][j+1] > t_var_best_overall_[target_word][source_word_best_overall]*d_var_best_overall_[a_best_overall][j+1]){
          a_best_overall = i;
          source_word_best_overall = source_sentence_words[a_best_overall];
	}
      }
      if(a_best != 0){
	gotten_alignments_file << data_pair_number << " " << a_best << " " << j+1 << endl;
      }
      if(a_best_average != 0){
	gotten_alignments_file_average << data_pair_number << " " << a_best_average << " " << j+1 << endl;
      }
      if(a_best_overall != 0){
	gotten_alignments_file_best_overall << data_pair_number << " " << a_best_overall << " " << j+1 << endl;
      }
    }
  }
  gotten_alignments_file.close();
  gotten_alignments_file_average.close();
  gotten_alignments_file_best_overall.close();
}

void IBM2Convex::print()
{
  /*
    This print the variables!
    NOT DONE AS OF YET.
   */
  for(unordered_map< string, unordered_map<string, double> >::const_iterator t_var_iterator = t_var_average_.begin(); t_var_iterator != t_var_average_.end(); ++t_var_iterator){
    string target_word(t_var_iterator->first);
    for(unordered_map<string, double>::const_iterator source_word_iterator = (t_var_iterator->second).begin(); source_word_iterator != (t_var_iterator->second).end(); ++source_word_iterator){
      string source_word(source_word_iterator->first);
      double t_value;
      cout <<"t_var_average_[" <<  target_word << "|" << source_word << "] = " << t_var_average_[target_word][source_word] << endl;
    }
  }

  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      cout <<"d_var_average_[" << i << "|" << j+1 << "] = " << d_var_average_[i][j+1] << endl;
    }
  }
}


