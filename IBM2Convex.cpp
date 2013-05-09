#include "IBM2Convex.h"
using namespace std;

IBM2Convex::IBM2Convex(string file_name, string initialization_type, double block_percentage, double lambda, double c, int total_iterations, double alpha, double theta): AlignmentCorpus(file_name), initialization_type_(initialization_type), lambda_(lambda), c_(c), total_iterations_(total_iterations), alpha_(alpha), theta_(theta), max_source_sentence_length_(1), max_target_sentence_length_(1), number_of_mistakes_(0), block_percentage_(block_percentage), iteration_number_(0)
{
  N_ = source_sentences_tokenized_.size();

  t_var_.resize(number_source_words_);
  t_var_average_.resize(number_source_words_);
  t_var_best_overall_.resize(number_source_words_);
  t_var_gradients_.resize(number_source_words_);

  if(initialization_type_ == "ibm1"){
    cout << "We are initializing with IBM1 via EM!" << endl;
    IBM1 ibm1(file_name);
    string N_string;      
    ostringstream convert;
    convert << N_;
    N_string = convert.str(); 
    string data_file_name(string("/proj/mlnlp/andrei/IBM2Convex C++/data/ibm1_data_") + translation_type_ + "_"+ N_string + string("_sentence_pairs.txt"));

    if(FILE * ifp = fopen(data_file_name.c_str(),"r")){
      ibm1.read_variables(data_file_name);
      fclose(ifp);
    }
    else{
      ibm1.em(15);
      ibm1.write_variables(data_file_name);
    }

    for(unordered_map<int, unordered_set < int > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
      int source_word = i->first;
      unordered_set<int> target_words = i->second;
      for(unordered_set<int>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
	int target_word = *j;
	t_var_[source_word].push_back(ibm1.t_var_[target_word][source_word]);
	t_var_average_[source_word].push_back(ibm1.t_var_[target_word][source_word]);
	t_var_best_overall_[source_word].push_back(ibm1.t_var_[target_word][source_word]);
	t_var_gradients_[source_word].push_back(0.0);
	t_var_lookup_[target_word][source_word] = t_var_[source_word].size()-1; // This is the index of the target_word probaility in the new representation.
      }
    }
  } 
  else{
    cout << "We did not initialize with IBM1, so t variables will be initialized uniformly!" << endl;
    for(unordered_map<int, unordered_set < int > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
      int source_word = i->first;
      unordered_set<int> target_words = i->second;
      for(unordered_set<int>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
	int target_word = *j;
	t_var_[source_word].push_back(static_cast<double>(1)/target_words.size());
	t_var_average_[source_word].push_back(static_cast<double>(1)/target_words.size());
	t_var_best_overall_[source_word].push_back(static_cast<double>(1)/target_words.size());
	t_var_gradients_[source_word].push_back(static_cast<double>(0));
	t_var_lookup_[target_word][source_word] = t_var_[source_word].size()-1; // This is the index of the target_word probaility in the new representation.
      }
    }
  }
  // CEHCK THIS!
    
  get_max_sentence_length();
  initialize_d_variables_uniformly();

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
  d_var_.resize(max_source_sentence_length_);
  d_var_average_.resize(max_source_sentence_length_);
  d_var_best_overall_.resize(max_source_sentence_length_);
  d_var_gradients_.resize(max_source_sentence_length_);

  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      d_var_[i].push_back(static_cast<double>(1)/max_source_sentence_length_);
      d_var_average_[i].push_back(static_cast<double>(1)/max_source_sentence_length_);
      d_var_best_overall_[i].push_back(static_cast<double>(1)/max_source_sentence_length_);
      d_var_gradients_[i].push_back(0.0);
    }
  }
}

void IBM2Convex::get_max_sentence_length()
{
  for(vector < vector<int> >::iterator source_sentence_iterator = source_sentences_tokenized_.begin(); source_sentence_iterator != source_sentences_tokenized_.end(); ++source_sentence_iterator){
    max_source_sentence_length_ = max_source_sentence_length_ < source_sentence_iterator->size() ? source_sentence_iterator->size() : max_source_sentence_length_;
  }
  for(vector < vector<int> >::iterator target_sentence_iterator = target_sentences_tokenized_.begin(); target_sentence_iterator != target_sentences_tokenized_.end(); ++target_sentence_iterator){
    max_target_sentence_length_ = max_target_sentence_length_ < target_sentence_iterator->size() ? target_sentence_iterator->size() : max_target_sentence_length_;
  }
}

void IBM2Convex::check_for_zero_gradients(){
  /*
    This function checks to see if all the gradients are zero after we have optimized a bactch of size B_.
   */
  for(unordered_map<int, unordered_set < int > >::iterator i = source_dictionary_.begin(); i != source_dictionary_.end(); ++i){
    int source_word = i->first;
    unordered_set<int> target_words = i->second;
    for(unordered_set<int>::iterator j = target_words.begin(); j!= target_words.end(); ++j){
      int target_word = *j;
      if(t_var_gradients_[source_word][t_var_lookup_[target_word][source_word]] != 0){
	cout << "This is not zero!" << endl;
	cout << "The variables is t[" << target_word << "|" << source_word << "] = " << t_var_gradients_[target_word][source_word] << endl; 
	exit(1);
      }
    }
  }
  
  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      if(d_var_gradients_[i][j] != 0){
	cout << "This is not zero!" << endl;
	cout << "The variables is d[" << i << "|" << j << "] = " << d_var_gradients_[i][j] << endl; 
	exit(1);
      }
    }
  }

  cout << "We have checked all the gradeitns and they are all reset to zero!" << endl;
}

void IBM2Convex::compute_gradients(int random_index){
  vector<int> source_sentence = source_sentences_tokenized_[random_index];
  vector<int> target_sentence = target_sentences_tokenized_[random_index];

  for(size_t j = 0; j < target_sentence.size(); ++j){
    int target_word = target_sentence[j];
    double Q_sum_1 = 0;
    double Q_sum_2 = 0;
    Q_sum_1 += lambda_;
    Q_sum_2 += lambda_;
    for(size_t i = 0; i < source_sentence.size(); ++i){
      int source_word = source_sentence[i];
      Q_sum_1 += min(alpha_*t_var_[source_word][t_var_lookup_[target_word][source_word]],d_var_[i][j]);
      Q_sum_2 += t_var_[source_word][t_var_lookup_[target_word][source_word]];
    }
    for(size_t i = 0; i < source_sentence.size(); ++i){
      int source_word = source_sentence[i];
      if(alpha_*t_var_[source_word][t_var_lookup_[target_word][source_word]] <= d_var_[i][j]){
	t_var_gradients_[source_word][t_var_lookup_[target_word][source_word]] += -(theta_)*(alpha_/Q_sum_1);
      }
      else{
	d_var_gradients_[i][j] += -(theta_)*(1/Q_sum_1);
	d_var_gradients_changed_.insert(pair<int,int>(i,j));
      }
      t_var_gradients_[source_word][t_var_lookup_[target_word][source_word]] += -(1-theta_)*(1/Q_sum_2);
      t_var_gradients_changed_.insert(pair<int,int>(target_word,source_word));
    }
  }
}

double IBM2Convex::compute_objective(vector<vector<double> > & t_var, vector<vector<double> > & d_var)
{
  double objective = 0;
  for(vector<vector<int> >::iterator source_sentence_iterator = source_sentences_tokenized_.begin(), target_sentence_iterator = target_sentences_tokenized_.begin(); source_sentence_iterator != source_sentences_tokenized_.end() && target_sentence_iterator != target_sentences_tokenized_.end(); ++source_sentence_iterator, ++target_sentence_iterator){
    vector <int> source_sentence =  *source_sentence_iterator;
    vector <int> target_sentence  = *target_sentence_iterator; 
    for(size_t j = 0; j < target_sentence.size(); ++j){
      int target_word = target_sentence[j];
      double Q_sum_1 = 0;
      Q_sum_1 += lambda_;
      double Q_sum_2 = 0;
      Q_sum_2 += lambda_;
      for(size_t i = 0; i < source_sentence.size(); ++i){
	int source_word = source_sentence[i];
	Q_sum_1 += alpha_*t_var[source_word][t_var_lookup_[target_word][source_word]] < d_var[i][j] ? alpha_*t_var[source_word][t_var_lookup_[target_word][source_word]] : d_var[i][j];
	Q_sum_2 += t_var[source_word][t_var_lookup_[target_word][source_word]];
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
  clock_t start;
  vector<int> index_block;
  set<int> source_words_in_block;
  set<int> target_lengths_in_block;
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
    
    start  = clock();
    for(size_t index = 0; index < index_block.size(); ++index){
      size_t block_random_index = index_block[index];
      compute_gradients(block_random_index);
      vector<int> source_sentence = source_sentences_tokenized_[block_random_index];
      vector<int> target_sentence = target_sentences_tokenized_[block_random_index];
      for(size_t i = 0; i < source_sentence.size(); ++i){
	int source_word = source_sentence[i];
	source_words_in_block.insert(source_word);
      }
      target_lengths_in_block.insert(target_sentence.size());
    }
    printf("Time for computing the gradients on the block indicies: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    max_target_length_in_block = *(max_element(target_lengths_in_block.begin(),target_lengths_in_block.end()));
    cout << "Updating the gradients!" << endl;

    start  = clock();
    for(unordered_set< pair <int, int> >::iterator it = t_var_gradients_changed_.begin(); it != t_var_gradients_changed_.end(); ++it){
	int target_word = it->first;
	int source_word = it->second;
	t_var_gradients_[source_word][t_var_lookup_[target_word][source_word]] *= static_cast<float>(1)/B_;
      }
	  
    for(unordered_set< pair <int, int> >::iterator it = d_var_gradients_changed_.begin(); it != d_var_gradients_changed_.end(); ++it){
	int i = it->first;
	int j = (it->second);
	d_var_gradients_[i][j] *= static_cast<float>(1)/B_;
      }
    printf("Time elapsed for normalizing the gradients: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    bool over_flow = true;
    double fake_t_var = 0;
    double fake_d_var = 0;

    start = clock();
    while(over_flow){
      bool t_var_exception_thrown = false;
      bool d_var_exception_thrown = false;
      
      for(set<int>::iterator source_words_in_block_iterator = source_words_in_block.begin(); source_words_in_block_iterator != source_words_in_block.end(); ++source_words_in_block_iterator){
	int source_word = *source_words_in_block_iterator;
	for(size_t j  = 0; j < t_var_[source_word].size(); ++j){
	  try{
	    fake_t_var = t_var_[source_word][j]*exp(-learning_rate_*t_var_gradients_[source_word][j]);  
	    fake_t_var = 0;
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
	    fake_d_var = (d_var_[i][j])*exp(-learning_rate_*(d_var_gradients_[i][j]));
	    fake_d_var = 0;
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
    printf("Time for checking there is no blow up: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    
    start = clock();
    for(set<int>::iterator source_words_in_block_iterator = source_words_in_block.begin(); source_words_in_block_iterator != source_words_in_block.end(); ++source_words_in_block_iterator){
      double Z_t_var_sum(0);
      int source_word = *source_words_in_block_iterator;
      size_t number_of_target_words(t_var_[source_word].size());
      for(size_t j = 0; j < number_of_target_words; ++j){
	t_var_[source_word][j] = t_var_[source_word][j]*exp(-learning_rate_*t_var_gradients_[source_word][j]);
      }
      Z_t_var_sum = accumulate(t_var_[source_word].begin(),t_var_[source_word].end(),0.0);
      transform(t_var_[source_word].begin(), t_var_[source_word].end(), t_var_[source_word].begin(),bind1st(multiplies<double>(),1/Z_t_var_sum));
    }

    for(size_t j = 0; j < max_target_sentence_length_;++j){
      double Z_d_var_sum(0);
      for(size_t i = 0; i < max_source_sentence_length_; ++i){
	d_var_[i][j] = (d_var_[i][j])*exp(-learning_rate_*(d_var_gradients_[i][j]));
	Z_d_var_sum += d_var_[i][j];
      }
      for(size_t i = 0; i < max_source_sentence_length_; ++i){
	d_var_[i][j] /= Z_d_var_sum;
      }
    }
    printf("Time elapsed for block updates: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    start = clock();
    for(unordered_set< pair <int, int> >::iterator it = t_var_gradients_changed_.begin(); it != t_var_gradients_changed_.end(); ++it){
	int target_word = it->first;
	int source_word = it->second;
	t_var_gradients_[source_word][t_var_lookup_[target_word][source_word]] = 0;
    }
	  
    for(unordered_set< pair <int, int> >::iterator it = d_var_gradients_changed_.begin(); it != d_var_gradients_changed_.end(); ++it){
	int i = it->first;
	int j = it->second;
	d_var_gradients_[i][j] = 0;
    }
    printf("Time elapsed for reseting gradients to zero: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    cout << "We've gone a block of data!" << endl;
    index_block.clear();
    source_words_in_block.clear();
    target_lengths_in_block.clear();
    max_target_length_in_block = 0;
    t_var_gradients_changed_.clear();
    d_var_gradients_changed_.clear();
  }
  
  iteration_number_++;

  start = clock();
  for(unordered_set < int >::iterator source_words_iterator = source_words_.begin(); source_words_iterator != source_words_.end(); ++source_words_iterator){
      int source_word = *source_words_iterator;
      for(size_t j = 0; j < t_var_[source_word].size(); ++j){
	t_var_average_[source_word][j] = (iteration_number_)*(t_var_average_[source_word][j]) + t_var_[source_word][j];
	t_var_average_[source_word][j] *= static_cast<double>(1)/(iteration_number_+1);
      }
  }

  for(int j = 0; j < max_target_sentence_length_; ++j){
    for(int i = 0; i < max_source_sentence_length_; ++i){
      d_var_average_[i][j] = (iteration_number_)*(d_var_average_[i][j]) + d_var_[i][j];
      d_var_average_[i][j] *= static_cast<double>(1)/(iteration_number_+1);
    }
  }
  printf("Time elapsed for computing the averaged parameters: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

  start = clock();
  double new_objective = compute_objective(t_var_, d_var_);
  double new_objective_average = compute_objective(t_var_average_, d_var_average_);
  printf("Time elapsed for getting the new objectives: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

  // Here we are changing the learning rate depending on how the averaged solutions does; should we use th current solution?                                 

  
  if(new_objective_average > old_objective_average){
    number_of_mistakes_++;
    learning_rate_ = c_/(1+number_of_mistakes_);
  }
  
  // This is where we update the running best solution.                                                                                                      
  // Notice that the appending is done here and not at the end of the function call, line the other calls.                                                   

  start = clock();
  if(objectives_best_overall_.back() > new_objective){
    for(unordered_set < int >::iterator source_words_iterator = source_words_.begin(); source_words_iterator != source_words_.end(); ++source_words_iterator){
      int source_word = *source_words_iterator;
      for(size_t j = 0; j < t_var_[source_word].size(); ++j){
	t_var_best_overall_[source_word][j] = t_var_[source_word][j];
      }
    }

    for(int i = 0; i < max_source_sentence_length_; ++i){
      for(int j = 0; j < max_target_sentence_length_; ++j){
	d_var_best_overall_[i][j] = d_var_[i][j];
      }
    }
    objectives_best_overall_.push_back(new_objective);
  }
  else{
    // If the current solution is not any better, then we just leave it alone.                                                                          
    objectives_best_overall_.push_back(objectives_best_overall_.back());
  }
  printf("Time elapsed for getting the best new best_overall variables: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

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
  string extra_marker = string("_N_") + number_to_string<int>(N_) + string("_block_percentage_") + number_to_string<double>(block_percentage_) + string("_lambda_") + number_to_string<double>(lambda_) + string("_c_") + number_to_string<double>(c_) + string("_alpha_") + number_to_string<double>(alpha_) + string("_theta_") + number_to_string<double>(theta_) + string("_evaluation_file_name_") + string(evaluation_file_name);
  string gotten_alignments_file_name = "ibm2convex_gotten_alignments_" + translation_type_ + extra_marker + "_final.txt";
  string gotten_alignments_file_name_average = "ibm2convex_gotten_alignments_" + translation_type_ + extra_marker + "_average.txt";
  string gotten_alignments_file_name_best_overall = "ibm2convex_gotten_alignments_" + translation_type_ + extra_marker + "_best_overall.txt";
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
      int a_best_average = 0;
      int a_best_overall = 0;
      int source_word_best = source_sentence_words[a_best];
      int source_word_best_average = source_sentence_words[a_best_average];
      int source_word_best_overall = source_sentence_words[a_best_overall];
      for(size_t i = 0; i < source_sentence_words.size();++i){
        int source_word = source_sentence_words[i];
        if(t_var_[source_word][t_var_lookup_[target_word][source_word]]*d_var_[i][j] > t_var_[source_word_best][t_var_lookup_[target_word][source_word_best]]*d_var_[a_best][j]){
          a_best = i;
          source_word_best = source_sentence_words[a_best];
	}
	if(t_var_average_[source_word][t_var_lookup_[target_word][source_word]]*d_var_average_[i][j] > t_var_average_[source_word_best_average][t_var_lookup_[target_word][source_word_best_average]]*d_var_average_[a_best_average][j]){
          a_best_average = i;
          source_word_best_average = source_sentence_words[a_best_average];
	}
	if(t_var_best_overall_[source_word][t_var_lookup_[target_word][source_word]]*d_var_best_overall_[i][j] > t_var_best_overall_[source_word_best_overall][t_var_lookup_[target_word][source_word_best_overall]]*d_var_best_overall_[a_best_overall][j]){
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
  for(unordered_map<int, unordered_set<int> >::iterator source_dictionary_iterator = source_dictionary_.begin(); source_dictionary_iterator != source_dictionary_.end(); ++source_dictionary_iterator){
    int source_word(source_dictionary_iterator->first);
    unordered_set<int> target_words;
    for(unordered_set<int>::iterator target_word_iterator = target_words.begin(); target_word_iterator != target_words.end(); ++target_word_iterator){
      int target_word(*target_word_iterator);
      double probability(t_var_average_[source_word][t_var_lookup_[target_word][source_word]]);
      cout <<"t_var_average_[" <<  target_word << "|" << source_word << "] = " << probability << endl;
    }
  }

  for(size_t j = 0; j < max_target_sentence_length_; ++j){
    for(size_t i = 0; i < max_source_sentence_length_; ++i){
      cout <<"d_var_average_[" << i << "|" << j << "] = " << d_var_average_[i][j] << endl;
    }
  }
}


