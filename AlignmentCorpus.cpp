#include "AlignmentCorpus.h"

AlignmentCorpus::AlignmentCorpus(string file_name): number_source_words_(0), number_target_words_(0)
{
  if(file_name.find("training_data_file_names_english_to_french",0) == 0){
    translation_type_ = "english_to_french";
  }
  else if(file_name.find("training_data_file_names_french_to_english",0) == 0){
    translation_type_ = "french_to_english";
  }
  else if(file_name.find("test_data_file_names_english_to_french.",0) == 0){
    translation_type_ = "english_to_french";
  }
  else if(file_name.find("test_data_file_names_french_to_english.",0) == 0){
    translation_type_ = "french_to_english";
  }
  else{
    cout << "A type of error happened! "<< endl;
    exit(1);
  }

  ifstream data_file;
  data_file.open(file_name.c_str());
  
  string data_line, source_file_name, target_file_name;
  string::size_type split_position;
  int i(0);
  while(true){
    getline(data_file,data_line);
    split_position = data_line.find(' ', 0);
    target_file_name = data_line.substr(split_position + 1);
    source_file_name = data_line.substr(0,split_position);

    if(source_file_name.size() == 0){
      break;
    }
    
    get_corpus(source_file_name, target_file_name);
  }
  data_file.close();
}

void AlignmentCorpus::get_corpus(string source_file_name, string target_file_name)
{
  ifstream source_file;
  source_file.open(source_file_name.c_str());
  ifstream target_file;
  target_file.open(target_file_name.c_str());

  string source_file_line, target_file_line;
  istringstream source_file_ss, target_file_ss; 
  
  while(true){
    getline(source_file,source_file_line);
    getline(target_file,target_file_line);

    // If this happens, we have no more data to read.
    if(source_file_line.size() == 0){
      break;
    }
    
    // Here we add the "null" word to the sentence. Notice that this is after the emptyness check is done.
    source_file_line.insert(0,"null ");

    istringstream source_file_ss(source_file_line);
    istringstream target_file_ss(target_file_line);
        
    vector <string> source_line_words_original, target_line_words_original;
    vector <int> source_line_words, target_line_words;
    string word;

    while (source_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      source_line_words_original.push_back(word); // Here we tokenize the sentence.
    }
    
    while (target_file_ss >> word){
      transform(word.begin(), word.end(), word.begin(), ::tolower);
      target_line_words_original.push_back(word);
    }

    for(size_t i = 0; i != source_line_words_original.size(); ++i){
      
      string source_word = source_line_words_original[i];
      
      if(source_word_int_tokens_.find(source_word) != source_word_int_tokens_.end()){
	source_line_words.push_back(source_word_int_tokens_[source_word]);
      }
      else{
	source_word_int_tokens_[source_word] = number_source_words_;
	source_line_words.push_back(source_word_int_tokens_[source_word]);
	++number_source_words_;
      }

    }

    for(size_t i = 0; i != target_line_words_original.size(); ++i){
      
      string target_word = target_line_words_original[i];
      
      if(target_word_int_tokens_.find(target_word) != target_word_int_tokens_.end()){
	target_line_words.push_back(target_word_int_tokens_[target_word]);
      }
      else{
	target_word_int_tokens_[target_word] = number_target_words_;
	target_line_words.push_back(target_word_int_tokens_[target_word]);
	++number_target_words_;
      }

    }

    source_sentences_tokenized_.push_back(source_line_words);
    target_sentences_tokenized_.push_back(target_line_words);

    for(vector<int>::iterator j = target_line_words.begin(); j != target_line_words.end(); ++j){
      for(vector<int>::iterator i = source_line_words.begin(); i != source_line_words.end(); ++i){
  
	int source_word = *i;
	int target_word = *j;
	
	if(source_dictionary_.find(source_word) == source_dictionary_.end()){
	  unordered_set <int> target_words;
	  target_words.insert(target_word);
	  source_dictionary_[source_word] = target_words;
	}
	else{
	  source_dictionary_[source_word].insert(target_word);
	}
	
	if(target_dictionary_.find(target_word) == target_dictionary_.end()){
	  unordered_set <int> source_words;
	  source_words.insert(source_word);
	  target_dictionary_[target_word] = source_words;
	}
	else{
	  target_dictionary_[target_word].insert(source_word);
	}

      }
    }
  }
  source_file.close();
  target_file.close();
}
