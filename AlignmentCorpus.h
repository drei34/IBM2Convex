#ifndef ALIGNMENTCORPUS_H
#define ALIGNMENTCORPUS_H

#include <iostream>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <utility>
#include <cmath>
using namespace std;
using namespace boost;

class AlignmentCorpus
{
public:
  AlignmentCorpus(){
    cout << "Default!" << endl;
  } // This should never be used.
  AlignmentCorpus(string file_name);
  // This is a list of all unique source and target language words in the corpus.
  unordered_set < int > source_words_;
  unordered_set < int > target_words_;
  // This is a unordered_map so that we have for each source or target word a unordered_set of possibly aligned source or targets words.
  unordered_map < int, unordered_set < int > > source_dictionary_;
  unordered_map < int, unordered_set < int > > target_dictionary_;
  // This is a vector which holds the source and foreign sentences, but tokeized.
  vector < vector <int> > source_sentences_tokenized_;
  vector < vector <int> > target_sentences_tokenized_;
  // These are the int tokens which each string has. The index in the vector corresponds to the token used.
  unordered_map<string, int> source_word_int_tokens_;
  unordered_map<string, int> target_word_int_tokens_;
  int number_source_words_;
  int number_target_words_;
  string translation_type_;
  void get_corpus(string source_file_name, string target_file_name);
};

#endif

