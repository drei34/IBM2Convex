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
  unordered_set < string > source_words_;
  unordered_set < string > target_words_;
  // This is a unordered_map so that we have for each source or target word a unordered_set of possibly aligned source or targets words.
  unordered_map < string, unordered_set < string > > source_dictionary_;
  unordered_map < string, unordered_set < string > > target_dictionary_;
  // This is the unordered_set of all source and target sentences in the corpus.
  vector < string > source_sentences_;
  vector < string > target_sentences_;
  // This is a vector which holds the source and foreign sentences, but tokeized.
  vector < vector <string> > source_sentences_tokenized_;
  vector < vector <string> > target_sentences_tokenized_;
  string translation_type_;
  void get_corpus(string source_file_name, string target_file_name);
};

#endif

