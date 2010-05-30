#!/usr/bin/env python
# encoding: utf-8
"""
vec_test.py

Created by mikepk on 2010-05-18.
Copyright (c) 2010 Michael Kowalchik. All rights reserved.
"""

import sys
import getopt

import math

import re

from numpy.random import rand, randint
from numpy import int8,int16,uint16,uint32,float32

from numpy import dot, resize
from numpy.linalg import norm
from numpy import array, zeros, ones, empty

from stopwords import stopwords

# using the snowball stemmer
# PyStemmer 1.1
import Stemmer

help_message = '''
A test script for experimenting with the term vector space engine
'''
# Split document on anything that's not alphanumeric apostrophe or 
# underscore
tokens=re.compile (r"[^a-zA-Z0-9'_]+" )

class Doc(object):
    '''Tag/Document object.'''
    def __init__(self,text='',doc_id=None,name=''): #indicies=None,values=None):
        self.id = doc_id
        self.name = name
        self.indicies = None
        self.term_freq = None
        if text:
            self.parse(text)
        self.values = None
        self.tfidf = None
        self.col = None

    def expand(self):
        '''Return expanded vector.'''
        A = zeros(mv.size, dtype=float32)
        A[self.indicies] = self.values
        return A

    def compute_binary(self):
        '''Use singular presence/absence weighting.'''
        pass

    def compute_tfidf(self):
        '''Apply the weight scheme to tfidf'''
        self.tfidf = array( self.normalized_term_freq() * people.idf[self.indicies],dtype=float32)
        self.values = self.tfidf
        return 1
    
    def normalized_term_freq(self):
        '''Normalize term frequency to account for document length.'''
        return array(self.term_freq / float(sum(self.term_freq)), dtype=float32)
    
    def normalize_vector(self):
        '''Pre compute the eculidean norm of the vector.'''
        self.values = self.values / norm(self.values)


    def parse(self,text,expand=True):
        '''Tokenize, stem, and filter document.'''
        words = tokens.split(text.lower())

        terms = {}
        for word in words:
            # drop empty strings, single letters/numbers and anything in the stopwords
            # list
            if len(word) <= 1 or word in stopwords:
                continue

            # stemming does a morphological reduction on terms
            # i.e. jumping and jumped collapse to jump
            word = stemmer.stemWord(word)
            word.strip("\'")

            # collect terms and term counts
            index = mv.index(word,expand)
            if index >= 0:
                try:
                    terms[index]+=1
                except KeyError:
                    terms[index] = 1

        term_count = len(terms)
        self.indicies = empty(term_count,uint16)
        self.term_freq = empty(term_count,uint16)

        i = 0
        for index in terms:
            self.indicies[i] = index
            self.term_freq[i] = terms[index]
            i += 1

        return 1


    def __repr__(self):
        return str("<Doc %s, %s>" % (self.id,self.name))

class Query(Doc):
    '''Query Object'''
    # TODO: change weighting schemes based on some
    # global setting
    def __init__(self,text):
        Doc.__init__(self)
        self.parse(text,False)
        self.values = self.normalized_term_freq()
        self.normalize_vector()
        

class DocumentCollection(object):
    def __init__(self,name):
        self.name = name
        self.term_df = zeros(0,dtype=uint32)
        self.idf = None
        self.total_docs = 0
        self.documents = []

    def __getitem__(self,index):
        '''Allow direct indexing of documents.'''
        return self.documents[index]

    def add_text(self,name=None,doc_id=None,text=None):
        '''Add a term set to the document collection.'''
        doc = Doc(name,doc_id,text)
        self.documents.append(doc)
        self.add_count(doc)

    def add_doc(self,doc):
        '''Add an existing document to the collection.'''
        self.documents.append(doc)
        self.add_count(doc)

    def add_count(self,doc):
        '''Increment the document term frequency.'''

        # expand to master vector size, then set 1, for all terms that
        # exist
        term_doc_count = zeros(mv.size, dtype=uint16)
        term_doc_count[doc.indicies] = 1

        self.total_docs += 1
        #if self.term_df.size != vec.size:
        self.term_df.resize(term_doc_count.size)
        self.term_df += term_doc_count

    def compute_idf(self):
        '''Recompute the inverse document frequency.'''
        # IDF is the log of the total documents / the number of documents that the term appears. This
        # term weight makes uncommon terms more significant than common ones.
        self.idf = array([math.log(self.total_docs / i, 10 ) for i in self.term_df],dtype=float32)

    # def query(self,text):
    def query(self,text):
        results = []
        # create a vector but don't add to the master vector
        q_vec = Query(text).expand()
        # compare the query to all docs
        for vec in self.documents:
            results.append((vec.id, vec.name,
            sim( 
                q_vec,
                vec.expand()
                ) 
            ))

        return sorted(results, key=lambda result: result[2], reverse=True)[:20]


    def __repr__(self):
        return str(self.documents)

class MasterVector(object):
    '''Object to hold the master vector, all terms seen by the engine.'''
    def __init__(self,loaded_master=None):
        self.master = []
        self.size = 0
    
    def __getitem__(self,index):
        '''Allow direct indexing of documents.'''
        return self.master[index]
    
    def index(self,word,expand=True):
        if word in self.master:
            return self.master.index(word)
        else:
            if not expand:
                return -1
            else:
                self.master.append(word)
                self.size=len(self.master)
                return len(self.master) - 1

    def get_term(self,pos):
        try:
            return self.master[pos]
        except IndexError:
            return -1
            
    def __repr__(self):
        return str("<MasterVector size:%d>" % (self.size))


class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg

stemmer = Stemmer.Stemmer('english')
people = DocumentCollection("People")

#def main(argv=None):
mv = MasterVector()

# create a series of documents for testing
docs = []
for i in range(1,8):
    doc_name = "doc-%.3d.txt" % i
    print doc_name
    doc_file = open(doc_name,"r")
    doc = doc_file.read()
    # docs.append(doc)
    doc_file.close()
    docs.append((doc_name,doc))


# test code multiplies documents by some number for a larger population
import copy

tv = []
ix = 0
for doc in docs:
    #term_vectors = make_bag_of_words(doc[1])
    vector_doc = Doc(doc[1],0,doc[0])
    vector_doc.values = vector_doc.normalized_term_freq()
    vector_doc.normalize_vector()

    for z in range(5000):
        ix+=1
        doc_cp = copy.deepcopy(vector_doc)
        doc_cp.id = ix
        
        people.add_doc(doc_cp)
        
        # tv.append(doc_cp)
        # 
        # term_doc_count = zeros(mv.size, dtype=uint16)
        # term_doc_count[doc_cp.indicies] = 1
        # 
        # people.add(term_doc_count)
        
        
print "Total Documents %d" % ix

print "Computing Document idf..."
people.compute_idf()
print "done"


def compare_docs(doc_id):
    q_vec = tv[doc_id].expand()
    results = []
    for vec in tv:
        results.append((vec.id, vec.name,
        sim( 
            q_vec,
            vec.expand()
            ) 
        ))

    return sorted(results, key=lambda result: result[2], reverse=True)[:20]
    

def query(text):
    results = []
    # create a vector but don't add to the master vector
    q_vec = Query(text).expand() # expand_norm_vector( make_bag_of_words(text,False) )
    # compare the query to all docs
    for vec in tv:
        results.append((vec.id, vec.name,
        sim( 
            q_vec,
            vec.expand()
            ) 
        ))

    return sorted(results, key=lambda result: result[2], reverse=True)[:20]


def ex_sim(vec1,vec2):
    return sim(vec1.expand(),vec2.expand())

def sim(vec1,vec2):
    '''Perform cosine similarity computation'''
    # compute the cosine similarity. Dot product of normalized vecotrs.
    return float( dot(vec1,vec2) )

def show(list):
    for i in list:
        print "%4d %10s -- %5.4f" % (i[0],i[1],i[2])

import timeit
def timed_query(query):
    t = timeit.Timer('''query("%s")''' % query,'''from __main__ import query''')
    print '''query("%s") %.3f seconds''' % (query,t.timeit(number=1))

def timed_compare(q_id):
    t = timeit.Timer('''compare_docs(%s)''' % str(q_id),'''from __main__ import compare_docs''')
    print '''compare_docs(%s) %.3f seconds''' % (str(q_id),t.timeit(number=1))
    
def time(vec1,vec2,num=1):
    t = timeit.Timer('''sim(tv[0],tv[1])''','''from __main__ import query, sim, tv''')
    time = t.timeit(number=num)
    per = time/num
    print '''Ran %s times total:%.3f seconds, per iteration: %.5f ''' % (num,time,per)
    