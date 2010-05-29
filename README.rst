Sextant
=======

Sextant is a very simple vector space search engine written in Python. Its primary purpose is to quickly determine similarity between "bags of words" which can either be 
documents or tag collectons. It's a relatively straightforward implementation of a traditional vector space model for similarity scoring. It's currently not very memory 
efficient (~40K documents averaging ~900 terms each takes about 200MB on my server) and a vector similarity computation takes about 1.3s on a virtual 1GHz machine. It can 
be used with both raw term frequency weighting or term frequency / inverse document frequency weighting.

Requirements
------------

  * numpy
  * PyStemmer 0.1.1 (included)

Future
------

  * Use a leader / follower vector search to improve memory and comparison speed

License
=======

Sextant is distributed under the `MIT license <http://www.opensource.org/licenses/mit-license.php>`_. See LICENSE.
