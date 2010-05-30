Sextant
=======

Sextant is a very simple vector space search engine written in Python. Its primary purpose is to quickly determine similarity between "bags of words" which can either be documents or tag collections. It's a relatively straightforward implementation of a traditional vector space model for similarity scoring. Even though it uses sparse vectors for term collections it's currently not very memory efficient (~40K documents averaging ~900 terms each takes about 200MB on my server). A vector similarity computation of that collection takes about one second on a virtual 1GHz machine. 

It can be used with both normalized raw term frequency weighting or term frequency / inverse document frequency weighting.

Requirements
------------

* `numpy <http://numpy.scipy.org/>`_
* `PyStemmer 1.1.0 (included) <http://snowball.tartarus.org/>`_

Future
------

* Use a leader / follower vector search to improve memory utilization and comparison speed
* Add memory paging of vector collections to reduce the requirement for the whole collection to be in memory
* Allow multiple document collections
* Improve memory usage
* Add additional weighting modes

License
=======

Sextant is distributed under the `MIT license <http://www.opensource.org/licenses/mit-license.php>`_. See LICENSE.
PyStemmer 0.1.1 and snowball carry their own licenses which can be read under the PyStemmer-1.1.0/ directory.