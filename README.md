# Coursework 1 

## Assignment: Implement and test a beat tracking system for ballroom dance music.
You may take a published paper and attempt to reimplement it, such as one of the approaches mentioned in lectures, or you may develop your own ideas, inspired and informed by the research literature.

## Usage 
`beats, downbeats = beatTracker(inputFile, plot_predictions=False)`

`inputFile = '/path/to/audio/file'`

`beats` is a vector of beat times in seconds. The beat times should correspond to the tactus or primary metrical level. 

`downbeats` is a vector of downbeat times in seconds. The downbeat times should correspond to the first beat of each bar. 

## Resources and References
### The Ballroom dataset:
http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html
Note that the link to the dataset is broken. It should be:
http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz

### Ground truth annotations of beats and downbeats 
Florian Krebs: https://github.com/CPJKU/BallroomAnnotations

### Evaluation Libraries:
https://craffel.github.io/mir_eval/

## Submission Materials
1. Submit a ZIP file containing the python code for your solution (documented sufficiently so that it is easy for the marker to run the code).
2. A PDF file (maximum 5 pages) containing a report describing and evaluating your beat tracking system.

# Citations

## References

### Beat Tracking & Neural Networks

* MatthewDavies, E. P., and Sebastian Böck. **"Temporal convolutional networks for musical audio beat tracking."** 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.
    * *Note:* Primary Implementation and usage of TCN's from this paper. 


* Böck, Sebastian, Florian Krebs, and Gerhard Widmer. **"A Multi-model Approach to Beat Tracking Considering Heterogeneous Music Styles."** ISMIR. 2014. 


* Böck, Sebastian, and Markus Schedl. **"Enhanced beat tracking with context-aware neural networks."** Proc. Int. Conf. Digital Audio Effects. 2011.


* Böck, Sebastian, Florian Krebs, and Gerhard Widmer. **"Joint Beat and Downbeat Tracking with Recurrent Neural Networks."** ISMIR. 2016.


* Hochreiter, Sepp, and Jürgen Schmidhuber. **"Long short-term memory."** Neural computation 9.8 (1997): 1735-1780.


### State-Space & Probabilistic Modeling

* Krebs, Florian, Sebastian Böck, and Gerhard Widmer. **"An Efficient State-Space Model for Joint Tempo and Meter Tracking."** ISMIR. 2015. 


* Krebs, Florian, Sebastian Böck, and Gerhard Widmer. **"Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio."** Ismir. 2013.


* Whiteley, Nick, Ali Taylan Cemgil, and Simon J. Godsill. **"Bayesian Modelling of Temporal Structure in Musical Audio."** ISMIR. 2006.


* *Rabiner, Lawrence R. **"A tutorial on hidden Markov models and selected applications in speech recognition."** Proceedings of the IEEE 77.2 (2002): 257-286.


### Datasets & Benchmarks

* Gouyon, Fabien, et al. **"An experimental comparison of audio tempo induction algorithms."** IEEE Transactions on Audio, Speech, and Language Processing 14.5 (2006): 1832-1844.


* Hainsworth, Stephen W., and Malcolm D. Macleod. **"Particle filtering applied to musical tempo tracking."** EURASIP Journal on Advances in Signal Processing 2004.15 (2004): 927847.
