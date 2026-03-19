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

* **Joint Beat and Downbeat Tracking with Recurrent Neural Networks** Bock, Sebastian; Krebs, Florian; Widmer, Gerhard *Abstract:* Presents a novel method for extracting beats/downbeats using RNNs on magnitude spectrograms combined with a Dynamic Bayesian Network. 


* **A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music Styles** Böck, Sebastian; Krebs, Florian; Widmer, Gerhard *Abstract:* Extends existing systems by using multiple style-specialized RNNs to estimate beat positions. 


* **Enhanced Beat Tracking With Context-Aware Neural Networks** *Böck, Sebastian (2011)* *Abstract:* Uses bidirectional LSTM RNNs for frame-by-frame beat classification followed by autocorrelation to determine tempo. 


* **Temporal Convolutional Networks for Musical Audio Beat Tracking** *Matthew Davies, E. P.; Bock, Sebastian (2019)* *DOI:* [10.23919/EUSIPCO.2019.8902578](https://doi.org/10.23919/EUSIPCO.2019.8902578) *Abstract:* Demonstrates the efficiency and performance of TCNs over recurrent approaches for music analysis. 

* **Long Short-Term Memory** *Hochreiter, Sepp; Schmidhuber, Jürgen (1997)* *DOI:* [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) *Note:* The foundational paper for LSTM architectures. 



### State-Space & Probabilistic Modeling

* **An Efficient State-Space Model for Joint Tempo and Meter Tracking** Krebs, Florian; Bock, Sebastian; Widmer, Gerhard *Abstract:* Proposes a new state-space discretisation to reduce computational complexity in Hidden Markov Models. 


* **Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio** Krebs, Florian; Boeck, Sebastian; Widmer, Gerhard *Abstract:* Introduces a HMM-based system that learns rhythmic patterns directly from data to reduce octave errors. 


* **Bayesian Modelling of Temporal Structure in Musical Audio** Whiteley, Nick; Cemgil, A. Taylan; Godsill, Simon *Abstract:* A probabilistic model of a "bar-pointer" mapping signals to latent periodic rhythmic patterns. 


* **A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition** Rabiner, L.R. (1989) *URL:* [IEEE Explore](http://ieeexplore.ieee.org/document/18626/) 



### Datasets & Benchmarks

* **An Experimental Comparison of Audio Tempo Induction Algorithms** Gouyon, F. et al. (2006) *Note:* Related to the Ballroom dataset benchmarks. 


* **Particle Filtering Applied to Musical Tempo Tracking** Macleod, Malcolm; Hainsworth, Stephen (2004) *Note:* Relates to the Hainsworth dataset performance. 