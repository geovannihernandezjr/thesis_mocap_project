# thesis_mocap_project
Texas State University - San Marcos MS of Engineering Thesis Project
Ingram School of Engineering Fall 2018 - Spring 2021
Committee Member:
  Damain Valles, Chair/Advisor
  Francis A. Mendez
  Jesus A. Jimenez
Link to Thesis: https://digital.library.txstate.edu/bitstream/handle/10877/13522/HERNANDEZ-THESIS-2021.pdf?sequence=1
ABSTRACT : Industrial Revolution 4.0 is defined as the interconnection of Information,
Communications Technologies (ICT) within the industry. In the occupation of laborers,
stock, and material mover they are often subjected to repetitive motions that cause
exhaustion (or fatigue) that could potentially lead to work-related musculoskeletal
disorder (WMSD). The most common repetitive motions are lifting, pulling, pushing,
carrying, and walking with load, which are also known as Manual Material Handling
(MMH) operations. There has been work using a machine learning technique known as
Recurrent Neural Network (RNN) to predict short and long-term motions from motion
capture measurements but research in using motion capture data related to MMH to
measure the fatigue needs exploration. For this research, only the lifting motion is
considered. Motion data is collected as time-stamped motion data using infrared cameras
at a rate of 100Hz of a subject performing repetitive lifting motion. The data is a
combination of XYZ coordinates from 39 reflective markers. Along with motion data, the
subject will self-report the perceived level of fatigue using the Borg scale every minute.
All this data can be merged into one to further be used for analysis. Since motions occur
over time for a duration of time, this data is used as input to a time-series deep learning
technique known as Long Short-Term Memory and Gated Recurrent Unit models. Using
these models, this research will evaluate the deep learning technique and motion capture
data to perform motion analysis to forecast univariate motion data and to also predict the
fatigue based on the displacement movement from each marker.

This github repo contains the code(s) that were created using time series univariate analysis of one single data marker as well as using multiple markers to predict fatigue from lifting motion conducted. The algorithms used were Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) with various parameters like number of neurons, epochs using Stochastic Gradient Descend optimizer. Based on research SGD performed well on motions therefore, this was the deciding factor to pick it. I compared both LSTM and GRU with LSTM performing far better than GRU. 
