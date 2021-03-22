## lce_tension_tests

This repo contains scripts to process full-field measurements from mechanical tensile tests with liquid crystal elastomers.
Images are correlated with the GOM Correlate software with coordinates and kinematic quantities output in .csv format.

Below is a brief description of each main script (denoted by prefix 'm_'):

i) m_process_to_pkl:
  This script loads in all .csv output files from GOM and generates a dataframe. 
  
  The cross-sectional area at each x coordinate is also calculated by loading
  in a separate csv file with x and y coordinates describing a polygon containing
  the specimen within the reference image. Since strains and displacements from
  GOM are computed in the reference coordinates (by coordinates output in deformed
  state for plotting), the reference geometry can be used to compute engineering
  stress at each section. 
  
  The coordinates of all points on the specimen can easily be generated in imageJ
  by tracing the specimen with the polygon tool and exporting coordinates in the
  Measure -> Tools menu. This file is then loaded into m_process_to_pkl as a data 
  frame, grouped by x coordinate and the max and min y coordinates at each unique x
  is used to compute the cross-section width. This is converted to mm using the 
  image scale and concatenated onto the master dataframe.
  
  The force at each image is also imported from the mech load frame csv file and used
  to compute the average x-stress at each cross-section. This is also concatenated 
  onto the master dataframe. 
  
  All data is then exported to pkl with the filename format: results_df_frame_x.pkl
  
ii) m_eda_multiple_frames:
  This script is used to perform exploratory data analysis (EDA) on the compiled
  data frames generated in m_process_to_pkl. The plots generated are as follows:
  
  a) histogram of average Exx strain at each frame
  b) boxplots of all strain components for each frame
  c) Exx strain vs frame for different strain bands (customizable)
  d) local Exx strain vs cross-section averaged stress for different strain bands
  e) the coordinates of each strain band masked onto all coordinates on the sample
  f) width-average Exx strain vs width-averaged stress
  
  * Notes: the script was originally written to load in all frames into one master
  * dataframe that allowed for easy filtering/plotting/descriptive statistical 
  * analysis. This approach works for thin dogbone specimens but struggles with memory
  * issues when the specimen aspect ratio is low. Therefore, a flag was added 
  * (load_multiple_frames (Bool)) to allow the user to specify whether each frame
  * should be loaded and manipulated separately.

The files with prefix ('f_') are function files and were generated for debugging/
proof-of-concept development, but were later integrated into the above main
scripts as functions.
  
