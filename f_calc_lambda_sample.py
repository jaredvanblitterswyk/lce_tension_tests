# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:56:48 2021

@author: jcv
"""
# take subset of data for each frame
num_samples = 5000

all_frames_sample_df = pd.DataFrame(columns = all_frames_df.columns)

for i in range(1,50):
    frame_df = all_frames_df[all_frames_df['frame'] == i]
    frame_sample_df = frame_df.sample(n = 5000, random_state = 1 )
    frame_sample_df['lambda_y'] = frame_df['Eyy'].map(lambda x: np.sqrt(2*x + 1))
    frame_sample_df['lambda_x'] = frame_df['Exx'].map(lambda x: np.sqrt(2*x + 1))

    all_frames_sample_df = pd.concat([all_frames_sample_df, frame_sample_df], 
            axis = 0, join = 'outer'
            )
    
    
plt.figure()
plt.scatter(all_frames_sample_df['lambda_y'], all_frames_sample_df['lambda_x'], s = 2, alpha = 0.1)
