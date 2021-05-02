# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:44:18 2021

@author: jcv
"""
mask_frame = 5
num_samples = 50000

# find points in strain band
cat_mask_df = all_frames_df[(all_frames_df['Eyy'] <0.2)& (all_frames_df['frame']==5)]

cat_df = all_frames_df[
        all_frames_df.index.isin(cat_mask_df.index)]

cat_sample_df = cat_df.sample(n = num_samples, random_state = 1)

fig, axs = plt.subplots(1,1, figsize=(5,5),
                        sharey=True, tight_layout=True)
axs.scatter(cat_sample_df['frame'], 
            cat_sample_df['Eyy'],
            s = 2, c = c[0], edgecolors = ec[0], 
            alpha = 0.4, linewidths = 0.5
            )

axs.set_ylabel('Eyy')
axs.set_xlabel('frame')
axs.grid(True, alpha = 0.5,zorder = 0)
        
_, max_ylim = plt.ylim()
axs.text(mask_frame*1.1, max_ylim*0.8,
                  'Mask frame: {:.0f}'.format(mask_frame),
                  fontsize = 4)

plt.show()

# create time dictionary

frame_list = np.array([0,5,10,15,20,25,30,35,40,45,50,
              100,150,200,250,300,350,400,450,500,
              750,1000,1250,1500,1750,2000,2250,2750,
              3000,3250,3500,3750, 4000,4250,4500,4750,5000])

ind_list = np.arange(0,len(frame_list),1)

dt = 0.2
time_list = frame_list*dt

frame_time_conv = {}

for i in range(0,len(frame_list)):
    frame_time_conv[i] = time_list[i]
    
    
def convert_frame_time(x):
    return frame_time_conv[x]

cat_sample_df['time'] = cat_sample_df['frame'].apply(convert_frame_time)