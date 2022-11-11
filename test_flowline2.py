# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:56:17 2020

@author: giova
"""

# %% DEPENDENCIES

# remove variables
#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]

import pandas as pd 
import numpy as np
import json
# from time import sleep
#import networkx as nx
import matplotlib.pyplot as plt # For drawing
import pickle
#from tqdm import tqdm
#import os
#import pickle
import sys
#from copy import deepcopy
#from statsmodels.graphics.factorplots import interaction_plot
#import subprocess
import os
# from shutil import copyfile
# import strict_rfc3339 # per conversione event log
import seaborn as sns

sys.path.append(r'C:\Users\THE FACTORY PC 2\OneDrive - Politecnico di Milano\WIP')
sys.path.append('../')
os.chdir(r'C:\Users\THE FACTORY PC 2\OneDrive - Politecnico di Milano\WIP')

from MSM.msmlib import gen_model_init, checktype
#from MSM.modelscores import score_calc 
# from MSM.modelred import local_search
from MSM.msm_parameval import fit_kernel, gen_samples, find_flowtimes, calc_th_fromdata, calc_st_fromdata, gen_from_ecdf
from MSM.other import find_ordered_arcs
from MSM.msm_plots import plot_model

# %% INPUT FILES READ

# Load parameters
doe = pd.read_excel(r'C:\Users\THE FACTORY PC 2\OneDrive - Politecnico di Milano\WIP\MSM_experiments\MSM_TEST11_RTS-Demo\doe.xlsx', header='infer')

# Load config file
with open(r'C:\Users\THE FACTORY PC 2\OneDrive - Politecnico di Milano\WIP\MSM_experiments\MSM_TEST11_RTS-Demo\config.json', encoding='utf-8') as f:
    config = json.load(f)

# %% INPUTS

# for parameters read i need only 1 row
i = 0

# la replica coincide con il log che si chiama "rep#" nella cartella "logs"
#config["datapath"] = r"C:\Users\giova\OneDrive - Politecnico di Milano\WIP\MSM_experiments\FLOWLINE_NEW_Journal\logs\log"+str(doe['rep'][i])+".txt"

# ws vettore
config['modelred']['weights']['ws'] = [ doe['w1'][i], doe['w2'][i], doe['w3'][i], doe['w4'][i], doe['w5'][i],  doe['w6'][i]]

# y1a
config['modelred']['weights']['y1a'] = doe['y1a'][i]

# y1n
config['modelred']['weights']['y1n'] = doe['y1n'][i]  

#y2a	
config['modelred']['weights']['y2a'] = doe['y2a'][i] 

#y2n
config['modelred']['weights']['y2n'] = doe['y2n'][i]  

#y4in
config['modelred']['weights']['y4in'] = doe['y4in'][i]  

#y4out
config['modelred']['weights']['y4out'] = doe['y4out'][i]   
	
#y6a
config['modelred']['weights']['y6a'] =  doe['y6a'][i]   

#y6n
config['modelred']['weights']['y6n'] = doe['y6n'][i]    
	
#nn
config['modelred']['n_aggregate'] = doe['na'][i]   
config['modelred']['n_reducing'] = doe['nr'][i]   

# size
config['modelred']['desired_size'] = doe['size'][i]   

# contemp
config['batching']['threshold_nodes'] = doe['contemp'][i]   
config['batching']['threshold_arcs'] = doe['contemp'][i]   


# %% PRE PROCESSING

pallet_num = 12

ordata = pd.read_csv( config["datapath"] , sep=",", header='infer')
#ordata, mappings = checktype(ordata)

# ADD ts COLUMN = translation of RFC3339 into UNIX
# cols = ['time']
# for col in cols:
#     ordata['ts'] = ordata[col].apply(lambda x: strict_rfc3339.rfc3339_to_timestamp(x))

ordata['ts'] = ordata['time'].apply(lambda x: x)

ordata['tag'] = ordata['type'].apply(lambda x: x)

ordata['id_new'] = ordata['id'].apply(lambda x: x%pallet_num)   # calculate id of the pallet

ordata.loc[ ordata['id_new'] == 0 , 'id_new'] = pallet_num
    
# %% PRE PROCESSING 2

num_stations = max(ordata['activity'])
# TODO QUI VA TROVATO IN AUTOMATICO IL VETTORE DELLE COMBINAZIONI DI STAZIONI

for iid in ordata['id'].unique():

    ordata.loc[ ( ordata['id'] == iid) & ( ordata['activity'] == 1) & (ordata['tag'] == 's'), 'cum'+str(i) ] = 1
    ordata.loc[ ( ordata['id'] == iid) & ( ordata['activity'] == 2) & (ordata['tag'] == 'f'), 'cum'+str(i) ] = -1
       
cumsum = ordata.sort_values(by=['ts'])['cum0'].cumsum()
cumsum[~cumsum.isna()]
plt.plot(cumsum[~cumsum.isna()])
max(cumsum)           # NUMERO DI PALLET NEL SISTEMA


# %% MODEL MINING

data = ordata
#data['id'] = ordata['id_new']

results_path = r"C:\Users\THE FACTORY PC 2\OneDrive - Politecnico di Milano\WIP\MSM_experiments\MSM_TEST11_RTS-Demo\results"

#generate initial model
model, unique_list, tracetoremove, id_trace_record = gen_model_init(data, config, tag = True)
a = plot_model(model, results_path+"\model_figure")


#SAVE MODEL
with open(results_path+'\\model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
# SAVE MODEL AS JSON
with open(results_path+'\model.json', 'w', encoding='utf-8') as f:
    json.dump(model, f, ensure_ascii=False, indent=4)


        
# %% POST PROCESSING- PARAMETERS EVALUATION     
    
model_w_times = find_flowtimes(model, data, tag = True, aggregate = False)

from MSM.msm_parameval import calc_nodes_dist

model_w_times = calc_nodes_dist(model_w_times, data)

model_wo_times = model_w_times

# TEMPORARY REMOVAL OF FLOWTIMES FOR ISSUES IN SAVING JSON
for node in model_wo_times['nodes']:
    node['flowtimes'] = []

with open(results_path+'\model.json', 'w', encoding='utf-8') as f:
    json.dump(model_wo_times, f, ensure_ascii = False, indent=4)


# %% POST PROCESSING- PLOT PROCESSING TIMES 
    
model_w_times = find_flowtimes(model, data, tag = True, aggregate = False)

ft = next(c['flowtimes'] for c in model_w_times['nodes'] if c['cluster'] == 1.0 )

fig, ax = plt.subplots()
for a in [ft]:
    sns.distplot(a, bins=30, ax=ax, kde=True)
ax.set_xlim([min(ft), max(ft)])
ax.legend(["KDE", "DATA"])
plt.savefig('flowtimes1.png')