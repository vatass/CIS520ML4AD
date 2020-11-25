import numpy as np
import pandas as pd
import pickle 
import sys 
import seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

with open('../ADNI.pkl', 'rb') as f:
    d = pickle.load(f)

unique_patients = d['participant_id'].unique() 

# grouped_df  = d.groupby('participant_id' ,as_index=False)
# print('Number of groups', grouped_df.ngroups)

patients = 0 
progression_patients = 0 
all_cn = 0 
all_dem = 0 
all_mci = 0 

trajectories = {'trajectory' : []}
abnormal_cases = {'trajectory' : [], 'time_interval':[]}

patient_to_remove = [] 

for patient in unique_patients: 
    patients +=1 
    print('Patient', patient)

    group = d.loc[d['participant_id'] == patient]

    dates = group['Date'].tolist() 

    diagnosis = group['Diagnosis'].tolist() 

    assert len(dates) == len(diagnosis)

    print('Initial Diagnosis', len(diagnosis))

    cleardiagnosis, cleardates = [], [] 
    for i,di in enumerate(diagnosis): 
        if str(di) != 'nan':
            cleardiagnosis.append(di)
            cleardates.append(dates[i])
        else:
            print('index of nan', i)
 
    assert len(cleardates) == len(cleardiagnosis)

    if len(cleardiagnosis) == 0 : 
        # print('No Diagnosis')
        patient_to_remove.append(patient)
        continue
    elif len(cleardiagnosis) ==1 : 
        # print('Single Visit')
        patient_to_remove.append(patient)
        continue
    
    assert cleardates[-1] > cleardates[0]

    ### Find the special cases #### 

    if cleardiagnosis[0] == 'MCI'  and  cleardiagnosis[-1] == 'CN' :
        print('A B N O R M A L C A S E !!  11111111111')
        print(cleardates[0], cleardates[-1])
        print('Interval', np.abs((cleardates[-1].year - cleardates[0].year)))
        abnormal_cases['trajectory'].append('MCI->CN')
        abnormal_cases['time_interval'].append(np.abs(cleardates[-1].year - cleardates[0].year))
        
    if cleardiagnosis[0] == 'Dementia'  and  cleardiagnosis[-1] == 'MCI' :
        print('A B N O R M A L C A S E !! 222222222222')
        print(cleardates[0], cleardates[-1])
        print('Interval', np.abs(cleardates[-1].year - cleardates[0].year))
        abnormal_cases['trajectory'].append('Dementia->MCI')
        abnormal_cases['time_interval'].append(np.abs(cleardates[-1].year - cleardates[0].year))

    trajectory = cleardiagnosis[0] + '->' + cleardiagnosis[-1]

    if cleardiagnosis[0] == cleardiagnosis[-1]: 
        # print('Stable')
        if cleardiagnosis[0] == 'CN' : 
            all_cn +=1
        elif cleardiagnosis[0] == 'Dementia': 
            all_dem +=1 
        elif cleardiagnosis[0] == 'MCI':
            all_mci +=1 
    else: 
        progression_patients+= 1


    trajectories['trajectory'].append(trajectory)

assert patients == len(unique_patients)

print('Total Patients', patients)
print('Patients with progression', progression_patients)
print('Patients to remove', len (patient_to_remove))
print('All CN', all_cn)
print('All Dem', all_dem)
print('All MCI', all_mci)

ax = sns.displot(x="trajectory", data=trajectories)
plt.title('Disease Trajectories in ADNI')
plt.xticks(rotation=45)
plt.show()
plt.savefig('./plots/disease_trajectories_in_adni.png')


ax = sns.barplot(x="trajectory", y='time_interval', data=abnormal_cases)
plt.title('Abnormal Disease Trajectories in ADNI')
plt.show()
plt.savefig('./plots/abnormal_trajectories_time_interval.png')
