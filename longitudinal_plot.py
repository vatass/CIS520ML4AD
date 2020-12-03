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

longitudinal_dataset = [] 

patients = 0 
progression_patients = 0 
all_cn = 0 
all_dem = 0 
all_mci = 0 

trajectories = {'trajectory' : []}
abnormal_cases = {'trajectory' : [], 'time_interval':[]} 
abnormal_cases_in_detail = {'timepoint' : [], 'diagnosis' : [], 'patient' : [], 'change_timepoint' : [], 'change_timepoint_months':[], 'change_timepoint_years':[]  }
patient_to_remove = [] 

cnt = 0

abnormal = False

for patient in unique_patients: 
    patients +=1 
    print('Patient', patient)

    group = d.loc[d['participant_id'] == patient]

    dates = group['Date'].tolist() 

    diagnosis = group['Diagnosis'].tolist() 

    assert len(dates) == len(diagnosis)

    print('Initial Diagnosis', len(diagnosis))

    cleared = False

    cleardiagnosis, cleardates = [], [] 
    for i,di in enumerate(diagnosis): 
        if str(di) != 'nan':
            cleardiagnosis.append(di)
            cleardates.append(dates[i])
        else:
            cleared = True
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
        continue
        print('A B N O R M A L C A S E !!  11111111111')
        print(cleardates[0], cleardates[-1])
        print('Interval', np.abs((cleardates[-1].year - cleardates[0].year)))
        abnormal_cases['trajectory'].append('MCI->CN')
        abnormal_cases['time_interval'].append(np.abs(cleardates[-1].year - cleardates[0].year))
        abnormal = True 

        if not cleared : 
            if cnt >= 5 : 
                pass
            else : 
                cnt+=1
                for visit in range(len(cleardiagnosis)): 
                   
                    abnormal_cases_in_detail['timepoint'].append(visit)
                    abnormal_cases_in_detail['diagnosis'].append(cleardiagnosis[visit])
                    abnormal_cases_in_detail['patient'].append(patient)

                    if visit >=1: 
                        if visit != cleardiagnosis[visit-1]:
                            # pivotal visit found 
                            abnormal_cases_in_detail['change_timepoint'].append(visit)
                            pivotal_time = (cleardates[visit].year - cleardates[visit-1].year)*12 +  cleardates[visit].month - cleardates[visit-1].month
                            abnormal_cases_in_detail['change_timepoint_months'].append(pivotal_time)
                            
                            pivotal_time_years = cleardates[visit].year - cleardates[visit-1].year
                            abnormal_cases_in_detail['change_timepoint_years'].append(pivotal_time_years)





    if cleardiagnosis[0] == 'Dementia'  and  cleardiagnosis[-1] == 'MCI' :
        continue
        print('A B N O R M A L C A S E !! 222222222222')
        print(cleardates[0], cleardates[-1])
        print('Interval', np.abs(cleardates[-1].year - cleardates[0].year))
        abnormal_cases['trajectory'].append('Dementia->MCI')
        abnormal_cases['time_interval'].append(np.abs(cleardates[-1].year - cleardates[0].year))
        abnormal = True 
        if not cleared : 
            if cnt >= 5 : 
                pass
            else : 
                cnt+=1
                for visit in range(len(cleardiagnosis)): 
                   
                    abnormal_cases_in_detail['timepoint'].append(visit)
                    abnormal_cases_in_detail['diagnosis'].append(cleardiagnosis[visit])
                    abnormal_cases_in_detail['patient'].append(patient)

                    if visit >=1: 
                        if visit != cleardiagnosis[visit-1]:
                            # pivotal visit found 
                            abnormal_cases_in_detail['change_timepoint'].append(visit)
                            pivotal_time = (cleardates[visit].year - cleardates[visit-1].year)*12 +  cleardates[visit].month - cleardates[visit-1].month
                            abnormal_cases_in_detail['change_timepoint_months'].append(pivotal_time)

                            pivotal_time_years = cleardates[visit].year - cleardates[visit-1].year
                            abnormal_cases_in_detail['change_timepoint_years'].append(pivotal_time_years)


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


    ### Trajetory is the label that should be assigned in every group ###
    # And then we need to filter only the neuroimaging features for the longitudinal set up 

    # Extract only the H_MUSE_* features 

    # keys = list(group.keys())
    # assert

    features = group.filter(regex="H_MUSE_Volume_*").to_numpy() 
    print('features', features.shape)
    
    target = trajectory
    print('Target', target)
    longitudinal_dataset.append((features, target))




assert patients == len(unique_patients)
assert len(longitudinal_dataset) < len(unique_patients)

# Store the longitudinal dataset 
datadict = {'dataset' : longitudinal_dataset} 

print(len(longitudinal_dataset)) 

print(type(datadict['dataset']))
with open('longitudinal_dataset.pkl', 'wb') as handle:
    pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)



print('Total Patients', patients)
print('Patients with progression', progression_patients)
print('Patients to remove', len (patient_to_remove))
print('All CN', all_cn)
print('All Dem', all_dem)
print('All MCI', all_mci)

sys.exit(0)

ax = sns.displot(x="trajectory", data=trajectories)
plt.title('Disease Trajectories in ADNI')
plt.xticks(rotation=45)
plt.savefig('../plots/disease_trajectories_in_adni.png')
plt.show()

# ax = sns.barplot(x="trajectory", y='time_interval', data=abnormal_cases)
# plt.title('Abnormal Disease Trajectories in ADNI')
# plt.ylabel('Time interval in months')
# plt.savefig('../plots/abnormal_trajectories_time_interval.png')
# plt.show()

ax = sns.lineplot(x="timepoint", y='diagnosis', hue='patient',  data=abnormal_cases_in_detail)
plt.title('Abnormal Disease Trajectories in ADNI')
plt.ylabel('Diagnosis')
plt.xlabel('Timepoint')
plt.savefig('../plots/abnormal_trajectories_in_detail.png')
plt.show()

ax = sns.displot(x="change_timepoint_months", data=abnormal_cases_in_detail)
plt.title('Histogram of pivotal timepoints')
plt.savefig('../plots/abnormal_case_pivotal_diagnosis_timepoint_in_months.png')
plt.show()

ax = sns.displot(x="change_timepoint_years", data=abnormal_cases_in_detail)
plt.title('Histogram of pivotal timepoints in years')
plt.savefig('../plots/abnormal_case_pivotal_diagnosis_timepoint_in_years.png')
plt.show()




