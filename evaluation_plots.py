import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle 

### LOAD SCORES FROM ALGORITHMS #### 
#TODO 

with open('scores_dataset1.pickle', 'rb') as handle:
    scores_1 = pickle.load(handle)
with open('sep_scores_dataset1.pickle', 'rb') as handle:
    scores1_ = pickle.load(handle)

with open('scores_dataset2.pickle', 'rb') as handle:
    scores_1 = pickle.load(handle)
with open('sep_scores_dataset2.pickle', 'rb') as handle:
    scores1_ = pickle.load(handle)

with open('scores_dataset3.pickle', 'rb') as handle:
    scores_1 = pickle.load(handle)
with open('sep_scores_dataset3.pickle', 'rb') as handle:
    scores1_ = pickle.load(handle)


# PLOT scores 
# 1 -- Grouped barplot 
plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="value", 
            hue="metric", 
            data=scores_1)
plt.xlabel("values")
plt.ylabel('algorithms') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/grouped_barplot_evaluation_metrics_dataset1.png')

## PLOT scores 
# 1 -- Grouped barplot 
plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="value", 
            hue="metric", 
            data=scores_2)
plt.xlabel("values")
plt.ylabel('algorithms') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/grouped_barplot_evaluation_metrics_dataset2.png')

## PLOT scores 
# 1 -- Grouped barplot 
plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="value", 
            hue="metric", 
            data=scores_3)
plt.xlabel("values")
plt.ylabel('algorithms') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/grouped_barplot_evaluation_metrics_dataset3.png')


# 2 : Grouped Plot: Performance of the two datasets side by side 

# merge scores1_ and scores_2 

total_scores = {}
for key,val in scores1_.items():
  total_scores[key] = [] 
  total_scores[key].extend(scores1_[key])
  total_scores[key].extend(scores2_[key])
  total_scores[key].extend(scores3_[key])

plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="sensitivity", 
            hue="dataset", 
            data=total_scores)
plt.xlabel("algorithm")
plt.ylabel('sensitivity') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/sensitivity_datasets_barplot.png')


plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="specificity", 
            hue="dataset", 
            data=total_scores)
plt.xlabel("algorithm")
plt.ylabel('specificity') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/specificity_datasets_barplot.png')

plt.figure(figsize=(10, 8))
sns.barplot(x="algorithm", 
            y="accuracy", 
            hue="dataset", 
            data=total_scores)
plt.xlabel("algorithm")
plt.ylabel('accuracy') 
plt.savefig('/content/drive/My Drive/CIS520_PROJECT/accuracy_datasets_barplot.png')
