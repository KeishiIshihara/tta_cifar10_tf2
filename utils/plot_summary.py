#--------------------------------------------------
# plot_summary.py
#
# Plot the accuracy of the validation and test set
# with mean and standard deviation over 5 iterations
# of each test case.
#--------------------------------------------------


import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#########################################
ID_list = ['20220116-073253',
           '20220116-073303',
           '20220116-121400',
           '20220116-121408',
           '20220116-123421']
RUN_ID = {'no tricks': '01_no_aug_no_tta_{}',
          'aug': '02_aug_no_tta_{}',
          'tta': '03_no_aug_tta_{}',
          'aug & tta': '04_aug_tta_{}'}
report_csv_path = '../report/history_{}.csv'
meta_json_path = '../logs/meta_{}.json'
#########################################


sns.set()
sns.set_style('darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))


# axis 0
for k, v in RUN_ID.items():
    history_df_list = [pd.read_csv( report_csv_path.format(v.format(id)) ).set_index('epoch') for id in ID_list]
    df_groupby = pd.concat(history_df_list).groupby('epoch')
    mean_df = df_groupby.mean()
    std_df = df_groupby.std()
    axes[0].plot(mean_df.index.values, mean_df['val_acc'], label=k)
    axes[0].fill_between(mean_df.index.values,
                         y1=(mean_df['val_acc']+std_df['val_acc']),
                         y2=(mean_df['val_acc']-std_df['val_acc']),
                         alpha=0.3)
axes[0].legend()
axes[0].set_xlabel('epoch')
axes[0].set_title('Validation Accuracy')

# axis 1
test_acc_score_df_list = []
for k, v in RUN_ID.items():
    meta_dict_list = []
    for id in ID_list:
        with open(meta_json_path.format(v.format(id)), 'r') as json_file:
            meta_dict_list.append(json.load(json_file))
    test_acc_score_df = pd.DataFrame(index=ID_list,
                                     data={f'test_acc_{k}': [d['test_acc_score'] for d in meta_dict_list] })
    test_acc_score_df_list.append(test_acc_score_df)
test_acc_score_df = pd.concat(test_acc_score_df_list)
test_summary_df = test_acc_score_df.describe()
error_bar_set = dict(lw=1, capthick=1, capsize=10)
axes[1].bar(list(range(4)), test_summary_df.loc['mean'],
            yerr=test_summary_df.loc['std'],
            tick_label=list(RUN_ID.keys()),
            error_kw=error_bar_set,
            color='steelblue')
axes[1].set_ylim([0.6,0.9])
axes[1].set_title('Test Accuracy')


plt.tight_layout()
plt.savefig(f'../report/summary.png')
