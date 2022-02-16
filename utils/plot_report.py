import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
sns.set()
sns.set_style('whitegrid')

#########################################################
# RUN_ID = {
#     'no_aug_no_tta': '01_no_aug_no_tta_20220110-182032',
#     'aug_no_tta': '02_aug_no_tta_20220110-182520',
#     'no_aug_tta': '03_no_aug_tta_20220110-182945',
#     'aug_tta': '04_aug_tta_20220110-183355',
# }
# RUN_ID = {
#     'no_aug_no_tta': 'debug_01_no_aug_no_tta_20220115-132253',
#     'aug_no_tta': 'debug_02_aug_no_tta_20220115-132300',
#     'no_aug_tta': 'debug_03_no_aug_tta_20220115-132307',
#     'aug_tta': 'debug_04_aug_tta_20220115-132315',
# }
# RUN_ID = {
#     'no_aug_no_tta': '01_no_aug_no_tta_20220115-132707',
#     'aug_no_tta': '02_aug_no_tta_20220115-132917',
#     'no_aug_tta': '03_no_aug_tta_20220110-182945',
#     'aug_tta': '04_aug_tta_20220115-151757',
# }
# RUN_ID = {
#     'no_aug_no_tta': '01_no_aug_no_tta_20220115-152334',
#     'aug_no_tta': '02_aug_no_tta_20220115-152544',
#     'no_aug_tta': '03_no_aug_tta_20220115-152755',
#     'aug_tta': '04_aug_tta_20220115-153007',
# }
# RUN_ID = {
#     'no_aug_no_tta': '01_no_aug_no_tta_20220115-154543',
#     'aug_no_tta': '02_aug_no_tta_20220115-154754',
#     'no_aug_tta': '03_no_aug_tta_20220115-155005',
#     'aug_tta': '04_aug_tta_20220115-155218',
# }
# RUN_ID = {
#     'no_aug_no_tta': '01_no_aug_no_tta_20220116-113537',
#     'aug_no_tta': '02_aug_no_tta_20220116-113812',
#     'no_aug_tta': '03_no_aug_tta_20220116-114046',
#     'aug_tta': '04_aug_tta_20220116-114321',
# }
# RUN_ID = {
#     'no_aug_no_tta': 'debug_01_no_aug_no_tta_20220116-142709',
#     'aug_no_tta': 'debug_02_aug_no_tta_20220116-142709',
#     'no_aug_tta': 'debug_03_no_aug_tta_20220116-142709',
#     'aug_tta': 'debug_04_aug_tta_20220116-142709',
# }

report_csv_path = '../report/history_{}.csv'
meta_json_path = '../logs/meta_{}.json'
#########################################################


parser = argparse.ArgumentParser(description='')
parser.add_argument('-id', '--RUN_ID', type=str, default=None, help='RUN_ID')
parser.add_argument('-f', '--filename', type=str, default='tta_results.png', help='Output filename')
args = parser.parse_args()

if os.getenv('RUN_ID') is not None:
    ID = os.getenv('RUN_ID')
elif args.RUN_ID:
    ID = args.RUN_ID
else:
    raise ValueError('Please specify RUN_ID in order to find logs through env variable or argment.')

RUN_ID = {
    'no_aug_no_tta': f'01_no_aug_no_tta_{ID}',
    'aug_no_tta': f'02_aug_no_tta_{ID}',
    'no_aug_tta': f'03_no_aug_tta_{ID}',
    'aug_tta': f'04_aug_tta_{ID}',
}


# Prepare DataFrame for validation accuracy line plot
history_df_list = []
for k, v in RUN_ID.items():
    history_df_list.append(pd.read_csv(report_csv_path.format(v)).set_index('epoch'))
history_df = pd.concat(
    [history_df_list[i][['val_acc']].rename(columns={'val_acc': k}) for i, k in enumerate(RUN_ID.keys())],
    axis=1)


# Prepare DataFrame for test accuracy bar plot
meta_dict_list = []
for k, v in RUN_ID.items():
    with open(meta_json_path.format(v), 'r') as json_file:
        meta_dict_list.append(json.load(json_file))
test_acc_score_df = pd.DataFrame(index=list(RUN_ID.keys()), data={'test_acc': [meta_dict_list[i]['test_acc_score'] for i in range(len(RUN_ID))]})


# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # figsize=(width, height)
# axis 1
history_df.plot(ax=axes[0], kind='line', title='Validation Accuracy')
# axis 2
x = np.arange(test_acc_score_df.shape[0])
axes[1].bar(x, test_acc_score_df['test_acc'], 0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(test_acc_score_df.index.values, rotation=0)
axes[1].set_ylim([0.6, 0.9])
axes[1].set_title('Test Accuracy')
plt.tight_layout()
plt.savefig(f'../report/{args.filename}')
