# Test-Time Augmentation on CIFAR10

Test-Time Augmentation(TTA)とは、テスト時にもAugmentationを施し、複製したデータの分だけモデルの出力を複数得て、その結果を統合して最終的な予測とする方法です。[1]  
ここでは、TTA時の画像加工(Augmentation)としてHorizontal Flipのみを使用しており、推論時はオリジナルと画像加工分あわせて２種類の出力が得られ、以下の方法でそれらを統合しています。
- Argmax(Mean(Softmax(Logits1), Softmax(Logits2)))

5回分異なるシード値で実行した結果が以下のグラフとなります。  
- `no tricks`: Augmentation、TTA両方不採用
- `aug`: Augmentationのみ採用
- `tta`: TTAのみ採用
- `aug & tta`: Augmentation、TTA両方採用

Test Accuracyは100epoch目の各モデルで推論を行って計算しています。
<!-- <img src=report/tta_experiment.png width=px height=px > -->
<img src=report/summary.png width=px height=px >


### Dependencies
- Ubuntu 20.04.1 LTS
- CUDA 11.1
- cuDNN 8.1.1.33-1
- Driver 460.91.03 (RTX 3080)
- Docker version 20.10. 5
- Python 3.9.2 (pyenv)
- TensorFlow 2.7.0


## Installation
Clone this repo and create python env by:
```bash
$ git clone ***
$ cd tta_cifar10/
$ pip install -r requirements.txt
```

If you don't want to mess with your local machine, you can run this project on a docker container by running:
```
$ cd docker/
$ make init
$ make bash
# cd tta_cifar10/
```


## Experiments

`run.sh` script automatically executes `$ python train.py` for different 4 settings loading config files from `config` folder with a fixed seed.
You can set a `SEED` in the script in advance and run multiple times to reproduce results (top figure's scores are calculated over 5 runs with different seeds (42 to 46)).

```bash
$ ./run.sh
```

```run.sh
# run.sh
CONFIG=('01_no_aug_no_tta.yaml' '02_aug_no_tta.yaml' '03_no_aug_tta.yaml' '04_aug_tta.yaml')
SEED='42' # 42, 43, 44, 45, 46
RUN_ID=$(python utils/generate_runid.py)

echo '-------------------------'
echo "RUN_ID: $RUN_ID"
echo '-------------------------'

export RUN_ID

for conf in ${CONFIG[@]}; do
    echo
    command="python train.py --config $conf --seed $SEED"
    echo "$ $command"
    eval $command
done

cd utils
command="python plot_reports.py --filename tta_results_seed-${SEED}_runid-${RUN_ID}.png"
echo
echo $command
eval $command
cd ..
```

To visualize the top figure, you can use `utils/plot_summary.py` where
you need to set `ID_list` to contain IDs that the `run.sh` prints at the very beginning:
```plot_summary.py
# These logs are provided in report folder
ID_list = ['20220116-073253',
           '20220116-073303',
           '20220116-121400',
           '20220116-121408',
           '20220116-123421']
```
```bash
cd utils
python plot_summary.py
```

---

Alternatively, you can train each model by running python script directly:
```bash
# No data augmentation and tta
$ python train.py --config 01_no_aug_no_tta.yaml --seed 42

# Data augmentation but no tta
$ python train.py --config 02_aug_no_tta.yaml --seed 42

# No data augmentation but tta
$ python train.py --config 03_no_aug_tta.yaml --seed 42

# Data augmentation and tta
$ python train.py --config 04_aug_tta.yaml --seed 42
```

Then, to visualize the result, you need to edit `RUN_ID` in `utils/plot_report.py` so that it can find right logs:
```python
RUN_ID = {
    # Change these IDs, which are a part of the log file names
    'no_aug_no_tta': '01_no_aug_no_tta_20220110-182032',
    'aug_no_tta': '02_aug_no_tta_20220110-182520',
    'no_aug_tta': '03_no_aug_tta_20220110-182945',
    'aug_tta': '04_aug_tta_20220110-183355',
}
```
Then, run below:
```bash
$ cd helper/
$ python plot_report.py
```
---

## Reference
1. [cifar10 で Test Time Augmentation (TTA) の実験 - Qiita](https://qiita.com/cfiken/items/7cbf63357c7374f43372)

2. [Comparison of TTA Prediction Procedures | Kaggle](https://www.kaggle.com/calebeverett/comparison-of-tta-prediction-procedures)