# NHTS2022から活動生成


## preparation
NHTS2022をダウンロードする

1. NHTSのWebサイトへアクセス  
[NHTS](https://nhts.ornl.gov/)

2. サイトからCSVファイルをダウンロード  
ダウンロードが完了したファイルの構成は以下のようになっている  
```bash
csv
|_Catation.pdf
|_hhv2pub.csv
|_ldtv2pub.csv
|_perv2pub.csv
|_tripv2pub.csv
|_vehv2pub.csv
```


## pre-processing
pathを指定して以下の2つのファイルを順番に実行する
1. ``src/preprocess/build_schedules.py``
2. ``src/preprocess/build_conditions.py``
