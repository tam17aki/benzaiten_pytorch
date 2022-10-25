# benzaiten_pytorch
## 概要
本リポジトリは「AIミュージックバトル！『弁財天』」から提供されたスターターキットのPyTorch版に相当するPythonスクリプト群を提供する。

## ライセンス
MIT licence.

- Copyright (C) 2022 Akira Tamamori
- Copyright (C) 2022 北原 鉄朗 (Tetsuro Kitahara)

## 依存パッケージ
実装はUbuntu 22.04上でテストした。Pythonのパージョンは`3.10.6`である。

- torch
- joblib
- midi2audio
- hydra-core
- progressbar2
- numpy
- scipy
- matplotlib

Ubuntuで動かす場合、**FluidSynthに関するサウンドフォントが必要**なので入れておく。

```bash

apt-get install fluidsynth

```

## 動かし方

|ファイル名|機能|
|---|---|
|preprocess.py | 前処理を実施するスクリプト|
|training.py |モデルの訓練を実施するスクリプト|
|synthesis.py | 訓練済のモデルを用いてメロディを合成するスクリプト|

各種の設定はyamlファイル（config.yaml）に記述する。

>
1. config.yamlの編集
2. preprocess.py による前処理の実施
3. training.pyによるモデル訓練の実施
4. synthesis.pyによるメロディ合成の実施

<u>preprocess.pyは一度だけ動かせばよい</u>。preprocess.pyにはモデル訓練に用いるMusicXML群（下記参照）のダウンロードや、それらからの特徴量抽出、またスターターキットが提供する伴奏データ・コード進行データのダウンロードが含まれる。

synthesis.pyには合成結果のMIDIファイルへの書き出し、Wavファイルへのポート、またメロディのピアノロール画像の作成・保存が含まれる。

### 使用データ
訓練データは以下のサイトから入手可能なMusicXMLを用いる。

https://homepages.loria.fr/evincent/omnibook/

Ken Deguernel, Emmanuel Vincent, and Gerard Assayag.
"Using Multidimensional Sequences for Improvisation in the OMax Paradigm",
in Proceedings of the 13th Sound and Music Computing Conference, 2016.
