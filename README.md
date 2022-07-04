# エッジAI & R&Dチーム 案件&パッケージ用 テンプレートリポジトリ

このリポジトリはエッジAI & R&Dチームの案件&パッケージ用テンプレートリポジトリです。



<br>

---
## 機能

このテンプレートリポジトリを使用することで、以下の設定を引き継いだリポジトリを作成することができます。

- [GitHub Actions](https://github.com/features/actions)でのCI:
  - main, */main, develop, */developへのPR時のみ
  - docstringをunittestに統合した単体テスト
- GitHubのIssueテンプレート
- main, developブランチへの直接コミットを禁止とするpre-commit hook
- Visual Studio Code設定
    - エディタ設定
        - [mypy](https://github.com/python/mypy)による静的Typeチェック
        - [darglint](https://github.com/terrencepreilly/darglint)による静的docstringチェック
        - [flake8](https://flake8.pycqa.org/en/latest/)によるLinting.
        - [isort](https://pycqa.github.io/isort/)と[black](https://black.readthedocs.io/en/stable/)によるフォーマッティング
    - 推奨Extentionとその設定
        - [Commit Message Editor](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor)など
- setup.pyによるパッケージ化
- Sphinxによるdocstringの自動ドキュメント化
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), [ClearML](https://github.com/allegroai/clearml)などを使用したサンプル

<br>

---
## 使用方法

### 初期設定

1. 本リポジトリGitHubページの`Use this template`ボタンを押し、このテンプレートを使用したリポジトリを作成します

2. 仮想環境を作成します (Python 3.7以上を推奨)

3. 一時的に使用するセットアップ用のパッケージをインストールし、パッケージ名などを置換するスクリプトを実行します

    ```
    pip install -r setup-requirements.txt
    python personalize.py
    ```

    リポジトリ名、パッケージ名などについてプロンプトが表示されますのでそれに従ってください。デフォルトのmy_packageなどがリネームされます
    (.github/workflows/unittest.ymlのpythonバージョンのみ自動で置換されないようになっています。手動で書き換えてください)


4. 開発用のパッケージをインストールします(リンター、フォーマッタ、sphinxなど)
    ```
    pip install -r requirements/dev-requirements.txt
    または
    pip install -e '.[dev]'
    ```

5. .githooksディレクトリがhookとして認識されるように設定します
    ```
    git config --local core.hooksPath .githooks
    chmod -R +x .githooks/
    ```

6. 変更をCommit & Pushし完了です

### パッケージ化方法

0. (requirements/dev-requirements.txtを使ってwheelパッケージをインストールしておく)

1. setup.pyを確認

2. ビルド実行
    - ビルド済みのファイルを配布する場合 (whl)
        ```
        python setup.py bdist_wheel
        ```

    - sdist: ソースコードを配布する場合 (tar.gz)
        ```
        python setup.py sdist
        ```

    build, dist, my_package.eggなどが生成される。
    上記どちらでも、完成物はdist以下に生成される。

    生成された.whlは、以下のコマンドでインストールできます
    ```
    pip install /path/to/wheel
    ```
### docstringからのドキュメント作成方法

0. (requirements/dev-requirements.txtを使ってsphinx関連のパッケージをインストールしておく (`pip install '.[dev]'`でも可) )

1. docs/source下のindex.rst, installation.rst, overview.rstを適宜書き換える(納品までは特に書き換える必要はないかもしれない)

2. パッケージ内に新しいディレクトリ/クラス/関数を実装する都度、下記コマンドでドキュメントの内部ファイルを生成する
    ```
    sphinx-apidoc -f -o ./docs/source/docstring ./my_package
    ```
    (下記プレビュー版使用時も、このコマンドの結果はリアルタイムで反映される)

3. ビルド実行
    - プレビュー(コードを更新するとリアルタイムで更新される)版
        ```
        make dev
        ```

        実行すると[http://127.0.0.1:8000](http://127.0.0.1:8000)からドキュメントにアクセスできるようになります

    - 完成版HTML
        ```
        make build
        ```

        実行するとdocs/build下にhtmlフィアル群が生成されます。zipなどで固めて納品などに使えるかと思います

<br>

---
## 標準ディレクトリ構成 & サンプルコード説明

### ディレクトリ構成
本テンプレートリポジトリには`my_package/`下に予めディレクトリ構成が用意されています。

本テンプレートリポジトリを使用してみなさんに作成いただいたパッケージ/リポジトリは今後エッジAI & R&Dチームの資産として統合されていく可能性があります。統合と再利用を容易にするため、用意されたディレクトリ構成にできるだけ準拠してコーディングをお願いします。

ただしこれは強制ではありません。また、ディレクトリの削除/追加は案件/パッケージ化がやりやすいように適宜自由に行ってください。

```
├── configs/                       <- hydraのconfigディレクトリ
│   ├── default_demo.yaml          <- gradioデモサンプルのhydraデフォルトconfig
│   ├── default_lightning.yaml     <- PyTorch Lightning学習スクリプトサンプルのhydraデフォルトconfig
│   ├── callbacks/                 <- コールバック用configディレクトリ
│   ├── datamodule/                <- PyTorch LightningのDatamodule用configディレクトリ
│   ├── debug/                     <- デバッグ時用configディレクトリ
│   ├── experiment/                <- 各実験設定を記述した実際に多用するconfigディレクトリ
│   ├── logger/                    <- PyTorch Lightningのlogger用configディレクトリ
│   ├── model/                     <- PyTorch Lightningのモデル(LightningModule)用configディレクトリ
│   ├── trainer/                   <- PyTorch LightningのTrainer用configディレクトリ
│   └── transforms/                <- 入力transform(data augmentation)用configディレクトリ
│
├── data/                          <- データ保存用ディレクトリ
│
├── docs/                          <- ドキュメント用ディレクトリ
│
├── examples/                      <- サンプルのスクリプト用ディレクトリ
│   ├── example_gradio.py          <- デモのサンプルスクリプト
│   └── example_train_lightning.py <- PyTorch Lightning学習サンプルスクリプト
│
├── LICENSE                        <- ライセンスファイル
│
├── logs/                          <- ログファイル用ディレクトリ
│
├── Makefile                       <- ドキュメント生成用Makefile(`make dev`, `make build`)
│
├── my_package/                    <- ソース用ディレクトリ
│   ├── version.py                 <- バージョンを記載するスクリプト。パッケージ化時には変更が必要
│   ├── applications/              <- デモに関するディレクトリ
│   ├── callbacks/                 <- コールバック用ディレクトリ
│   ├── datamodules/               <- PyTorch LightningのDatamodule用ディレクトリ
│   ├── datasets/                  <- データセット用ディレクトリ
│   ├── experimental/              <- 実験的コード用ディレクトリ
│   ├── loggers/                   <- PyTorch Lightningのロガー用ディレクトリ
│   ├── losses/                    <- 損失関数用ディレクトリ
│   ├── metrics/                   <- メトリック用ディレクトリ
│   ├── models/                    <- モデル用ディレクトリ
│   ├── modules/                   <- PyTorch Lightningのモデル(LightningModule)用ディレクトリ
│   ├── optimizers/                <- optimizer用ディレクトリ
│   ├── postprocessing/            <- 後処理コード用ディレクトリ
│   ├── preprocessing/             <- 前処理コード用ディレクトリ
│   ├── trainers/                  <- PyTorch LightningのTrainer用ディレクトリ
│   ├── transforms/                <- 入力transform(data augmentation)用ディレクトリ
│   ├── utils/                     <- utilityコード用ディレクトリ
│   └── visualizations/            <- 可視化コード用ディレクトリ
│
├── notebooks/                     <- Jupyter notebook用ディレクトリ
│
├── personalize.py                 <- パッケージ名などを置換するスクリプト
│
├── README.md                      <- 本REAMDE
│
├── requirements/                  <- 開発用、デモ用など使用状況に基づくrequirements.txtを格納するディレクトリ
│   ├── dev-requirements.txt       <- 開発用requirements.txt
│   └── sample-requirements.txt    <- サンプルコード用requirements.txt
│
├── requirements.txt               <- 基本となるrequirements.txt
│
├── setup.py                       <- パッケージ化用スクリプト
│
├── setup-requirements.txt         <- `personalize.py`を動かすためのrequirements.txt
│
└── tests/                         <- テストコード用ディレクトリ
```


### サンプルコード
本テンプレートリポジトリには下記エッジAI & R&Dチーム推奨ツールを使用したサンプルコードが用意されています。
* ディープラーニングフレームワーク[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* 機械学習パラメータ管理ツール[hydra](https://github.com/facebookresearch/hydra)
* 実験管理ツール[ClearML](https://github.com/allegroai/clearml)
* データバージョン管理ツール[DVC](https://dvc.org)
* 機械学習デモ作成ライブラリ[Gradio](https://github.com/gradio-app/gradio)

#### 使用方法
0. 事前準備
    - sample-requirements.txtを使ってwheelパッケージをインストールしておく (`pip install '.[dev,sample]'`でも可)
    - ClearMLの初期設定を行っておく([こちら](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/)を参考に、[こちらのAmba君ClearML管理画面](http://10.10.10.79:8080/settings/workspace-configuration)からcredentialを取得してください)

1. PyTorch LightningによるMNISTの学習サンプル実行(GPU使用で6分程度で学習終了)

    適宜`configs/experiment`下のスクリプトを修正し、下記コマンドを実行
    ```
    python examples/example_train_lightning.py  +experiment=example_experiment_mnist
    ```

2. 実験結果の確認

    Amba君に用意したClearMLサーバーで実験管理の結果を確認([http://10.10.10.79:8080/dashboard](http://10.10.10.79:8080/dashboard))

3. 1.で学習したモデルを使用したMNIST画像分類デモ実行
    ```
    python examples/example_gradio.py model_state_dict=data/lightning_sample/exp_mnist/checkpoints/last.ckpt
    ```
    上記コマンド実行後、実行したマシンの`localhost:7860`あたりにデモページが起動する(ipを指定すれば手元のマシンからでも確認可能)

* DVCについて
    1. **本リポジトリテンプレートを案件用に使用する場合**
        * データをDVC管理しながら案件を遂行する分には本リポジトリはDVCに関与しません(DVCによるデータバージョンコントロールはdvcコマンドとgitで完結するため)
    2. 終了した案件のコードやデータの再利用可能性を高めるために**資産化するリポジトリのテンプレートとして本リポジトリテンプレートを使用する場合**
        * [mnistのdatamoduleサンプル](https://github.com/arayabrain/rd-prj-template/blob/develop/my_package/datamodules/image/classification/datamodule_general.py)にあるように、既存の(外部の)リポジトリでDVC管理している特定のコミットのデータをダウンロード(dvc get)するための関数とサンプルが含まれています
        * ユースケースとして、本リポジトリテンプレートから作成した資産化用リポジトリで、上記1.の案件用リポジトリで管理されたDVC情報を参照してデータセットをダウンロードするようにdvc getのパラメータ設定することで、案件中にdvc管理していた既存のデータセット(の履歴含む)をそのまま資産化リポジトリでも活用することができます
            * 例: [サンプルにおけるdvc getのパラメータ設定ファイル](https://github.com/arayabrain/rd-prj-template/blob/develop/configs/datamodule/mnist.yaml)では、[疑似案件用リポジトリ](https://github.com/arayabrain/dummy_prj_repo_mnist)でDVC管理されていたデータセットを再利用している想定で作っています


# TODOs

- [x] 機能拡充([Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)などを参考に)
- [x] 実験管理(ClearML)などのサンプルの追加
- [x] DVC

<br> <br>


**本READMEの内容は適宜各自のREADMEに変更してください(以下は例)**

<div align="center">

# Your Project/Package Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## 概要


## 環境構築

Install dependencies

```bash
# clone project
git clone GITREPOURL
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=PYTHONVERSION
conda activate myenv

# install requirements
pip install -r requirements.txt
```

githooksディレクトリがhookとして認識されるように設定します
```
git config --local core.hooksPath .githooks
chmod -R +x .githooks/
```

## 使用方法
...