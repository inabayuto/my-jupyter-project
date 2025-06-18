# ml-project

A Python ML/DL/data analysis project.

## セットアップ手順

1. **uvのインストール**
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **パスをzshに通す**
   ```sh
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **インストール確認**
   ```sh
   which uv
   ```

4. **仮想環境構築**
   ```
   source .venv/bin/activate
   ```

5. **Pythonバージョン指定で仮想環境作成**
   ```sh
   uv venv --python 3.11
   ```

6. **依存パッケージのインストール**
   ```sh
   uv pip install -r pyproject.toml
   ```

---

## 仮想環境の構造

作成される `.venv` ディレクトリの構造例：

```
.venv/
├── bin/          # 実行ファイル (Linux/macOS)
├── include/      # ヘッダーファイル
├── lib/          # ライブラリ
└── pyvenv.cfg    # 環境設定
```

---

## pyproject.toml の依存内容

主要な依存パッケージ:

- python = "^3.11"
- pandas, numpy, scipy
- matplotlib, seaborn, plotly
- scikit-learn, xgboost, lightgbm, catboost
- tensorflow, torch, torchvision
- jupyterlab, notebook, ipython, tqdm

---

## uvコマンドの説明

- `uv pip install -r requirements.txt`
  - `pyproject.toml` で指定した依存パッケージを仮想環境に高速インストールします。
  - 依存関係を自動解決し、必要なバージョンをインストールします。

---
