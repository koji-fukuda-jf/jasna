# モザイク Detection Model に新モデルを追加する手順

このプロジェクトで検出モデルを追加する場合は、**「実装を追加する」** と **「モデル名を配線する」** の2段階で行います。

## 1. 追加するモデル形式を決める

現在の実装は大きく2系統です。

- **RF-DETR系（ONNX）**: `jasna/mosaic/rfdetr.py`
- **YOLO系（.pt / TensorRT engine）**: `jasna/mosaic/yolo.py`

追加したいモデルが既存どちらかの入出力に合わせられるなら、まずはその系統に寄せるのが最小変更です。

---

## 2. 既存系統へ「新バリアント」を追加する（最小変更ルート）

### 2-1. モデル名と重みファイル名をレジストリへ登録

`jasna/mosaic/detection_registry.py` を更新します。

- `RFDETR_MODEL_NAMES` または `YOLO_MODEL_NAMES` に新しい名前を追加
- YOLO系なら `YOLO_MODEL_FILES` に `"モデル名": "重みファイル名.pt"` を追加
- `model_weights/` に重みファイルを配置（RF-DETR は `*.onnx`、YOLO は `*.pt`）

### 2-2. CLI の選択肢を追加

`jasna/main.py` の `--detection-model` の `choices` にモデル名を追加します。

### 2-3. GUI の選択肢とデフォルト注釈を更新

以下を更新します。

- `jasna/gui/settings_panel.py` の `CTkOptionMenu(... values=[...])`
- `jasna/gui/models.py` の `AppSettings.detection_model` コメント（利用可能モデル一覧）

### 2-4. 事前チェックの分岐が必要なら更新

GUI の事前チェックでモデル種別ごとの分岐を使っているため、必要に応じて次を確認します。

- `jasna/gui/engine_preflight.py`

通常は `detection_registry.py` に正しく追加すれば動きます。

---

## 3. 新しい推論バックエンド自体を増やす場合（実装追加ルート）

既存の RF-DETR/YOLO に載らない場合は、検出器クラスを新規実装し、`Pipeline` に分岐を追加します。

### 3-1. 新しい検出器クラスを作る

`jasna/mosaic/` 配下に新規ファイルを作り、`__call__(frames_uint8_bchw, target_hw=...) -> Detections` を実装します。

返り値の `Detections` 仕様は `jasna/mosaic/detections.py` を踏襲してください。

- `boxes_xyxy`: フレームごとの `N x 4`（ピクセル座標）
- `masks`: フレームごとの `N x Hm x Wm`（追跡側で座標変換して使う）

### 3-2. `Pipeline` でモデル名→クラスの分岐を追加

`jasna/pipeline.py` の初期化処理で、モデル名に応じて新クラスを生成する分岐を追加します。

### 3-3. レジストリ・CLI・GUI も同時に配線

- `jasna/mosaic/detection_registry.py`
- `jasna/main.py`
- `jasna/gui/settings_panel.py`
- `jasna/gui/models.py`

---

## 4. 動作確認チェックリスト

1. `model_weights/` に対象重みが存在する。
2. CLI で新モデル名を指定して起動できる。
3. GUI の Detection Model に新モデルが出る。
4. 推論時に `FileNotFoundError` や model type 分岐エラーが出ない。
5. 検出結果が `Detections` 形式で返り、追跡〜復元まで流れる。

---

## 補足（運用上の注意）

- 新モデル名が `detection_registry.py` に未登録だと、入力は `coerce_detection_model_name()` でデフォルト (`rfdetr-v3`) に丸められます。
- YOLO系は初回に TensorRT エンジン生成が走ることがあります（環境依存で時間がかかる）。
