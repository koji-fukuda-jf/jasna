# Jasna ファイル構成ガイド（概要）

このドキュメントは、`jasna` リポジトリの「各ファイルの役割」と「処理フロー」を、実装を追いやすい粒度で整理したものです。

## 1. エントリポイントと起動

- `jasna/__main__.py`
  - `python -m jasna` 実行時の入口。
  - `main.py` の `main()` を呼ぶだけの薄いラッパーです。
- `jasna/main.py`
  - CLI 引数定義（検出モデル、復元モデル、TVAI/Swin2SR、エンコード設定など）。
  - 実行前チェック（ffmpeg/ffprobe/mkvmerge、GPU 要件、入力ファイル存在、数値検証）。
  - 復元パイプラインを構築し、`Pipeline.run()` を呼びます。
- `jasna/bootstrap.py`
  - ローカル開発時の `sys.path` を安全に正規化。
- `jasna/os_utils.py`
  - 実行ファイル探索、ffmpeg バージョン検証、Windows 固有設定チェック、設定ディレクトリ解決。

## 2. コア処理（動画パイプライン）

- `jasna/pipeline.py`
  - 実行の中核。読み込み → 検出 → 追跡 → 復元 → ブレンド → エンコードを1ループで進行。
  - `NvidiaVideoReader` / `NvidiaVideoEncoder` を使った GPU ベース I/O を管理。
- `jasna/pipeline_processing.py`
  - フレームバッチ単位の実処理ロジック。
  - 検出結果を `ClipTracker` に渡し、終了クリップを `RestorationPipeline` へ流し、出力可能フレームを返す。
- `jasna/pipeline_overlap.py`
  - temporal overlap と crossfade の重み・保持範囲計算。

## 3. モザイク検出

- `jasna/mosaic/detection_registry.py`
  - 検出モデル名（RF-DETR / YOLO）の正規化とデフォルト重みパス決定。
- `jasna/mosaic/rfdetr.py`
  - RF-DETR ベース検出器実装。
- `jasna/mosaic/yolo.py`
  - YOLO ベース検出器実装。
- `jasna/mosaic/detections.py`
  - 検出結果データ構造。
- `jasna/mosaic/yolo_tensorrt_compilation.py`
  - YOLO の TensorRT 化支援。

## 4. 追跡・ブレンド

- `jasna/tracking/clip_tracker.py`
  - フレーム時系列上で同一モザイク領域をクリップとして追跡。
  - max clip / overlap 条件でクリップを分割・終了管理。
- `jasna/tracking/frame_buffer.py`
  - 復元待ち・ブレンド待ちのフレーム状態を保持。
  - どのトラックの処理が終われば出力可能かを管理。
- `jasna/tracking/blending.py`
  - 復元結果を原フレームに合成する関数群。

## 5. 復元パイプライン

- `jasna/restorer/restoration_pipeline.py`
  - クリップ単位の入力準備（bbox 拡張、256x256 前処理）と復元処理の司令塔。
  - 1次復元（BasicVSR++）→ 任意2次復元（TVAI / Swin2SR）→ denoise → blend まで接続。
- `jasna/restorer/basicvsrpp_mosaic_restorer.py`
  - BasicVSR++ 本体呼び出し。
- `jasna/restorer/basicvrspp_tenorrt_compilation.py`
  - BasicVSR++ TensorRT 利用/コンパイル判定。
- `jasna/restorer/secondary_restorer.py`
  - 2次復元の抽象インターフェース。
- `jasna/restorer/tvai_secondary_restorer.py`
  - Topaz Video AI (ffmpeg filter) を使う2次復元。
- `jasna/restorer/swin2sr_secondary_restorer.py`
  - Swin2SR 2次復元。
- `jasna/restorer/swin2sr_tensorrt_compilation.py`
  - Swin2SR TensorRT 化支援。
- `jasna/restorer/denoise.py`
  - 復元後ノイズ低減。
- `jasna/restorer/restored_clip.py`
  - 復元済みクリップ表現。

## 6. 動画 I/O とメディア補助

- `jasna/media/video_decoder.py` / `video_nv_decoder.py`
  - フレームデコード（GPU/NV 系含む）。
- `jasna/media/video_encoder.py`
  - 出力エンコード（HEVC 等設定）。
- `jasna/media/rgb_to_p010.py`
  - 色空間/フォーマット変換補助。
- `jasna/media/__init__.py`
  - メディア系公開 API（メタデータ取得・エンコード設定パース等）。

## 7. GUI

- `jasna/gui/app.py`
  - GUI 全体のウィンドウ・画面構築、イベント制御。
- `jasna/gui/processor.py`
  - GUI から CLI 相当処理をジョブとして実行。
- `jasna/gui/settings_panel.py` / `queue_panel.py` / `control_bar.py` / `log_panel.py`
  - 設定、キュー、操作バー、ログ UI。
- `jasna/gui/validation.py`
  - GUI 開始前の入力/環境チェック。
- `jasna/gui/wizard.py`
  - 初回セットアップ導線。
- `jasna/gui/locales.py`
  - i18n 文言とロケール管理。
- `jasna/gui/system_stats.py`
  - GPU/CPU/VRAM 等の統計表示。
- `jasna/gui/components.py` / `theme.py` / `models.py`
  - 共通コンポーネント、テーマ定数、GUI データモデル。

## 8. TensorRT・モデル関連

- `jasna/trt/trt_runner.py`
  - TensorRT 実行ユーティリティ。
- `jasna/trt/torch_tensorrt_export.py`
  - torch-tensorrt エクスポート支援。
- `jasna/models/basicvsrpp/*`
  - BasicVSR++ 実装一式。
  - `mmagic/` 以下は移植・互換レイヤを含む補助群。

## 9. パッケージング・ビルド

- `jasna/packaging/windows_dll_paths.py`
  - Windows DLL パス調整。
- `jasna/packaging/pyinstaller_runtime_hook_windows_dll_paths.py`
  - PyInstaller 実行時フック。
- `jasna.spec`
  - PyInstaller ビルド定義。
- `build_exe.py`
  - 実行ファイル作成補助。

## 10. テスト

- `tests/`
  - CLI 起動、GUI 検証、追跡、ブレンド、TensorRT コンパイル分岐、ロケール、OS 依存処理などをカバー。
  - 例:
    - `test_main_entry.py`: エントリポイント
    - `test_pipeline_processing.py`: バッチ処理中心
    - `test_restoration_pipeline.py`: 復元フロー
    - `test_gui_*`: GUI ロジック

## 11. 処理フロー（入口から出口まで）

1. `main.py` が引数を解釈し、実行環境とパラメータを検証。
2. 検出器（RF-DETR/YOLO）と復元器（BasicVSR++ + optional secondary）を構築。
3. `Pipeline.run()` が動画をバッチ読込。
4. 各バッチで検出 → 追跡し、終了したクリップを抽出。
5. `RestorationPipeline` がクリップを前処理し、1次/2次復元と denoise を実施。
6. `FrameBuffer` で元フレームへブレンド、出力可能順にエンコーダへ渡す。
7. 最終 flush 後に残フレームを吐き出して完了。

## 12. まず読む順番（おすすめ）

1. `README.md`
2. `jasna/main.py`
3. `jasna/pipeline.py`
4. `jasna/pipeline_processing.py`
5. `jasna/restorer/restoration_pipeline.py`
6. `jasna/tracking/clip_tracker.py` + `frame_buffer.py`
7. 目的に応じて `mosaic/`・`gui/`・`media/` を深掘り
