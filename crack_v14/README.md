# カセットワレ検査 V14

## 割れの定義
U字の内側（開口部）から外縁（左右の縦リブ・下端）まで、**端から端まで割れているもの**だけを NG にする。
途中で止まっている半割れ・表面の傷・成形線・影は OK。

- **B** = 明るい割れ（割れ目から背面光が漏れて白く見える）。背面光がある画像だけ検査
- **D** = 暗い割れ（黒い割れ線）。全部の画像で検査
- **製品の種類** = type1（U字の脚が上まで長く見える）/ type2（U字の内側にネジ筒が見える）

## 処理の流れ（inspect_v14.py）
1. **位置合わせ**: `template_table1.png`（右側のネジ2つと留め具）を `cv2.matchTemplate` で探し、X・Y のずれを補正して 960x660 に縮小（align.py）
2. **種類の判定**: type1・type2 の基準画像と輪郭を比べ、よく合う方にする（明るさは使わない）
3. **背面光の判定**: U字の開口部が明るいか
4. **割れらしさ**: 画素ごとに割れっぽさを 0〜1 で数値化（features.py）
   - D: 周りより暗く、細長い所
   - B: 周りの樹脂より明るい所
5. **経路探索**: 内縁から外縁まで、割れっぽい所だけを通って抜けられる道を探す（crossing.py, geodesic.py）
6. **判定**: 下の条件を全部満たす道があれば NG

## 判定の条件と閾値（inspect_v14.py の PARAMS）
| 閾値 | 値 | 意味 | 上げると |
|---|---|---|---|
| ratio_th | 0.35 | 道の上で割れていない所を通った割合の上限 | 見逃しが減り、過検出が増える |
| gap_th | 25 px | 割れていない区間の最長の上限 | 見逃しが減り、過検出が増える |
| end_gap_inner | 10 px | 割れが内側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| end_gap_outer | 25 px | 割れが外側の端から手前で止まってよい距離 | 半割れも NG になりやすい |
| rim_break_th | 0.30 | D: 起点で U字の縁取りが切れている度合いの下限 | 過検出が減り、見逃しが増える |
| reach_th | 0.90 | B: 漏れた光が開口部までつながっている割合の下限 | 過検出が減り、見逃しが増える |
| d_lo / d_hi | 0.10 / 0.30 | D: 暗さ（×細長さ）を割れらしさ 0〜1 にする範囲 | 薄い線を拾いにくくなる |
| b_lo / b_hi | 2.5 / 6.0 | B: 明るさを割れらしさ 0〜1 にする範囲 | 淡い光漏れを拾いにくくなる |

## 使い方
```
pip install -r requirements.txt

# 1. 基準を作る（最初の1回）。OK画像のフォルダは種類が混ざっていてよい
#    仮の基準画像 data/refs/ref_type1.png, ref_type2.png が必要
python -m crack_v14.make_ref --ok-dir pos1ok --template template_table1.png --out refs
#    -> refs/check_type1.jpg, check_type2.jpg で線を確認
#       緑 = U字内縁の縁取りのすぐ外側、赤 = 縦リブと下端、青 = 樹脂の部分 になっていれば OK
#       ずれていたら --move-geom DX DY（右・下に何px）、--scale-geom（何倍）を付けて作り直す

# 2. 検査
python -m crack_v14.cli --input <画像フォルダ> --template template_table1.png --ref-dir refs --out out

# 3. 正解CSVとの比較
python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out --role VALIDATION
```

## ファイル
| ファイル | 中身 |
|---|---|
| cli.py | 検査の実行、結果の描画と CSV 出力 |
| inspect_v14.py | 検査の流れと判定（閾値はここ） |
| align.py | テンプレートマッチングでの位置合わせ |
| features.py | 割れらしさマップ（B・D） |
| crossing.py | 内縁 -> 外縁の経路探索、縁取りの途切れ |
| geodesic.py | 最短経路探索（ダイクストラ法） |
| geometry.py | 検査範囲の座標とマスク |
| make_ref.py | OK画像から基準を作る（最初の1回） |
| register.py | 特徴点照合（make_ref だけで使用） |
| evaluate.py | 正解CSVとの比較 |
| screen_crop.py | （テスト用）画面撮影写真の切り出し |

## 未確認のこと
- OK画像での過検出率（OK画像で回していないので不明）
- 本物の template_table1.png・実画像での動作（テストは画面写真を加工したもので実施）
- No.16 の中央の細い線は検出できていない（同じ位置に成形線があり、OK画像との比較が必要）
