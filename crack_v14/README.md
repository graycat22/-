# カセットワレ検査 V14（横断経路方式）

## 考え方
割れ = 「U字内縁 → 外縁（左右リブ／下端）を、割れらしい画素だけを通って抜けられる経路」。
画像上で最小コスト経路（MCP / Dijkstra）を1回解き、

- `ratio`   = 経路上で健全部を通らされた長さ ÷ その位置の健全な肉厚（0=端から端まで割れ、1=健全）
- `max_gap` = 経路上で割れらしくない区間の最長連続長 [px]

- `rim_break` = 経路の起点で U字内縁の白い縁取りがどれだけ途切れているか（0=無傷、1=完全に切れている）

で判定する。
- B-NG: `ratio<=0.35 AND max_gap<=25`
- D-NG: 上記 `AND rim_break>=0.30`
  （人間の判断と同じ: 本当にパックリ割れていれば縁取りが切れる。傷・成形線・影は縁取りを切らない。
  ゲート切り欠きの位置は `rim_ignore` で除外）
3帯域支持・gap・角度ずれ・span・fallback・SVM は全部これに置き換わる。

## 処理
1. 種類判定 B/D（開口部が一様に明るい＝背面光あり＝B）
2. 基準画像へ位置合わせ（SIFT+RANSAC ホモグラフィ → ECC 微調整）
3. 基準座標で定義した `opening / plate / inner / outer` マスク（`geometry.py`）
4. 割れらしさマップ
   - D: 相対暗さ（多スケール black-hat ÷ 局所明るさ）× 細長さ（Hessian 固有値比）
   - B: 局所ボディ輝度に対する明るさ（背面光の漏れ）
5. 横断経路探索。見つけた経路の周囲を消して最大3本まで探す（複数割れ対応）
6. D 候補は縁取りの途切れ（`rim_break`）を必須にする。B画像上の D 候補は B 開口の暗い縁を除外

## 実行
```
pip install -r requirements.txt
python -m crack_v14.cli --input <画像フォルダ> --out out            # 実画像
python -m crack_v14.cli --input data/raw_screen --out out --screen  # 画面撮影写真
```
出力: `out/<名前>.jpg`（経路と BBOX）、`out/result_grid.jpg`、`out/results.csv`

## 実画像に載せ替えるとき
- `data/refs/ref_B.png`, `ref_D.png` を実画像の代表1枚ずつに差し替え、
  `geometry.py` の座標をその画像上で描き直す（今は画面写真 960x660 の座標）。
- 画角固定なので位置合わせは小さなずれ補正だけになり、今より安定するはず。

## 未解決（OK画像が来たらやること）
- 誤検出率は未測定（OK画像ゼロのため）。閾値 `ratio_th / gap_th / d_lo / d_hi / rim_break_th` は暫定。
- OK画像から画素ごとの median/MAD を作り、割れらしさを「OKからのずれ」に置き換える
  → 縁・リブ・刻印・ゲート・成形線などの固定構造が消え、`edge_margin` 等の手当てが不要になる。
- 刻印違い（MADE IN CHINA / eDV-…）は OK基準を分ける。
