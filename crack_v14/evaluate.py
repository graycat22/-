"""Compare results with labels/defect_annotations.csv.

python -m crack_v14.evaluate --labels labels/defect_annotations.csv --pred out [--role VALIDATION]

labels columns: record_type,path,dataset_role,label_status,type,judgment,severity,x1,y1,x2,y2,note
  - only label_status == CONFIRMED is used
  - an image is NG if any of its rows has judgment == NG
  - rows with x1..y2 filled are ground-truth defect boxes (input-image coordinates)
pred: judgments.csv + results.csv written by crack_v14.cli
A NG image counts as "located" if any predicted defect box overlaps a GT box (GT box padded by --pad px).
"""
import argparse, csv, os
from collections import defaultdict

def key(p):
    return os.path.splitext(os.path.basename(p))[0]

def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def overlap(a, b, pad):
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", default="out")
    ap.add_argument("--role", default=None, help="TRAIN / VALIDATION / TEST (default: all)")
    ap.add_argument("--pad", type=float, default=20)
    a = ap.parse_args()

    gt_ng, gt_box, role = {}, defaultdict(list), {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8-sig")):
        if r.get("label_status", "CONFIRMED").strip().upper() != "CONFIRMED":
            continue
        if a.role and r.get("dataset_role", "").strip().upper() != a.role.upper():
            continue
        k = key(r["path"]); role[k] = r.get("dataset_role", "")
        j = r.get("judgment", "").strip().upper()
        gt_ng[k] = gt_ng.get(k, False) or j == "NG"
        box = [num(r.get(c)) for c in ("x1", "y1", "x2", "y2")]
        if j == "NG" and None not in box:
            gt_box[k].append(box)

    pred = {r["img"]: r["judgment"] == "NG" for r in csv.DictReader(open(os.path.join(a.pred, "judgments.csv")))}
    pbox = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(a.pred, "results.csv"))):
        if r["ng"] == "True":
            pbox[r["img"]].append([float(r[c]) for c in ("x1", "y1", "x2", "y2")])

    tp = fp = fn = tn = located = loc_total = 0
    misses, false_alarms, mislocated = [], [], []
    for k, ng in sorted(gt_ng.items()):
        if k not in pred:
            print("not predicted:", k); continue
        p = pred[k]
        if ng and p:
            tp += 1
            if gt_box[k]:
                loc_total += 1
                if any(overlap(b, g, a.pad) for b in pbox[k] for g in gt_box[k]):
                    located += 1
                else:
                    mislocated.append(k)
        elif ng:
            fn += 1; misses.append(k)
        elif p:
            fp += 1; false_alarms.append(k)
        else:
            tn += 1
    n = tp + fp + fn + tn
    print(f"images={n}  TP={tp} FN={fn} FP={fp} TN={tn}")
    if tp + fn: print(f"NG recall (見逃し率の逆) = {tp / (tp + fn):.3f}")
    if fp + tn: print(f"OK false-alarm rate (過検出率) = {fp / (fp + tn):.3f}")
    if loc_total: print(f"location hit = {located}/{loc_total}")
    for name, lst in (("見逃し", misses), ("過検出", false_alarms), ("位置ずれ", mislocated)):
        if lst: print(f"{name}: {' '.join(lst)}")

if __name__ == "__main__":
    main()
