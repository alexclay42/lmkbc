import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

RELATION_TYPE = {
    "awardWonBy": "string",
    "hasCapacity": "numeric",
    "hasArea": "numeric",
    "countryLandBordersCountry": "string",
    "personHasCityOfDeath": "string",
    "companyTradesAtStockExchange": "string"
}


def read_jsonl_file(file_path: Union[str, Path]) -> List[Dict]:
    with open(file_path, "r") as f:
        rows = [json.loads(line) for line in f]
    return rows


def true_positives(preds: List[str], gts: List[str], rel: str, rel_type: str, tolerance: float = 0.05) -> int:
    if rel_type == "numeric":
        return numeric_true_positives(preds, gts, tolerance)
    elif rel_type == "string":
        return string_true_positives(preds, gts)
    else:
        raise ValueError(f"Unknown relation type: {rel_type}")


def try_parse_number(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def numeric_true_positives(preds: List[str], gts: List[str], tolerance: float = 0.05) -> int:
    tp = 0
    gt_nums = [try_parse_number(gt) for gt in gts]
    gt_nums = [gt for gt in gt_nums if gt is not None]

    for pred in preds:
        pred_num = try_parse_number(pred)
        if pred_num is None:
            continue
        for gt_num in gt_nums:
            if abs(pred_num - gt_num) / gt_num <= tolerance:
                tp += 1
                break
    return tp


def string_true_positives(preds: List[str], gts: List[str]) -> int:
    return sum(1 for pred in preds if pred.strip() in gts)


def precision(preds: List[str], gts: List[str]) -> float:
    try:
        # When nothing is predicted, precision = 1
        # irrespective of the ground truth value
        if len(preds) == 0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1
        # even if there are predictions (edge case)
        if len(gts) == 0:
            return 1.0
        # When the ground truth is not empty
        return min(true_positives(preds, gts) / len(gts), 1.0)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    """ Index the ground truth/prediction rows by subject entity and relation. """
    return {
        (r["SubjectEntity"], r["Relation"]): list(set(r["ObjectEntities"]))
        for r in rows
    }


def evaluate_per_sr_pair(pred_rows, gt_rows, rel_types: Dict[str, str], tolerance: float = 0.05) -> List[
    Dict[str, float]]:
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        gts = gt_dict[(subj, rel)]
        preds = pred_dict.get((subj, rel), [])

        rel_type = rel_types.get(rel, "string")  # default to "string" if not found
        tp = true_positives(preds, gts, rel=rel, rel_type=rel_type, tolerance=tolerance)

        p = tp / len(preds) if preds else 1.0
        r = tp / len(gts) if gts else 1.0
        f1 = f1_score(p, r)

        results.append({
            "SubjectEntity": subj,
            "Relation": rel,
            "p": p,
            "r": r,
            "f1": f1,
            "tp": tp,
            "total_pred": len(preds),
            "total_gt": len(gts),
        })

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def macro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """ Compute the macro average scores per relation """
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append({
            "p": r["p"],
            "r": r["r"],
            "f1": r["f1"],
        })

    macro_averages = {}
    for rel in scores:
        macro_averages[rel] = {
            "macro-p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "macro-r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "macro-f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    # Macro average for all relations
    all_rel_macro_p = sum([x["p"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_r = sum([x["r"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_f1 = sum([x["f1"] for x in scores_per_sr]) / len(
        scores_per_sr)

    macro_averages["*** All Relations ***"] = {
        "macro-p": all_rel_macro_p,
        "macro-r": all_rel_macro_r,
        "macro-f1": all_rel_macro_f1,
    }

    return macro_averages


def micro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """ Compute the micro average scores per relation """
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = {
                "tp": 0,
                "total_pred": 0,
                "total_gt": 0,
            }
        scores[r["Relation"]]["tp"] += r["tp"]
        scores[r["Relation"]]["total_pred"] += r["total_pred"]
        scores[r["Relation"]]["total_gt"] += r["total_gt"]

    micro_averages = {}
    for rel in scores:
        micro_p = scores[rel]["tp"] / scores[rel]["total_pred"] if scores[rel][
                                                                       "total_pred"] > 0 else 1.0
        micro_r = scores[rel]["tp"] / scores[rel]["total_gt"] if scores[rel][
                                                                     "total_gt"] > 0 else 1.0

        micro_averages[rel] = {
            "micro-p": micro_p,
            "micro-r": micro_r,
            "micro-f1": f1_score(micro_p, micro_r),
        }

    # Micro average for all relations
    total_tp = sum([x["tp"] for x in scores.values()])
    total_pred = sum([x["total_pred"] for x in scores.values()])
    total_gt = sum([x["total_gt"] for x in scores.values()])

    all_rel_micro_p = total_tp / total_pred if total_pred > 0 else 1.0
    all_rel_micro_r = total_tp / total_gt if total_gt > 0 else 1.0

    micro_averages["*** All Relations ***"] = {
        "micro-p": all_rel_micro_p,
        "micro-r": all_rel_micro_r,
        "micro-f1": f1_score(all_rel_micro_p, all_rel_micro_r),
    }

    return micro_averages


def prediction_statistics(scores_per_sr: List[Dict[str, float]]) -> dict:
    """ Get the average numbers of predictions and the numbers of empty predictions per relation. """
    stats = {}
    for r in scores_per_sr:
        if r["Relation"] not in stats:
            stats[r["Relation"]] = {
                "num_sr_pairs": 0,
                "total_pred": 0,
                "empty_pred": 0,
            }
        stats[r["Relation"]]["num_sr_pairs"] += 1
        stats[r["Relation"]]["total_pred"] += r["total_pred"]
        if r["total_pred"] == 0:
            stats[r["Relation"]]["empty_pred"] += 1

    final_stats = {}
    for rel in stats:
        final_stats[rel] = {
            "avg. #preds": stats[rel]["total_pred"] / stats[rel][
                "num_sr_pairs"],
            "#empty preds": stats[rel]["empty_pred"],
        }

    # Average numbers of predictions and the numbers of empty predictions for all relations
    total_sr_pairs = len(scores_per_sr)
    total_preds = sum([x["total_pred"] for x in stats.values()])
    total_empty_preds = sum([x["empty_pred"] for x in stats.values()])

    final_stats["*** All Relations ***"] = {
        "avg. #preds": total_preds / total_sr_pairs,
        "#empty preds": total_empty_preds,
    }

    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall and F1-score of predictions")

    parser.add_argument(
        "-p", "--predictions",
        type=str,
        required=True,
        help="Path to the predictions file (required)"
    )
    parser.add_argument(
        "-g", "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground truth file (required)"
    )

    args = parser.parse_args()

    # Read the predictions and ground truth
    pred_rows = read_jsonl_file(args.predictions)
    gt_rows = read_jsonl_file(args.ground_truth)

    # Evaluate the predictions
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows, RELATION_TYPE, tolerance=0.05)

    # Macro average
    macro_per_relation = macro_average_per_relation(scores_per_sr_pair)
    macro_df = pd.DataFrame(macro_per_relation).transpose().round(3)

    # Micro average
    micro_per_relation = micro_average_per_relation(scores_per_sr_pair)
    micro_df = pd.DataFrame(micro_per_relation).transpose().round(3)

    # Statistics
    stats = prediction_statistics(scores_per_sr_pair)
    stats_df = pd.DataFrame(stats).transpose().round(3)
    stats_df["#empty preds"] = stats_df["#empty preds"].astype(int)

    # Combine the results
    results = pd.concat([macro_df, micro_df, stats_df], axis=1)
    print(results)

    #modifications
    #return 
    results.to_csv("data/res.tsv", sep="\t")


if __name__ == "__main__":
    main()
