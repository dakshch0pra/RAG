import pandas as pd
import ast, re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# ------------------------
# Text Normalization
# ------------------------
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^a-z0-9 %:.,-]', '', s)
    return s

# ------------------------
# Token-level Precision, Recall, F1 (SQuAD style)
# ------------------------
def token_prf(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    common = Counter(p) & Counter(r)
    num_same = sum(common.values())
    if len(p)==0 or len(r)==0:
        return 0.0, 0.0, 0.0
    prec = num_same/len(p)
    rec = num_same/len(r)
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

# ------------------------
# ROUGE-L (Longest Common Subsequence)
# ------------------------
def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_l_prf(pred, ref):
    pred_tokens, ref_tokens = pred.split(), ref.split()
    lcs_len = lcs(pred_tokens, ref_tokens)
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision+recall)>0 else 0.0
    return precision, recall, f1

# ------------------------
# TF-IDF Cosine Similarity
# ------------------------
def tfidf_cosine(preds, refs):
    texts = list(preds) + list(refs)
    vec = TfidfVectorizer(min_df=1).fit(texts)
    P = vec.transform(preds); R = vec.transform(refs)
    return [cosine_similarity(P[i], R[i])[0,0] for i in range(len(preds))]

# ------------------------
# Citation Fuzzy Matching
# ------------------------
def to_listish(x):
    try:
        v = ast.literal_eval(x)
        if isinstance(v, dict): v = list(v.values())
        if isinstance(v, (list, tuple, set)): return list(v)
    except: pass
    return []

def fuzzy_match(a, b, thresh=0.8):
    return SequenceMatcher(a=a.lower(), b=b.lower()).ratio() >= thresh

def citation_prf(gt_cites, pred_cites, thresh=0.8):
    matched = set()
    tp = 0
    for pc in pred_cites:
        for i, gc in enumerate(gt_cites):
            if i in matched: continue
            if fuzzy_match(pc, gc, thresh):
                matched.add(i); tp += 1; break
    precision = tp/len(pred_cites) if pred_cites else 0.0
    recall = tp/len(gt_cites) if gt_cites else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return precision, recall, f1

# ------------------------
# Main Evaluation
# ------------------------
def evaluate_rag(csv_file):
    df = pd.read_csv(csv_file)
    smooth = SmoothingFunction().method1
    results = []

    tfidf_scores = tfidf_cosine(df.llm_response, df.ground_truth)

    for i, row in df.iterrows():
        gt = row["ground_truth"]
        pred = row["llm_response"]

        # BLEU
        bleu = sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth)

        # ROUGE-L
        rougeP, rougeR, rougeF1 = rouge_l_prf(pred, gt)

        # Token-level PRF
        tokP, tokR, tokF1 = token_prf(pred, gt)

        # TF-IDF cosine
        tfidf = tfidf_scores[i]

        # Citation metrics
        gt_list, pred_list = to_listish(row["ground_truth_citation"]), to_listish(row["llm_response_citation"])
        c_prec, c_rec, c_f1 = citation_prf(gt_list, pred_list)

        results.append({
            "question": row["question"],
            "BLEU": bleu,
            "ROUGE-L Precision": rougeP,
            "ROUGE-L Recall": rougeR,
            "ROUGE-L F1": rougeF1,
            "Token Precision": tokP,
            "Token Recall": tokR,
            "Token F1": tokF1,
            "TF-IDF Cosine": tfidf,
            "Citation Precision": c_prec,
            "Citation Recall": c_rec,
            "Citation F1": c_f1
        })

    results_df = pd.DataFrame(results)
    print("\n=== Per-question Results ===\n", results_df)
    print("\n=== Average Scores ===\n", results_df.mean(numeric_only=True))
    return results_df


if _name_ == "_main_":
    df_results = evaluate_rag("evaluation_sheet.csv")
