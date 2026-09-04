
def failure_rate_report(results, name, rot_thresh, trans_thresh, diagnostic_cols):
    df = results.copy()
    df["pose_err"] = df[["R_err_deg", "t_err_deg"]].max(axis=1)
    df["rot_failure"] = df["R_err_deg"] > rot_thresh
    df["trans_failure"] = df["t_err_deg"] > trans_thresh
    df["is_failure"] = df["rot_failure"] | df["trans_failure"]

    both = (df["rot_failure"] & df["trans_failure"]).mean()
    either = df["is_failure"].mean()
    print(f"\n{'='*15} {name} {'='*15}")
    print(f"rot_failure={df['rot_failure'].mean():.2%} | trans_failure={df['trans_failure'].mean():.2%} | "
          f"both={both:.2%} | either={either:.2%} | n={len(df)} ({int(df['is_failure'].sum())} failures)")

    available = [c for c in diagnostic_cols if c in df.columns]
    # reindex columns=[False, True] гарантирует обе колонки даже если одна из групп пуста
    comp = df.groupby("is_failure")[available].mean().T.reindex(columns=[False, True])
    comp.columns = ["ok", "failure"]
    comp["diff"] = comp["failure"] - comp["ok"]
    print("\nok vs failure (mean diagnostics):")
    print(comp.to_string())

    if df["is_failure"].sum() == 0:
        print("(0 failures — Spearman corr с pose_err не показателен, распределение вырождено)")
        return df

    corr = (
        df[available + ["pose_err"]]
        .corr(method="spearman")["pose_err"]
        .drop("pose_err")
        .sort_values(key=abs, ascending=False)
    )
    print("\nSpearman corr with pose_err:")
    print(corr.to_string())
    return df

def percentile_threshold(val_results, col, q=0.90):
    return val_results[col].quantile(q)