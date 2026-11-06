"""2026-10-20：图文阶段日志分任务曲线、文字验证与低频校准摘要。"""
import argparse
import json
from pathlib import Path
import statistics


def rows(path):
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();out=args.out
    log=rows(out/'train.log');updates=[r for r in log if r.get('success')]
    failed=[r for r in log if r.get('success') is False]
    valid=[json.loads((out/'validation_initial.json').read_text()),*rows(out/'validation.jsonl')]
    calibration=rows(out/'text_aux_calibration.jsonl')
    tasks=('T1','O1','O2','S1','S2','S3','S4','S5')
    task_summary={t:dict(first30=statistics.median(r['task_nll'][t] for r in updates[:30] if r['task_nll'][t] is not None),
                         last30=statistics.median(r['task_nll'][t] for r in updates[-30:] if r['task_nll'][t] is not None)) for t in tasks}
    report=dict(logged_successful_updates=len(updates),logged_failed_updates=len(failed),
        oom_retries=sum(bool(r.get('capacity_oom_retry')) for r in log),task_nll=task_summary,
        validation_initial=valid[0],validation_latest=valid[-1],
        calibration_count=len(calibration),alpha_first=calibration[0]['alpha'],alpha_last=calibration[-1]['alpha'],
        alpha_min=min(r['alpha'] for r in calibration),alpha_max=max(r['alpha'] for r in calibration))
    (out/'curves_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for ax,selected in zip(axes[0],(('O1','S2'),('S4',),('T1','O2','S1','S3','S5'))):
        for task in selected:
            bins={}
            for r in updates:
                if r['task_nll'][task] is not None:bins.setdefault(int(r['successful_inputs']//10000000),[]).append(r['task_nll'][task])
            ax.plot([k*10+5 for k in sorted(bins)],[statistics.median(bins[k]) for k in sorted(bins)],label=task)
        ax.set(xlabel='Phase inputs (million)',ylabel='Body geometric NLL');ax.legend();ax.grid(alpha=.2)
    x=[v['successful_inputs']/1e6 for v in valid]
    for group in ('all','first','continuation'):
        axes[1,0].plot(x,[v['text_aux'][group]['shape_ce'] for v in valid],label=group)
    axes[1,0].set(ylabel='Held-out text shape CE',xlabel='Phase inputs (million)');axes[1,0].legend()
    axes[1,1].plot(x,[v['content_nll'] for v in valid]);axes[1,1].set(ylabel='Held-out text NLL',xlabel='Phase inputs (million)')
    axes[1,2].plot([v['successful_inputs']/1e6 for v in calibration],[v['alpha'] for v in calibration],'.-')
    axes[1,2].set(ylabel='Calibrated alpha',xlabel='Phase inputs (million)')
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('logged_successful_updates','logged_failed_updates','oom_retries','task_nll','calibration_count','alpha_first','alpha_last')}))


if __name__=='__main__':main()
