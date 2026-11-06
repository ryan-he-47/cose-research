"""CPU临时目录核对检查点清理：保留final/恢复源，元数据变化时先拒绝全部删除。"""
from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.hpc.prune_checkpoints import RESUME,build_plan,apply_plan


def main():
    workspace=Path(__file__).resolve().parents[2]
    fixture_parent=workspace/'tmp_smoke';fixture_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prune_check_',dir=fixture_parent) as temporary:
        root=Path(temporary).resolve()
        assert root.is_relative_to(fixture_parent.resolve())
        for name in (RESUME,'r_t1_linear13_chinchilla_20261014/final.pt',
                     'r_t1_linear13_chinchilla_20261014/ckpt_1.pt','ae/final.pt','ae/best.pt',
                     'unfinished/ckpt_1.pt','r_t1_quadrature_aug_chinchilla_20261010/ckpt_1.pt',
                     'r_t1_quadrature_aug_resume_20261011/final.pt'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
        pointer=root/'r_t1_quadrature_aug_chinchilla_20261010/latest.json'
        pointer.write_text(json.dumps({'checkpoint':'ckpt_1.pt'}),encoding='utf-8')
        plan=build_plan(root)
        assert len(plan['candidates'])==3 and len(plan['retained'])==4 and len(plan['untouched'])==1
        changed=root/'ae/best.pt';changed.write_bytes(b'changed')
        try:apply_plan(plan,root/'failed.jsonl')
        except RuntimeError:pass
        else:raise AssertionError('变动文件未拒绝')
        assert all((root/r['path']).exists() for r in plan['candidates'])
        plan=build_plan(root);result=apply_plan(plan,root/'deleted.jsonl')
        assert result['deleted_count']==3
        assert json.loads(pointer.read_text())['status']=='pruned_superseded'
        assert all((root/r['path']).exists() for r in plan['retained']+plan['untouched'])
        assert not any((root/r['path']).exists() for r in plan['candidates'])
    print('清理临时目录检查通过：final/24948/未收官文件保留，变更清单拒绝，旧中间权重删除。')


if __name__=='__main__':main()
