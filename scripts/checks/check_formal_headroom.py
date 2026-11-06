"""2026-10-08：6GiB整卡余量、OOM同批重试和完整FP16恢复的短工程检查。"""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.formal_capacity import backward_documents, limit_cuda_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/formal_headroom_check_20261008')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.out_dir
    out.mkdir(parents=True,exist_ok=False)
    with patch('torch.cuda.empty_cache'), patch('torch.cuda.mem_get_info',return_value=(31*2**30,32*2**30)), \
         patch('torch.cuda.memory_reserved',return_value=0), patch('torch.cuda.set_per_process_memory_fraction') as fraction:
        limit = limit_cuda_memory('cuda:0',6)
        assert limit['allocator_limit_bytes'] == 25*2**30
        fraction.assert_called_once_with(25/32,0)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
            self.cfg = SimpleNamespace(max_context_strokes=1024)
            self.fail = True
        def forward(self,batch):
            value = self.weight*torch.rand(())
            if batch == 1 and self.fail:
                self.fail = False
                raise torch.cuda.OutOfMemoryError('fixture')
            return {'content_sum':value,'control_sum':value*0,'task_sums':{0:value}}
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scaler = SimpleNamespace(scale=lambda loss:loss)
    recipe = {'padded_stroke_budget':8192}
    torch.manual_seed(42)
    with patch('src.st.formal_capacity.iter_microbatches',return_value=(0,1)), \
         patch('torch.autocast',side_effect=lambda *a,**k:nullcontext()), patch('torch.cuda.empty_cache'):
        _,_,finite,count = backward_documents(model,optimizer,scaler,(),{},0,recipe,'cpu',
            lambda result,*unused:result['content_sum'],lambda batch,device:batch,('T1',),
            torch.get_rng_state,torch.set_rng_state)
    torch.manual_seed(42)
    expected = torch.rand(())+torch.rand(())
    torch.testing.assert_close(model.weight.grad,expected,rtol=0,atol=0)
    assert finite and count == 2 and recipe['padded_stroke_budget']==7168
    fixture = ROOT/'tmp_smoke/formal_launch_check_20261007'
    common = [sys.executable,'-u','src/st/train_formal.py','--data_manifest',
              str(ROOT/'tmp_smoke/formal_check_20261003/data_args.json'),
              '--resume',str(fixture/'fp16/final.pt'),'--steps','2','--max_seconds','0','--log_every','1',
              '--memory_headroom_gib','.5']
    for name,calibrate in (('calibrated',True),('plain',False)):
        command = common+['--output_dir',str(out/name)]
        if calibrate:
            command.append('--calibrate')
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    a = torch.load(out/'calibrated/final.pt',map_location='cpu',weights_only=False)
    b = torch.load(out/'plain/final.pt',map_location='cpu',weights_only=False)
    assert a['step']==b['step']==2
    for name,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for field,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][key][field],rtol=0,atol=0)
    for key in ('data','rng','counters','scheduler','scaler'):
        if key != 'rng':
            assert a[key]==b[key],key
    report = {'allocator_accounts_for_outside_cuda':True,'six_gib_limit':True,
              'oom_retry_clears_partial_gradient_and_restores_rng':True,
              'fp16_resumed_step':2,'calibration_model_adam_data_wsd_scaler_equal':True,
              'fixture_headroom_gib':.5,'production_headroom_gib':6}
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
