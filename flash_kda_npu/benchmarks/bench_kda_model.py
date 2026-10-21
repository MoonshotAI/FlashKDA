"""Pinned unsigned-sigmoid ComplexKDA model evaluation on Ascend.

Only the core recurrence differs across FLA/FlashKDA runs. The checkpoint's
projection, convolution, normalization, MLP and LM head implementations stay
shared. Decode always uses FLA. This harness is inference-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import types

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from flash_kda import fwd
from flash_kda_npu import clear_auto_graph_cache


class CoreBackend:
    def __init__(self):
        self.mode = 'fla'
        self.counts = {}
        self.capture = False
        self.captured = []
        self.offsets = None

    def select(self, mode):
        clear_auto_graph_cache()
        self.mode = mode
        os.environ['KDA_AUTO_GRAPH'] = '1' if mode == 'auto' else '0'
        self.counts = {}

    def run(self, layer, q, k, v, g, beta, initial, use_cache, cu):
        decode = q.shape[1] == 1 and initial is not None
        backend = 'fla_decode' if decode else self.mode
        self.counts[backend] = self.counts.get(backend, 0) + 1
        args = dict(q=q, k=k, v=v, g=g, beta=beta, A_log=layer.A_log,
                    dt_bias=layer.dt_bias, scale=layer.head_k_dim**-.5,
                    initial_state=initial, output_final_state=bool(use_cache),
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, lower_bound=layer.lower_bound,
                    state_v_first=True, cu_seqlens=cu)
        if backend in ('fla', 'fla_decode'):
            if decode:
                out, state = fused_recurrent_kda(**args)
            else:
                out, state = chunk_kda(**args, safe_gate=True)
        else:
            out = torch.empty_like(q)
            sequences = q.shape[0] if cu is None else cu.numel()-1
            state = (torch.empty((sequences, q.shape[2], 128, 128),
                                 device=q.device, dtype=torch.float32) if use_cache else None)
            fwd(q, k, v, g, beta, layer.head_k_dim**-.5, out,
                layer.A_log, layer.dt_bias.view(q.shape[2],128), layer.lower_bound,
                initial_state=initial, final_state=state, cu_seqlens=cu)
        if self.capture:
            self.captured.append((out.detach().float().cpu(),
                                  None if state is None else state.detach().float().cpu()))
        return out, state


def install_adapter(model, backend):
    """Bridge unsigned raw-gate inputs; reject unsupported checkpoint variants."""
    def forward(layer, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError('use packed cu_seqlens rather than padded attention masks')
        cu = kwargs.get('cu_seqlens')
        last = None if past_key_values is None else past_key_values.get(layer.layer_idx)
        conv_state = (None, None, None) if last is None else last['conv_state']
        def convolve(conv, x, state):
            if cu is None:
                return conv(x=x, cache=state, output_final_state=use_cache)
            # Keep checkpoint convolutions on both paths; process boundaries
            # separately because the portable Conv1d does not support packing.
            if backend.offsets is None:
                raise ValueError('packed execution requires CPU offsets from the harness')
            values, states = [], []
            for i,(a,b) in enumerate(zip(backend.offsets,backend.offsets[1:])):
                y,s = conv(x=x[:,a:b], cache=None if state is None else state[i:i+1],
                           output_final_state=use_cache)
                values.append(y)
                if s is not None: states.append(s)
            return torch.cat(values,1), torch.cat(states,0) if states else None
        q,cq = convolve(layer.q_conv1d,layer.q_proj(hidden_states),conv_state[0])
        k,ck = convolve(layer.k_conv1d,layer.k_proj(hidden_states),conv_state[1])
        v,cv = convolve(layer.v_conv1d,layer.v_proj(hidden_states),conv_state[2])
        g = layer.f_proj(hidden_states)
        beta = layer.b_proj(hidden_states).contiguous()
        q,k,v,g = [x.reshape(*x.shape[:-1],-1,128).contiguous() for x in (q,k,v,g)]
        initial = None if last is None else last['recurrent_state']
        out,state = backend.run(layer,q,k,v,g,beta,initial,use_cache,cu)
        if use_cache and past_key_values is not None:
            past_key_values.update_recurrent(layer.layer_idx,state,(cq,ck,cv),True,q.shape[1])
        gate = layer.g_proj(hidden_states).reshape(*out.shape)
        out = layer.o_norm(out,gate).reshape(*hidden_states.shape[:-1],-1)
        return layer.o_proj(out),None,past_key_values

    layers = []
    for block in model.model.layers:
        layer = block.attn
        assert layer.gate == 'sigmoid' and layer.lower_bound == -5.
        assert not layer.allow_neg_eigval and layer.use_short_conv
        assert layer.head_k_dim == layer.head_v_dim == 128
        assert layer.num_heads == layer.num_v_heads
        # Native public ABI requires FP32 gate parameters. These casts preserve
        # the loaded checkpoint values and are identical for all timed paths.
        layer.A_log.data = layer.A_log.data.float()
        layer.dt_bias.data = layer.dt_bias.data.float()
        layer._original_forward = layer.forward
        layer.forward = types.MethodType(forward,layer)
        layers.append(layer)
    return layers


def rel(actual, expected):
    return float((actual-expected).norm()/expected.norm().clamp_min(1e-20))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='/data/models/kda-sigmoid-1.3B-100B')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['smoke','correctness','benchmark','all'],default='all')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--decode-tokens',type=int,default=128)
    parser.add_argument('--batches',default='1,4,8')
    parser.add_argument('--lengths',default='128,512,2048,4096')
    args=parser.parse_args()
    torch.manual_seed(42)
    torch.npu.set_device(0)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    results={'settings':vars(args),'device':torch.npu.get_device_name(0)}
    def save(): output.write_text(json.dumps(results,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,
                 local_files_only=True,dtype=torch.bfloat16).eval().to('npu:0')
    module=__import__(type(model).__module__,fromlist=['ComplexKDACache'])
    backend=CoreBackend()
    # Cache baseline original bound methods for an independent portable check.
    layers=install_adapter(model,backend)
    results['parameters']=sum(p.numel() for p in model.parameters())
    results['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results['adapter_layers']=len(layers)
    text=('</s>The development of efficient language models requires careful measurement. '
          'A fair experiment keeps the input and mathematical operations identical. ')*5000
    corpus=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids
    def tokens(batch,length):
        assert corpus.shape[1]>=length
        return corpus[:,:length].repeat(batch,1).to('npu:0')
    def call(ids,use_cache=True,cu=None,all_logits=False,cache=None):
        return model(input_ids=ids,use_cache=use_cache,past_key_values=cache,
                     logits_to_keep=0 if all_logits else 1,cu_seqlens=cu)
    def timed(fn):
        torch.npu.synchronize();start=time.perf_counter();value=fn()
        torch.npu.synchronize();return value,(time.perf_counter()-start)*1000
    def sample(fn):
        for _ in range(args.warmup):fn()
        torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
        times=[]
        for _ in range(args.repeats):
            value,ms=timed(fn);times.append(ms);del value
        return {'median_ms':statistics.median(times),'min_ms':min(times),
                'max_ms':max(times),'samples_ms':times,
                'peak_allocated_GB':torch.npu.max_memory_allocated()/1e9}

    with torch.inference_mode():
        ids=tokens(1,128)
        value,ms=timed(lambda:call(ids))
        results['first_fla_forward_ms']=ms
        assert value.logits.isfinite().all()
        results['smoke']={'logits_shape':list(value.logits.shape),'counts':dict(backend.counts)}
        del value
        save();print('SMOKE',results['smoke'],flush=True)
        if args.stage=='smoke':return
        if args.stage in ('all','correctness'):
            correctness={}
            captured={}
            logits={}
            losses={}
            for mode in ('fla','native'):
                backend.select(mode);backend.capture=True;backend.captured=[]
                value=call(ids,all_logits=True)
                captured[mode]=backend.captured
                logits[mode]=value.logits.float().cpu()
                losses[mode]=torch.nn.functional.cross_entropy(
                    value.logits[:,:-1].float().reshape(-1,model.config.vocab_size),
                    ids[:,1:].reshape(-1)).item()
                backend.capture=False;backend.captured=[]
                del value
            layer_errors=[{'layer':i,'output_relative_l2':rel(n[0],f[0]),
                'state_relative_l2':rel(n[1],f[1])}
                for i,(f,n) in enumerate(zip(captured['fla'],captured['native']))]
            correctness.update(layer_errors=layer_errors,loss=losses,
                logits_relative_l2=rel(logits['native'],logits['fla']),
                argmax_agreement=float((logits['native'].argmax(-1)==logits['fla'].argmax(-1)).float().mean()))
            # Original portable model at a shorter length remains independent
            # of both kernel adapters. It uses K-first state layout.
            backend.select('fla')
            small=tokens(1,16)
            adapted=call(small,all_logits=True).logits.float().cpu()
            for layer in layers:layer.forward=layer._original_forward
            portable=call(small,all_logits=True).logits.float().cpu()
            install_adapter(model,backend)
            correctness['fla_vs_portable_logits_relative_l2']=rel(adapted,portable)
            correctness['generation']={}
            for mode in ('fla','native'):
                backend.select(mode)
                value=call(ids);generated=[]
                for i in range(32):
                    token=value.logits[:,-1].argmax(-1,keepdim=True)
                    generated.append(token.cpu())
                    if i<31:value=call(token,cache=value.past_key_values)
                generated=torch.cat(generated,1)
                correctness['generation'][mode]={'ids':generated.tolist(),
                    'text':tokenizer.decode(generated[0]),'counts':dict(backend.counts)}
                del value
            correctness['generation_ids_equal']=(correctness['generation']['fla']['ids']==correctness['generation']['native']['ids'])
            results['correctness']=correctness;save()
            print('CORRECTNESS',json.dumps({k:v for k,v in correctness.items() if k not in ('layer_errors','generation')}),flush=True)
            assert correctness['logits_relative_l2']<.03,correctness
            assert correctness['fla_vs_portable_logits_relative_l2']<.03,correctness
            assert max(e['output_relative_l2'] for e in layer_errors)<.05
            if args.stage=='correctness':return
        results['prefill']=[]
        for batch in map(int,args.batches.split(',')):
            for length in map(int,args.lengths.split(',')):
                ids=tokens(batch,length)
                # Alternate FLA/native order to expose order/thermal effects.
                modes=('fla','native','auto') if length%1024 else ('native','fla','auto')
                for mode in modes:
                    backend.select(mode)
                    row={'batch':batch,'length':length,'mode':mode}
                    row.update(sample(lambda:call(ids)))
                    row['tokens_per_s']=batch*length*1000/row['median_ms']
                    row['calls']=dict(backend.counts)
                    if mode=='auto':
                        from flash_kda_npu.auto_graph import _LOCAL
                        cache=getattr(_LOCAL,'cache',{})
                        row['auto_cached_entries']=len(cache)
                        row['auto_captured_entries']=sum(e['graph'] is not None for e in cache.values())
                    results['prefill'].append(row);save();print('PREFILL',json.dumps(row),flush=True)
                del ids
        # One mixed-length packed batch; no padding tokens count as useful work.
        lengths=[128,512,2048,4096];offsets=[0]
        for length in lengths:offsets.append(offsets[-1]+length)
        ids=torch.cat([tokens(1,n) for n in lengths],1)
        cu=torch.tensor(offsets,dtype=torch.int64,device='npu:0')
        backend.offsets=offsets
        results['packed']=[]
        packed_logits={}
        for mode in ('fla','native'):
            backend.select(mode)
            packed_logits[mode]=call(ids,use_cache=False,cu=cu,all_logits=True).logits.float().cpu()
            row={'lengths':lengths,'mode':mode}
            row.update(sample(lambda:call(ids,use_cache=False,cu=cu)))
            results['packed'].append(row);save();print('PACKED',json.dumps(row),flush=True)
        results['packed_logits_relative_l2']=rel(packed_logits['native'],packed_logits['fla'])
        results['generation']=[]
        for batch,length in ((1,128),(1,2048),(1,4096),(4,512),(8,512)):
            if batch not in list(map(int,args.batches.split(','))):continue
            ids=tokens(batch,length)
            for mode in ('fla','native','auto'):
                backend.select(mode)
                def generate():
                    def first_token():
                        value=call(ids)
                        return value,value.logits[:,-1].argmax(-1,keepdim=True)
                    (value,token),prefill_ms=timed(first_token)
                    cache=value.past_key_values
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(args.decode_tokens-1):
                        value=call(token,cache=cache)
                        token=value.logits[:,-1].argmax(-1,keepdim=True)
                        cache=value.past_key_values
                    torch.npu.synchronize();decode_ms=(time.perf_counter()-start)*1000
                    return prefill_ms,decode_ms
                generate() # Warm decode kernels; fixed length without EOS early exit.
                samples=[generate() for _ in range(args.repeats)]
                row={'batch':batch,'length':length,'mode':mode,'generated_tokens':args.decode_tokens,
                     'ttft_ms':statistics.median(x[0] for x in samples),
                     'decode_ms':statistics.median(x[1] for x in samples),
                     'total_ms':statistics.median(sum(x) for x in samples),
                     'decode_tokens_per_s':batch*(args.decode_tokens-1)*1000/statistics.median(x[1] for x in samples),
                     'samples_ms':samples,'calls':dict(backend.counts)}
                results['generation'].append(row);save();print('GENERATION',json.dumps(row),flush=True)
        if args.profile:
            results['profiles']={}
            ids=tokens(1,2048)
            for mode in ('fla','native'):
                backend.select(mode)
                call(ids);torch.npu.synchronize()
                trace=output.with_name(output.stem+'_'+mode+'_trace.json')
                with torch_npu.profiler.profile(activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU]) as prof:
                    call(ids);torch.npu.synchronize()
                prof.export_chrome_trace(str(trace))
                results['profiles'][mode]=str(trace)
                save()
        results['complete']=True;save()

if __name__=='__main__':main()
