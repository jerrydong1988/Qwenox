"""Reproducible teacher-forced comparison of serial vs compensated HGN math.

Uses a bounded WikiText-2 test slice plus original Chinese/code fixtures. It is
not the full WikiText benchmark and does not measure quantization vs BF16.
Download the explicit Salesforce dataset rows separately; no implicit network.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def encode(cli, tokenizer, text):
    r = subprocess.run([str(cli.resolve()), '--dump-req', '--tokenizer', tokenizer,
                        json.dumps({'prompt': text}, ensure_ascii=True)], check=True, capture_output=True)
    p = r.stdout.decode().split(); start=4+int(p[3])
    return list(map(int,p[start+1:start+1+int(p[start])]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine',type=Path,required=True);ap.add_argument('--model',required=True)
    ap.add_argument('--tokenizer',required=True);ap.add_argument('--wiki-rows',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cli',type=Path,default=Path('build/eng-cli-win.exe'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    wiki=json.loads(args.wiki_rows.read_text(encoding='utf-8'))
    text=''.join(r['row']['text'] for r in wiki['rows'])
    # Clip at a character boundary well below the Windows command-line limit.
    wiki_ids=encode(args.cli,args.tokenizer,text[:14000])[:2049]
    chinese='''二分查找适用于已经排序的数据。每次比较中间位置的元素，如果目标较小，就在左半区间继续查找；否则在右半区间继续查找。
数据库事务通常具有原子性、一致性、隔离性和持久性。事务失败时，系统应当撤销尚未提交的修改。隔离级别会影响并发事务能够观察到哪些数据。
一辆汽车以每小时六十公里的速度行驶两小时，路程是一百二十公里。如果停留半小时后再以每小时八十公里行驶一小时，总路程为二百公里，总用时为三点五小时。
图书馆借阅记录包括读者编号、书籍编号、借出日期和归还日期。统计逾期图书时，需要排除已经归还的记录，并根据约定的借阅期限判断是否逾期。
在操作系统中，进程拥有独立的地址空间，线程共享所属进程的资源。互斥锁用于保护共享数据；条件变量用于等待状态变化。程序应避免持锁等待不受控制的外部操作。
机器学习模型的参数决定了输入到输出的映射。量化使用较少的位数表示参数，可以减少存储和带宽开销，但不等于完全保留原始浮点数值。评估时需要使用相同输入和明确的参考实现。
排序算法的时间复杂度描述输入规模增长时计算量的变化。归并排序把序列分成两部分，递归排序后合并；快速排序选择枢轴并划分区间。稳定性表示相同键的元素能否保持原来的相对顺序。
写入文件时，先写临时文件，再在同一文件系统中原子替换目标文件，可以降低中途失败造成文件损坏的风险。程序仍需要检查返回值，并保留可验证的恢复路径。
'''
    code=Path('tools/qwen27_quality.py').read_text(encoding='utf-8').split('def main():')[0]+'\n'+Path('tools/qwen27_compare.py').read_text(encoding='utf-8')
    cases=[('wikitext2-test-prefix',wiki_ids),('chinese-original',encode(args.cli,args.tokenizer,chinese)[:513]),('python-source',encode(args.cli,args.tokenizer,code)[:513])]
    result={'method':'teacher forced, same HGN, serial FP32 vs compensated prefill512',
            'wiki_source':'https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=test&offset=0&length=100',
            'wiki_rows_sha256':hashlib.sha256(args.wiki_rows.read_bytes()).hexdigest(),
            'engine_sha256':hashlib.sha256(args.engine.read_bytes()).hexdigest(),'cases':[]}
    for name,ids in cases:
        assert len(ids)>100
        tokenfile=args.output/(name+'.tokens');tokenfile.write_text(','.join(map(str,ids)),encoding='ascii')
        scores=[]
        for mode in ['serial','compensated']:
            outfile=args.output/(name+'-'+mode+'.txt');errfile=args.output/(name+'-'+mode+'.log')
            command=[str(args.engine.resolve()),args.model,'--tokens-file',str(tokenfile),'--gen','0','--maxctx','4096','--qwen27-score','--qwen27-no-mtp','--qwen27-no-dflash']
            if mode=='serial':command+=['--qwen27-serial-prefill']
            with outfile.open('w',encoding='utf-8') as out,errfile.open('w',encoding='utf-8') as err:
                subprocess.run(command,stdout=out,stderr=err,check=True)
            rows=[line.split() for line in outfile.read_text().splitlines() if line.startswith('score ')]
            assert len(rows)==len(ids)-1 and all(int(r[1])==i and int(r[2])==ids[i+1] for i,r in enumerate(rows))
            scores.append([(int(r[3]),float(r[4])) for r in rows])
            print(json.dumps({'case':name,'mode':mode,'tokens':len(rows),'mean_nll':-sum(r[1] for r in scores[-1])/len(rows)}),flush=True)
        a,b=scores;d=[abs(x[1]-y[1]) for x,y in zip(a,b)];n=len(a)
        nll=[-sum(r[1] for r in s)/n for s in scores]
        item={'case':name,'tokens':n,'tokens_sha256':hashlib.sha256(tokenfile.read_bytes()).hexdigest(),
              'serial_nll':nll[0],'compensated_nll':nll[1],
              'serial_ppl':math.exp(nll[0]),'compensated_ppl':math.exp(nll[1]),
              'nll_delta':nll[1]-nll[0],'logprob_max_abs':max(d),'logprob_mean_abs':sum(d)/n,
              'top1_matches':sum(x[0]==y[0] for x,y in zip(a,b))}
        assert all(math.isfinite(v) for v in nll) and abs(item['nll_delta'])<1e-3 and item['logprob_max_abs']<.02,item
        result['cases'].append(item)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)


if __name__=='__main__':main()
