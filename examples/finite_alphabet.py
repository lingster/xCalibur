"""Exact finite-alphabet contracts from the kernel-construction studies.

Run with Python's standard library only:
    python examples/finite_alphabet.py --out build/finite_alphabet

Checks all paired FP4/FP8-scale cases and bounded BF16-to-FP8 decompositions.
Writes scalar fixtures and a contract report. No GPU, model or network is used;
scalar equality does not prove matrix accumulation or a hardware speedup.
"""
import argparse
import bisect
import json
import math
from pathlib import Path
import struct

MAG=(0.,.5,1.,1.5,2.,3.,4.,6.)


def fp4(code):
    return math.copysign(MAG[code&7],-1. if code&8 else 1.)


def fp8(code):
    e,m=(code>>3)&15,code&7
    x=m/512 if e==0 else math.nan if (e,m)==(15,7) else math.ldexp(8+m,e-10)
    return math.copysign(x,-1. if code&128 else 1.)


def bf16(code):
    return struct.unpack('<f',struct.pack('<I',code<<16))[0]


def bits(x):
    word=struct.unpack('<I',struct.pack('<f',x))[0]
    if math.isfinite(x):assert word&65535==0,'exact BF16 value required'
    return word>>16


def pair(a,b):return bits(a)|(bits(b)<<16)


def tables():
    normal=[];subnormal=[]
    for mantissa in range(8):
        for b in range(8):
            for a in range(8):
                normal.append(pair(MAG[a]*(1+mantissa/8)/64,MAG[b]*(1+mantissa/8)/64))
                subnormal.append(pair(MAG[a]*mantissa/512,MAG[b]*mantissa/512))
    return normal,subnormal


def decode(byte,scale,normal,subnormal):
    e,m=(scale>>3)&15,scale&7
    if e==15 and m==7:return 0x7fc07fc0
    i=(byte&7)|((byte>>1)&56)|(m<<6)
    value=(normal if e else subnormal)[i]
    if e:
        mask=(65535 if byte&7 else 0)|(0xffff0000 if byte&112 else 0)
        value+=((e-1)*0x00800080)&mask
    signs=((byte&8)<<12)|((byte&128)<<24)
    if scale&128:signs^=0x80008000
    return value^signs


def lut_contract(out):
    normal,subnormal=tables();records=[];finite=0;nan=0
    for scale in range(256):
        for byte in range(256):
            actual=decode(byte,scale,normal,subnormal)
            if math.isnan(fp8(scale)):
                assert actual==0x7fc07fc0;nan+=1
            else:
                expected=pair(fp4(byte&15)*fp8(scale),fp4(byte>>4)*fp8(scale))
                assert actual==expected,(byte,scale,hex(actual),hex(expected))
                finite+=1
            records.append((byte,scale,actual))
    for byte in range(256):
        tiny=((byte&7)<<6)|((byte&8)<<12)|((byte&112)<<18)|((byte&128)<<24)
        assert tiny==pair(math.ldexp(fp4(byte&15),-126),math.ldexp(fp4(byte>>4),-126))
    (out/'pair_tables.bin').write_bytes(struct.pack('<1024I',*(normal+subnormal)))
    (out/'pair_cases.bin').write_bytes(b''.join(struct.pack('<III',*r) for r in records))
    return {'finite_pair_scale_cases':finite,'nan_pair_scale_cases':nan,'raw_byte_cases':256,
            'raw_pair_table_bytes':1024,'sign_factored_raw_pair_bytes':256,
            'normal_scale_pair_table_bytes':2048,'all_finite_scale_pair_table_bytes':4096,
            'tiny_embedding':'BF16bits(q * 2^-126) = ((q & 7) << 6) | ((q & 8) << 12)',
            'scope':'exact values and signed zeros; NaNs checked by class; no latency claim'}


def fp8_contract(out):
    alphabet={fp8(c):c for c in range(256) if math.isfinite(fp8(c))}
    values=sorted(alphabet);coverage=[]
    for shift in range(9):
        attainable={a+math.ldexp(b,-shift) for a in values for b in values}
        bins=[]
        for exponent in range(-15,10):
            xs=[math.ldexp(128+m,exponent-7) for m in range(128)]
            exact=sum(x in attainable for x in xs)
            bins.append({'exponent':exponent,'exact_of_128':exact})
        coverage.append({'residual_shift':shift,'attainable_values':len(attainable),
                         'complete_binades':[r['exponent'] for r in bins if r['exact_of_128']==128],
                         'binades':bins})
    def nearest(x):
        i=bisect.bisect_left(values,x)
        choices=values[max(0,i-1):min(len(values),i+1)]
        return min(choices,key=lambda y:(abs(x-y),alphabet[y]&1))
    constructions=[]
    for shift,low_exp,method in ((4,-6,'truncate_bf16_low4'),(5,-7,'nearest_fp8')):
        records=[];scalar_cases=0
        for sign in (0,0x8000):
            for exponent in range(low_exp,8):
                for mantissa in range(128):
                    code=sign|((exponent+127)<<7)|mantissa
                    x=bf16(code);hi=bf16(code&~15) if shift==4 else nearest(x)
                    lo=math.ldexp(x-hi,shift)
                    assert hi in alphabet and lo in alphabet
                    assert hi+math.ldexp(lo,-shift)==x
                    scalar_cases+=1
                    for q in range(16):
                        a=alphabet[fp4(q)];h=alphabet[hi];l=alphabet[lo]
                        records.append((a*0x01010101,h*0x01010101,l*0x01010101,fp4(q)*x))
        name=f'fp8_cases_r{shift}.bin'
        (out/name).write_bytes(b''.join(struct.pack('<IIIf',*r) for r in records))
        constructions.append({'residual_shift':shift,'high_method':method,'scalar_cases':scalar_cases,
                              'native_dot_cases':len(records),'input_file':name,'exponent_window':[low_exp,7]})
    return {'component_format':'E4M3','residual_shifts_searched':list(range(9)),
            'coverage':coverage,'constructions':constructions,
            'split':'x = hi + 2^-r lo; lo = 2^r (x-hi)',
            'limitations':['common block scaling must place inputs in the covered window',
                           'no representation claim for arbitrary BF16 exponent spread',
                           'scalar reconstruction does not prove matrix accumulation equivalence',
                           'NVFP4 K16 block scales must remain inside their own K reduction'],
            'native_status':'not executed by this CPU example'}


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'lut':lut_contract(out),'fp8':fp8_contract(out),'hardware_executed':False}
    (out/'contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'lut':report['lut'],'fp8_constructions':report['fp8']['constructions'],
                      'complete_binades':[{k:r[k] for k in ('residual_shift','complete_binades')} for r in report['fp8']['coverage']]}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='build/finite_alphabet')
    run(p.parse_args().out)
