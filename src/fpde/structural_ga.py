# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# The search for fractional terms and orders by scoring forward solutions follows
# J. Im, F. P. J. de Barros, S. Masri, M. Sahimi and R. M. Ziff (2023), Phys. Rev. E
# 107, L013301. https://doi.org/10.1103/PhysRevE.107.L013301
# GPfSI reference implementation by Jinwoo Im: https://github.com/Jinwoousc/GPSI
# This study uses a poroelastic term library, with the population updated by the
# tournament selection, crossover and mutation routines below.

from __future__ import annotations
import copy, math
from pathlib import Path
import numpy as np
from runtime import load_json,save_json
from candidates import mutate,cross,active,parse,build,ORDER_KEYS


def initialize_population(base, size, seed, wide=True, *, baseline):
    rng = np.random.default_rng(seed)
    pop = [copy.deepcopy(base)]
    if size > 1:
        pop.append(copy.deepcopy(baseline))
    while len(pop)<size:
        c=copy.deepcopy(base)
        if wide and len(pop)%4==0:
            p=parse(c);key=ORDER_KEYS[(len(pop)//4-1)%len(ORDER_KEYS)]
            p[key]=float(rng.uniform(.1,3.))
            c=build(p,active(c))
        else:
            for _ in range(1+int(rng.integers(0,3))):
                c=mutate(c,rng,structural=True,scale=1.0 if wide else .45)
        pop.append(c)
    return pop


def _tournament(pool,rng,k=3):
    ids=rng.integers(0,len(pool),k)
    return min((pool[int(i)] for i in ids),key=lambda r:r['loss'])['candidate']


def next_population(records,best,size,rng,structural_rate=.55,scale=.8):
    pool=sorted((r for r in records if r['success']),key=lambda r:r['loss'])
    if not pool:raise RuntimeError('All structural-GA candidates failed')
    elite=max(2,int(math.ceil(size*.10)))
    result=[copy.deepcopy(pool[i%len(pool)]['candidate']) for i in range(elite)]

    result[0]=copy.deepcopy(best['candidate'])
    while len(result)<size:
        a=_tournament(pool,rng);b=_tournament(pool,rng)
        if rng.random()<.80:
            ca,cb=cross(a,b,rng,structural=True)
        else:
            ca,cb=copy.deepcopy(a),copy.deepcopy(b)
        for c in (ca,cb):
            if rng.random()<.90:
                c=mutate(c,rng,structural=(rng.random()<structural_rate),scale=scale)
            result.append(c)
            if len(result)>=size:break
    return result


def evolve(cfg,ev,label,initial,generations,substeps,seed,structural_rate=.55,scale=.8):
    root=Path(cfg['outdir']);path=root/'.states'/(label+'.json');done=root/(label+'_DONE.json')
    if done.exists():return load_json(done)
    rng=np.random.default_rng(seed)
    state=load_json(path) if path.exists() else dict(generation=1,population=copy.deepcopy(initial),history=[],best=None,rng=rng.bit_generator.state)
    rng.bit_generator.state=state['rng']
    def save():state['rng']=rng.bit_generator.state;save_json(path,state)
    save()
    while True:
        gen=state['generation']
        recs=sorted(ev.evaluate(state['population'],substeps,f'{label}/generation{gen}'),key=lambda r:r['loss'])
        good=[r for r in recs if r['success']]
        if not good:raise RuntimeError(label+': all candidates failed; inspect evaluation cache')
        if state['best'] is None or good[0]['loss']<state['best']['loss']:state['best']=good[0]
        best={**state['best'],'generation':gen,'phase':label,'training_only':True,'absolute_state':True,
              'derivative_support_fixed':False,'active_support':list(active(state['best']['candidate']))}
        save_json(root/(label+'_BEST.json'),best)
        if len(state['history'])<gen:
            state['history'].append(dict(generation=gen,loss=best['loss'],
                active_support=list(active(best['candidate'])),initial_error=best['initial_error'],**best['metrics']))
        save_json(root/(label+'_HISTORY.json'),state['history']);save()
        if gen>=generations:
            final={**best,'generations':gen,'stop_reason':'declared_structural_GA_budget'}
            save_json(done,final);return final
        local_scale=scale
        if len(state['history'])>=6 and state['history'][-6]['loss']-best['loss']<1e-5:local_scale*=1.5
        state['population']=next_population(recs,state['best'],len(state['population']),rng,structural_rate,local_scale)
        state['generation']+=1;save()
