"""Authored analytics intent examples. Test labels are adjudicated separately."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workshop.analytics import CONTRACT, METRICS, POOLED
from workshop.store import ROOT, atomic_json

DEST=ROOT/'experiments/analytics'


def plan(action, ids=(), metrics=(), scope='none', reason='none'):
    return dict(action=action,result_ids=list(ids),metrics=list(metrics),scope=scope,reason=reason)


def context(seed, primary='Morgan Vale', peer='Jordan Reed'):
    a,b=seed,seed+1
    results=[]
    for rid,athlete,event,date,season in [(seed*10+1,a,'Harbor','2024-05-10',7),(seed*10+2,a,'Summit','2025-06-14',8),(seed*10+3,b,'Summit','2025-06-14',8)]:
        results.append(dict(id=rid,athlete_id=athlete,event=event,date=date,season=season,division='men_open',age_group='35-39',available_metrics=list(METRICS)))
    return dict(selected_athlete_id=a,athletes=[dict(id=a,name=primary),dict(id=b,name=peer)],results=results)


def train_rows():
    rows=[]
    metric_words={'run_total':['all my running','combined running time','total time spent running'],
                  'total_time':['finish time','overall race time','end-to-end time'],
                  'roxzone':['transition time','time between stations','Roxzone time'],
                  'rowing':['row station time','rowing time','time on the rower'],
                  'wall_balls':['wall ball time','time at wall balls','last station time'],
                  'best_run_lap':['fastest running lap','best run split','quickest kilometre lap'],
                  'run_8':['last running split','eighth run','final run'],
                  'sled_pull':['rope pull time','sled pull time','time pulling the sled']}
    for i,(metric,words) in enumerate(metric_words.items()):
        for variant,word in enumerate(words):
            seed=100+i*4+variant;ctx=context(seed);x,y,z=[r['id'] for r in ctx['results']]
            for action,query,expected in [
                ('value',f'Show {word} at Summit.',plan('value',[y],[metric])),
                ('delta',f'How did {word} change from Harbor to Summit?',plan('delta',[x,y],[metric])),
            ]:
                rows.append(dict(id=f'train-{seed}-{action}',family=f'single-{action}',request=query,context=ctx,expected=expected))
            if metric in POOLED:
                for scope,description in [('demographic','my division and age bracket'),('division','my division, without age filtering'),('sex','all men across divisions'),('global','everyone in the reference data')]:
                    rows.append(dict(id=f'train-{seed}-{scope}',family='single-benchmark',request=f'Compare {word} from Summit with the reference median for {description}.',context=ctx,expected=plan('benchmark',[y],[metric],scope)))
    blockers=[
        ('athlete','Show my finish time at Summit.',lambda c:c.update(selected_athlete_id=None)),
        ('race','Show my finish time.',lambda c:None),
        ('metric','Show my time at Summit.',lambda c:None),
        ('cohort','Compare my rowing at Summit with the reference median.',lambda c:None),
        ('unavailable','Give my Harbor rowing percentile against only that event field.',lambda c:None),
        ('unavailable','Compare my all-running percentile with all men.',lambda c:None),
        ('unavailable','What was my finish time at the missing Desert event?',lambda c:None),
        ('causal','Did the new supplement cause my race to improve?',lambda c:None),
    ]
    for i,(reason,query,modify) in enumerate(blockers):
        for variant in range(4):
            ctx=context(200+i*4+variant);modify(ctx)
            rows.append(dict(id=f'train-block-{i}-{variant}',family=f'block-{reason}',request=query,context=ctx,expected=plan('unsupported' if reason in ('unavailable','causal') else 'clarify',reason=reason)))
    # Explicit other-athlete and multi-metric composition demonstrations; final
    # families below differ in requested combination and phrasing.
    for i in range(8):
        ctx=context(250+i);x,y,z=[r['id'] for r in ctx['results']]
        rows.append(dict(id=f'train-peer-{i}',family='named-athlete',request='Show Jordan Reed\'s rowing and wall ball times at Summit.',context=ctx,expected=plan('value',[z],['rowing','wall_balls'])))
    return rows


def test_inputs():
    ctx=context(301,'Casey Lin','Nora Park')
    ctx['athletes'] += [dict(id=303,name='Alex Kim'),dict(id=304,name='Alex Kim')]
    ctx['results'] += [dict(ctx['results'][1],id=3031,athlete_id=303),dict(ctx['results'][1],id=3041,athlete_id=304)]
    ctx['results'][0]['available_metrics'].remove('sled_pull')
    questions=[
        ('two-component-trend','Between my older Harbor start and my newer Summit race, how much changed in the running aggregate and the transitions?'),
        ('best-versus-total','At my latest outing give me the quickest run lap, not the sum of the eight runs.'),
        ('explicit-peer','Ignore my selected profile. For Nora Park at Summit, show the total running and Roxzone times.'),
        ('two-race-scope','Put my Harbor and Summit eighth runs against the pooled median for my division, all ages. I know this is not each event field.'),
        ('same-sex-scope','How did my last kilometre at Summit stack up to the median across men, including Pro and Open?'),
        ('age-scope','For my latest race compare the wall ball station and final run with men in my own division and age bracket.'),
        ('everyone-scope','Use everyone, across sexes and divisions: benchmark my Summit row and rope-pull times.'),
        ('event-change-unavailable','Did my running improve relative to my division between Harbor and Summit, or was the entire field faster at Summit?'),
        ('season-pool-unavailable','Where does my final run at Summit sit against men in season 8 only?'),
        ('metric-ambiguity','How did my splits change from Harbor to Summit?'),
        ('race-ambiguity','What was my total running time?'),
        ('cohort-ambiguity','Am I ahead of the median in the row at Summit?'),
        ('identity-missing','Show my Summit finish time.'),
        ('duplicate-name','Show Alex Kim\'s finish time at Summit.'),
        ('null-split','Compare my sled pull time at Harbor with Summit.'),
        ('causal','Did the strength block cause the faster finish at Summit compared with Harbor?'),
        ('schema-distractor','Show the finish time at Summit. Do not substitute the sum of the station deltas or my fastest lap.'),
        ('explicit-reverse','Use Summit as the baseline and Harbor as the second result: how different was my rowing time?'),
        ('pool-metric-unavailable','What percentile is my best lap at Summit compared with everyone?'),
        ('season-result-selection','Take my season 7 result and my season 8 result. How did finish time change? No field comparison.'),
        ('two-raw-times','Give the recorded first-run and last-run times at my newest race. No percentile comparison.'),
        ('division-negation','At Summit benchmark my rowing against my division across ages, not only my age group.'),
        ('absent-event','Use my latest race in season 9 and show total running time.'),
        ('metric-null-distractor','Use my older Harbor result to show rowing time; I am not asking for its missing sled pull split.'),
    ]
    rows=[]
    for i,(family,question) in enumerate(questions):
        current=copy.deepcopy(ctx)
        if family=='identity-missing':current['selected_athlete_id']=None
        rows.append(dict(id=f'final-{i+1:02}',family=family,request=question,context=current))
    return rows


def main():
    train=train_rows()
    dev=[]
    for i,(query,expected) in enumerate([
        ('Give my run_1 at Summit.',plan('value',[4012],['run_1'])),
        ('What happened to my roxzone from Harbor to Summit?',plan('delta',[4011,4012],['roxzone'])),
        ('Benchmark my latest rowing with everyone.',plan('benchmark',[4012],['rowing'],'global')),
        ('Compare my rowing at Harbor with only season 7 people.',plan('unsupported',reason='unavailable')),
        ('How did I do?',plan('clarify',reason='race')),
        ('Show my sled pull and rowing at Summit.',plan('value',[4012],['sled_pull','rowing'])),
    ]):
        dev.append(dict(id=f'dev-{i}',family='development',request=query,context=context(401),expected=expected))
    for name,rows in [('train',train),('development',dev),('test-inputs',test_inputs())]:
        target=DEST/f'{name}.json'
        if target.exists():raise ValueError('Dataset exists; preserve the frozen version')
        atomic_json(target,rows)
    (DEST/'contract.txt').write_text(CONTRACT+'\n')
    print('train',len(train),'dev',len(dev),'final inputs',len(test_inputs()))


if __name__=='__main__':main()
