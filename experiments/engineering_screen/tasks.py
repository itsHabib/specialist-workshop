"""Independently synthetic engineering tasks; exact host-only references."""
from fractions import Fraction as F
from itertools import product
from functools import lru_cache
import random

GEOMETRY = '''Implement evaluate(request) using only Python standard library. Request has rings: a list of closed polygonal walks, each a list of [x,y] integer vertices (last-to-first closes it), and rule: "evenodd" or "nonzero". Walks can self-intersect, overlap, retrace edges, contain repeated vertices, or have either orientation. Sum signed winding numbers of ALL walks, not a union of independently filled rings. Filled means total winding odd (evenodd), or nonzero (nonzero). Return {"area": canonical rational string} for exact nonnegative area: integers have no /1, fractions reduced with positive denominator. Boundary has zero area. Empty/degenerate walks contribute zero. At most 60 total vertices with integer coordinates |v|<=1000000000. No floating-point tolerance is adequate. Define evaluate, no stdin handling.'''

LEASES = '''Implement evaluate(request) using Python standard library. Decide whether a concurrent history admits a legal linearization of a one-job leased queue. Return {"linearizable": bool}. Initially generation=0, lease=None, done=False. Sequential operations:
claim(worker,now,ttl): if done or (lease exists and now < deadline), return null with no change; otherwise increment generation, install (worker,generation,now+ttl), return generation.
finish(worker,token,now): return true and set done=True ONLY if not done and lease exists, worker/token match it, and now < deadline; otherwise return false unchanged.
reset(): increment generation, clear lease, done=False; return generation.
Times 'now' are supplied values, NOT invocation/return positions, and need not be monotonic. ttl is positive. Expiry does not itself increment generation. Equality at deadline is expired. All operations atomic.
request.history is <=9 operations, each with id (unique string), start (unique integer), end (unique integer greater than start) OR null, op (claim/finish/reset), args mapping, result. All non-null start/end event positions are distinct. Completed operations must all occur once with matching results. Pending end=null operations may be dropped, or linearized once anywhere after invocation, with unconstrained result. Real-time order: A must precede B if A has non-null end < B.start. Overlapping calls can order either way. Numeric tokens must not compare equal to boolean results. No hidden timers, real clocks or external effects. Define evaluate, no stdin handling.'''

FLOW = '''Implement evaluate(request) using only Python standard library. Find exact minimum COST of a feasible integral circulation with required net flow amount from source to sink. Nodes 0..n-1; request has n, source, sink (distinct), amount >=0, edges list of {u,v,lower,upper,cost} integers, 0<=lower<=upper. Parallel edges, self-loops, negative costs, disconnected components and negative cycles are allowed. For each node, outgoing flow minus incoming flow must be amount at source, -amount at sink, 0 elsewhere. Minimize sum(flow*cost) over ALL edges, including useful circulations disconnected from source/sink. Return {"cost": integer} or {"cost": null} if infeasible. n<=30, edges<=60, upper<=1000000, |cost|<=1000; exact arithmetic. Do not enumerate every assignment of edge flow. Define evaluate, no stdin handling.'''


def integer_text(value):
    # Exact rational outputs can exceed Python's default integer-string limit.
    if value < 0:return '-'+integer_text(-value)
    chunks=[]
    while value >= 10**9:
        value,part=divmod(value,10**9);chunks.append(part)
    return str(value)+''.join(f'{part:09d}' for part in reversed(chunks))


def geometry(q):
    edges=[]; ys=set()
    for ring in q['rings']:
        for p,r in zip(ring,ring[1:]+ring[:1]):
            x,y=map(F,p); u,v=map(F,r);ys.update((y,v))
            if y==v:continue
            a=(u-x)/(v-y);b=x-a*y
            edges.append((min(y,v),max(y,v),a,b,1 if v>y else -1))
    for i,e in enumerate(edges):
        for f in edges[i+1:]:
            if e[2]==f[2]:continue
            y=(f[3]-e[3])/(e[2]-f[2])
            if max(e[0],f[0])<y<min(e[1],f[1]):ys.add(y)
    total=F(0);ys=sorted(ys)
    for lo,hi in zip(ys,ys[1:]):
        mid=(lo+hi)/2;active=sorted((a*mid+b,a,b,d) for l,h,a,b,d in edges if l<mid<h)
        winding=0
        for e,f in zip(active,active[1:]):
            winding+=e[3]
            if (winding%2 if q['rule']=='evenodd' else winding!=0):
                total+=((f[1]-e[1])*(lo+hi)/2+f[2]-e[2])*(hi-lo)
    numerator=integer_text(total.numerator)
    return {'area':numerator if total.denominator==1 else numerator+'/'+integer_text(total.denominator)}


def step(state,operation):
    generation,lease,done=state;args=operation['args'];kind=operation['op']
    if kind=='reset':return (generation+1,None,False),generation+1
    if kind=='claim':
        if done or (lease is not None and args['now']<lease[2]):return state,None
        g=generation+1;return (g,(args['worker'],g,args['now']+args['ttl']),False),g
    ok=not done and lease is not None and (args['worker'],args['token'])==lease[:2] and args['now']<lease[2]
    return ((generation,lease,True) if ok else state),ok


def leases(q):
    ops=q['history'];n=len(ops);required=sum(1<<i for i,o in enumerate(ops) if o['end'] is not None)
    before=[sum(1<<j for j,p in enumerate(ops) if p['end'] is not None and p['end']<o['start']) for o in ops]
    @lru_cache(None)
    def visit(mask,state):
        if mask&required==required:return True
        for i,o in enumerate(ops):
            if mask>>i&1 or before[i]&mask!=before[i]:continue
            nxt,result=step(state,o)
            if o['end'] is not None and (type(result)!=type(o['result']) or result!=o['result']):continue
            if visit(mask|1<<i,nxt):return True
        return False
    return {'linearizable':visit(0,(0,None,False))}


def flow(q):
    """Feasible circulation via augmenting paths; optimize by canceling negative cycles."""
    n=q['n'];edges=q['edges'];values=[e['lower'] for e in edges]
    target=[0]*n;target[q['source']]=q['amount'];target[q['sink']]=-q['amount']
    balance=[0]*n
    for e,v in zip(edges,values):balance[e['u']]+=v;balance[e['v']]-=v
    def residual():
        arcs=[]
        for i,(e,v) in enumerate(zip(edges,values)):
            if v<e['upper']:arcs.append((e['u'],e['v'],e['upper']-v,e['cost'],i,1))
            if v>e['lower']:arcs.append((e['v'],e['u'],v-e['lower'],-e['cost'],i,-1))
        return arcs
    while balance!=target:
        source=next((i for i in range(n) if balance[i]<target[i]),None)
        if source is None:return {'cost':None}
        prev={source:None};queue=[source];arcs=residual();end=None
        for node in queue:
            if balance[node]>target[node]:end=node;break
            for arc in arcs:
                if arc[0]==node and arc[1] not in prev:prev[arc[1]]=arc;queue.append(arc[1])
        if end is None:return {'cost':None}
        path=[];node=end
        while node!=source:path.append(prev[node]);node=prev[node][0]
        delta=min(target[source]-balance[source],balance[end]-target[end],*(a[2] for a in path))
        for a in path:values[a[4]]+=delta*a[5]
        balance[source]+=delta;balance[end]-=delta
    while True:
        arcs=residual();dist=[0]*n;prev=[None]*n;changed=None
        for _ in range(n):
            changed=None
            for a in arcs:
                u,v,cap,cost,_,_=a
                if dist[v]>dist[u]+cost:dist[v]=dist[u]+cost;prev[v]=a;changed=v
            if changed is None:break
        if changed is None:break
        node=changed
        for _ in range(n):node=prev[node][0]
        start=node;cycle=[]
        while True:
            a=prev[node];cycle.append(a);node=a[0]
            if node==start:break
        delta=min(a[2] for a in cycle)
        for a in cycle:values[a[4]]+=delta*a[5]
    return {'cost':sum(v*e['cost'] for e,v in zip(edges,values))}


def brute_flow(q):
    best=None
    for values in product(*(range(e['lower'],e['upper']+1) for e in q['edges'])):
        balance=[0]*q['n'];cost=0
        for e,v in zip(q['edges'],values):balance[e['u']]+=v;balance[e['v']]-=v;cost+=v*e['cost']
        target=[0]*q['n'];target[q['source']]=q['amount'];target[q['sink']]=-q['amount']
        if balance==target and (best is None or cost<best):best=cost
    return {'cost':best}


def cases(seed):
    rng=random.Random(seed);geometry_cases=[];lease_cases=[];flow_cases=[]
    for _ in range(16):
        rings=[[[rng.randrange(-9,10),rng.randrange(-9,10)] for _ in range(rng.randrange(3,8))] for _ in range(rng.randrange(1,4))]
        geometry_cases.append({'rings':rings,'rule':rng.choice(['evenodd','nonzero'])})
    geometry_cases += [{'rings':[[[0,0],[4,4],[0,4],[4,0]]],'rule':'evenodd'},
      {'rings':[[[0,0],[4,0],[4,4],[0,4]]]*2,'rule':'nonzero'},
      {'rings':[[[0,0],[4,0],[4,4],[0,4]]]*2,'rule':'evenodd'},
      {'rings':[[[0,0],[1000000000,1],[999999999,1]]],'rule':'nonzero'}]
    for k in range(30):
        state=(0,None,False);ops=[];n=rng.randrange(4,9)
        for i in range(n):
            kind=rng.choice(['claim','claim','finish','reset']);args={}
            if kind=='claim':args={'worker':rng.choice(['a','b']),'now':rng.randrange(8),'ttl':rng.randrange(1,5)}
            if kind=='finish':args={'worker':rng.choice(['a','b']),'now':rng.randrange(8),'token':rng.randrange(1,5)}
            o={'id':str(i),'op':kind,'args':args,'start':i*3,'end':i*3+2,'result':None};state,o['result']=step(state,o);ops.append(o)
        # Preserve valid history under overlapping intervals; some operations pending.
        positions=list(range(n*2));rng.shuffle(positions)
        for i,o in enumerate(ops):o['start']=i;o['end']=n+i if k%3 else i*3+2
        if k%3==0:
            for i,o in enumerate(ops):o['start']=i*3
        if k%4==0:ops[rng.randrange(n)]['end']=None
        if k%2:
            o=rng.choice(ops);o['result']= (not o['result']) if type(o['result']) is bool else 99
        lease_cases.append({'history':ops})
    # Pending claim can justify a completed finish; expired boundary and fencing.
    lease_cases += [{'history':[{'id':'a','op':'claim','args':{'worker':'a','now':0,'ttl':2},'start':0,'end':None,'result':None}, {'id':'b','op':'finish','args':{'worker':'a','token':1,'now':1},'start':1,'end':2,'result':True}]}]
    for _ in range(26):
        n=rng.randrange(2,6);edges=[]
        for _ in range(rng.randrange(2,8)):
            upper=rng.randrange(3);edges.append({'u':rng.randrange(n),'v':rng.randrange(n),'lower':rng.randrange(upper+1),'upper':upper,'cost':rng.randrange(-5,6)})
        flow_cases.append({'n':n,'source':0,'sink':1,'amount':rng.randrange(3),'edges':edges})
    for _ in range(16):
        topology=[(0,2),(2,1),(2,3),(3,2),(1,0),(0,3),(3,1)]
        es=[{'u':u,'v':v,'lower':0,'upper':rng.randrange(1,4),'cost':rng.randrange(-5,6)} for u,v in topology]
        if rng.randrange(2):es[0]['lower']=es[1]['lower']=1
        flow_cases.append({'n':4,'source':0,'sink':1,'amount':1,'edges':es})
    flow_cases += [{'n':4,'source':0,'sink':1,'amount':2,'edges':[{'u':0,'v':1,'lower':0,'upper':3,'cost':2},{'u':2,'v':3,'lower':0,'upper':1000000,'cost':-4},{'u':3,'v':2,'lower':0,'upper':1000000,'cost':1}]},
      {'n':2,'source':0,'sink':1,'amount':0,'edges':[{'u':0,'v':0,'lower':2,'upper':1000000,'cost':-3}]}]
    return {'geometry':geometry_cases,'leases':lease_cases,'flow':flow_cases}

CONTRACTS={'geometry':GEOMETRY,'leases':LEASES,'flow':FLOW}
REFERENCES={'geometry':geometry,'leases':leases,'flow':flow}
