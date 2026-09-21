import itertools
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tasks


class OracleTests(unittest.TestCase):
    def test_flow_against_exhaustive_integer_assignments(self):
        for seed in range(40):
            for q in tasks.cases(seed)['flow'][:-2]:
                self.assertEqual(tasks.flow(q),tasks.brute_flow(q),q)

    def test_flow_disconnected_cycle_and_self_loop(self):
        a,b=tasks.cases(1)['flow'][-2:]
        self.assertEqual(tasks.flow(a),{'cost':-2999996})
        self.assertEqual(tasks.flow(b),{'cost':-3000000})

    def test_geometry_hand_answers(self):
        for ring,answer in [([[0,0],[4,4],[0,4],[4,0]],'8'),([[0,0],[4,0],[4,3],[0,3]],'12'),([[0,0],[1000000000,1],[999999999,1]],'1/2'),([[0,0],[0,0],[1,1]],'0')]:
            for rule in ('evenodd','nonzero'):
                self.assertEqual(tasks.geometry({'rings':[ring],'rule':rule}),{'area':answer})
        square=[[0,0],[4,0],[4,4],[0,4]]
        self.assertEqual(tasks.geometry({'rings':[square,square],'rule':'evenodd'}),{'area':'0'})
        self.assertEqual(tasks.geometry({'rings':[square,square],'rule':'nonzero'}),{'area':'16'})
        self.assertEqual(tasks.geometry({'rings':[square,square[::-1]],'rule':'nonzero'}),{'area':'0'})

    def test_geometry_against_independent_unit_cell_rectangle_oracle(self):
        rng=random.Random(72)
        for _ in range(100):
            rectangles=[];rings=[]
            for _ in range(4):
                x,u=sorted(rng.sample(range(-4,5),2));y,v=sorted(rng.sample(range(-4,5),2));sign=rng.choice([-1,1]);rectangles.append((x,u,y,v,sign));ring=[[x,y],[u,y],[u,v],[x,v]];rings.append(ring if sign==1 else ring[::-1])
            for rule in ('evenodd','nonzero'):
                area=0
                for x,y in itertools.product(range(-4,4),repeat=2):
                    winding=sum(s for a,b,c,d,s in rectangles if a<=x<b and c<=y<d)
                    area+=int(winding%2!=0 if rule=='evenodd' else winding!=0)
                self.assertEqual(tasks.geometry({'rings':rings,'rule':rule}),{'area':str(area)})

    def test_geometry_translation_scale_and_reversal(self):
        for q in tasks.cases(8)['geometry']:
            expected=tasks.F(tasks.geometry(q)['area'])
            rings=[[[x*3+100,y*3-27] for x,y in ring[::-1]] for ring in q['rings']]
            self.assertEqual(tasks.F(tasks.geometry({'rings':rings,'rule':q['rule']})['area']),expected*9)

    def test_lease_solver_against_permutation_oracle(self):
        # Different traversal and state implementation, no reuse of production step().
        def brute(q):
            ops=q['history'];pending=[i for i,o in enumerate(ops) if o['end'] is None];completed=[i for i,o in enumerate(ops) if o['end'] is not None]
            for mask in range(1<<len(pending)):
                chosen=completed+[i for j,i in enumerate(pending) if mask>>j&1]
                for order in itertools.permutations(chosen):
                    where={v:i for i,v in enumerate(order)}
                    if any(a['end'] is not None and a['end']<b['start'] and where[i]>where[j] for i,a in enumerate(ops) for j,b in enumerate(ops) if i in where and j in where):continue
                    generation=0;owner=None;token=None;expiry=0;done=False;valid=True
                    for i in order:
                        o=ops[i];a=o['args']
                        if o['op']=='reset':generation+=1;owner=None;token=None;done=False;result=generation
                        elif o['op']=='claim':
                            result=None
                            if not done and (owner is None or a['now']>=expiry):
                                generation+=1;owner=a['worker'];token=generation;expiry=a['now']+a['ttl'];result=token
                        else:
                            result=not done and owner is not None and owner==a['worker'] and token==a['token'] and a['now']<expiry
                            if result:done=True
                        if o['end'] is not None and (type(result)!=type(o['result']) or result!=o['result']):valid=False;break
                    if valid:return {'linearizable':True}
            return {'linearizable':False}
        for seed in range(8):
            for q in tasks.cases(seed)['leases']:
                if len(q['history'])<=6:self.assertEqual(tasks.leases(q),brute(q),q)

    def test_pending_claim_and_expiry_boundary(self):
        q=tasks.cases(1)['leases'][-1]
        self.assertTrue(tasks.leases(q)['linearizable'])
        q['history'][1]['args']['now']=2
        self.assertFalse(tasks.leases(q)['linearizable'])

    def test_distinct_event_positions(self):
        for seed in range(10):
            for q in tasks.cases(seed)['leases']:
                positions=[p for o in q['history'] for p in (o['start'],o['end']) if p is not None]
                self.assertEqual(len(positions),len(set(positions)))

if __name__=='__main__':unittest.main()
