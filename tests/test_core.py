import unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag import BM25, rank, rrf, metrics, load_chunks, read_jsonl, ROOT, calibrate
class RetrievalTests(unittest.TestCase):
    def test_multievidence_recall(self):
        m=metrics(['a','x','b'],['a','b'],1)
        self.assertEqual(m['recall'],.5)
        self.assertEqual(m['mrr'],1)
        self.assertIsNone(metrics(['a'],[]))
    def test_mrr_and_ndcg(self):
        m=metrics(['x','a','b'],['a','b'],3)
        self.assertEqual(m['mrr'],.5)
        self.assertLess(m['ndcg'],1)
        self.assertEqual(m['recall'],1)
    def test_fusion_rewards_agreement(self):
        self.assertEqual(rrf([[0,1,2],[3,1,4]])[0],1)
    def test_identifier_and_unknown(self):
        bm=BM25(['AUTH_401_EXPIRED 令牌过期','SEARCH_429_LIMIT 限流','搜索技术资料'])
        self.assertEqual(rank(bm.scores('SEARCH_429_LIMIT'),1),[1])
        self.assertTrue(all(x==0 for x in bm.scores('uniqueterm98765')))
    def test_evidence_ids_and_split(self):
        chunks=load_chunks(); ids={c['id'] for c in chunks}
        self.assertEqual(len(chunks),len(ids))
        qs=read_jsonl(ROOT/'data/questions.jsonl')
        self.assertEqual(sum(q['split']=='test' for q in qs),32)
        self.assertTrue(all(set(q['relevant_ids'])<=ids for q in qs))
        self.assertTrue(all(q['label_status']=='AI初稿，待用户审核' for q in qs))
    def test_calibration(self):
        rows=[{'top_score':2,'relevant_ids':['a']},{'top_score':-2,'relevant_ids':[]}]
        c=calibrate(rows)
        self.assertEqual(c['threshold'],0)
        self.assertEqual(c['dev_balanced_accuracy'],1)
if __name__=='__main__': unittest.main()
