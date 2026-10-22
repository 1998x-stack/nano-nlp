"""论文关键数学不变量：即使 toy 数据过拟合，也不能违反算法机制。"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def module(chapter):
    path=ROOT/chapter/"model.py"
    name="model_for_math_"+chapter.replace("-","_")
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

class MathCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        cls.torch=torch
        cls.torch.manual_seed(7)
        cls.torch.set_num_threads(1)

    def test_skipgram_objective_is_finite(self):
        m=module("05-word2vec-skipgram")
        words,pairs=m.make_pairs(m.TEXT,window=2)
        net=m.SkipGram(len(words))
        negatives=self.torch.randint(0,len(words),(len(pairs),3))
        loss=net(pairs[:,0],pairs[:,1],negatives)
        self.assertTrue(self.torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(net.center.weight.grad)

    def test_lstm_next_character_shape(self):
        m=module("06-rnn-lstm")
        net=m.LSTMLM(vocab_size=8)
        logits,loss=net(self.torch.tensor([[1,2,3,4]]),self.torch.tensor([[2,3,4,5]]))
        self.assertEqual(tuple(logits.shape),(1,4,8))
        self.assertTrue(self.torch.isfinite(loss))

    def test_seq2seq_attention_distribution(self):
        m=module("07-seq2seq-attention")
        net=m.TinyTranslator()
        out,alpha=net(m.SOURCE,m.TARGET)
        self.assertEqual(tuple(out.shape),(3,3,7))
        self.assertEqual(tuple(alpha.shape),(3,3,3))
        self.assertTrue(self.torch.allclose(alpha.sum(-1),self.torch.ones(3,3),atol=1e-6))

    def test_causal_attention_never_leaks_future(self):
        m=module("08-transformer-attention")
        net=m.CausalSelfAttention(16,2).eval()
        a=self.torch.randn(1,5,16)
        b=a.clone();b[:,-1,:]+=100
        with self.torch.no_grad():
            y1,att1=net(a)
            y2,att2=net(b)
        self.assertTrue(self.torch.allclose(y1[:,:4],y2[:,:4],atol=1e-5))
        mask=self.torch.ones(5,5,dtype=self.torch.bool).triu(1)
        self.assertTrue(self.torch.all(att1[...,mask]==0))

    def test_bert_reads_right_hand_context(self):
        m=module("09-bert-mlm")
        net=m.TinyMLM().eval()
        first=self.torch.tensor([[1,0,3,4]])
        second=self.torch.tensor([[1,0,3,5]])
        with self.torch.no_grad():
            v=net(first)[0,1]
            w=net(second)[0,1]
        self.assertGreater(float((v-w).abs().max()),1e-7)

    def test_gpt_prefix_independent_of_future_tokens(self):
        m=module("10-gpt-causal-lm")
        net=m.TinyGPT(8).eval()
        x=self.torch.tensor([[0,1,2,3]])
        z=self.torch.tensor([[0,1,2,4]])
        with self.torch.no_grad():
            left,_=net(x)
            right,_=net(z)
        self.assertTrue(self.torch.allclose(left[:,:3],right[:,:3],atol=1e-6))

    def test_t5_decoder_prefix_cannot_read_future_target(self):
        m=module("11-t5-span-corruption")
        net=m.TinyT5().eval()
        original=m.DECODER_INPUT
        changed=original.clone();changed[:,-1]=9
        with self.torch.no_grad():
            y=net(m.SOURCE,original)
            z=net(m.SOURCE,changed)
        self.assertTrue(self.torch.allclose(y[:,:4],z[:,:4],atol=1e-5))

    def test_lora_base_is_frozen_and_adapter_starts_at_zero(self):
        m=module("16-lora-finetuning")
        adapter=m.LoRALinear(in_dim=8,out_dim=4,rank=2)
        x=self.torch.randn(2,8)
        self.assertTrue(self.torch.allclose(adapter(x),adapter.base(x),atol=1e-7))
        adapter(x).sum().backward()
        self.assertIsNone(adapter.base.weight.grad)
        self.assertIsNotNone(adapter.B.weight.grad)

    def test_real_verifier_separate_from_policy(self):
        m=module("14-reasoning-rl-verifier")
        self.assertEqual(m.verify(0,3,4,7),1.0)
        self.assertEqual(m.verify(1,3,4,12),1.0)
        self.assertEqual(m.verify(1,3,4,7),0.0)

if __name__=="__main__":
    unittest.main()
