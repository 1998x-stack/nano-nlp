/* Teach negative sampling objective using adjustable hypothetical logits. */
(()=>{const V=NanoViz,u=V.mount("Skip-gram：正负采样损失","调整假设向量点积；这里计算真实目标公式，但不是宣称这些数值来自训练好的词向量。");
let p=1,n=-1,k=3;V.slider(u.controls,"正样本 dot",-4,4,1,.25,x=>{p=x;render();});
V.slider(u.controls,"负样本 dot",-4,4,-1,.25,x=>{n=x;render();});
V.slider(u.controls,"负样本个数 K",1,8,3,1,x=>{k=x;render();});
function sp(x){return Math.log1p(Math.exp(x));}
function render(){
V.clear(u.output);const positive=sp(-p),negative=k*sp(n);
V.table(u.output,["分量","计算","Loss"],[
["真实上下文","softplus(−pos_dot)",positive.toFixed(5)],
["负样本集合","K × softplus(neg_dot)",negative.toFixed(5)],
["总损失","positive + negative",(positive+negative).toFixed(5)]]);
V.bars(u.output,[["σ(positive)",1/(1+Math.exp(-p))],["σ(negative)",1/(1+Math.exp(-n))]]);
V.note(u.output,"使正样本内积变大、负样本内积变小。真实代码使用 Embedding 和梯度下降更新参数。");
}render();
})();