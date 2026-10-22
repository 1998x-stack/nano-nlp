/* Live corpus editing + add-alpha smoothed Bigram probability. */
(()=>{const V=NanoViz,u=V.mount("训练语料 → 频数 → 条件概率","修改语料和上下文，现场训练一个 N-gram 模型；所有候选输出的概率之和应为 1。");
const corpus=V.input(u.controls,"语料（使用 | 分隔句子）","the cat likes fish | the dog likes fish | the cat likes milk | the dog eats fish",render);
const prev=V.input(u.controls,"上一个词","likes",render);
let alpha=1;V.slider(u.controls,"平滑 α",.1,3,1,.1,v=>{alpha=v;render();});
function render(){
V.clear(u.output);const rows=corpus.value.toLowerCase().split("|").map(x=>x.trim()).filter(Boolean).slice(0,30);
const counts={},vocab=new Set(["<eos>"]);
rows.forEach(line=>{const words=line.split(/\s+/).filter(Boolean);words.forEach(w=>vocab.add(w));words.push("<eos>");
words.forEach((w,i)=>{const h=i?words[i-1]:"<bos>";if(!counts[h])counts[h]={};counts[h][w]=(counts[h][w]||0)+1;});});
const count=counts[prev.value.toLowerCase().trim()]||{};
const n=Object.values(count).reduce((a,b)=>a+b,0);
const probs=[...vocab].map(w=>[w,((count[w]||0)+alpha)/(n+alpha*vocab.size)]).sort((a,b)=>b[1]-a[1]);
V.table(u.output,["统计量","值"],[["count(context)",n],["词表大小",vocab.size],["分母 n+αV",(n+alpha*vocab.size).toFixed(2)]]);
V.bars(u.output,probs);V.note(u.output,"校验：Σ P(w|context) = "+probs.reduce((a,b)=>a+b[1],0).toFixed(5)+"。");
}render();
})();