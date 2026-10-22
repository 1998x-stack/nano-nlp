/* Full BM25 formula: term count, idf, length, query by user. */
(()=>{const V=NanoViz,u=V.mount("BM25 排名：拆解每个 Query Term 的贡献","尝试更换查询与 b，观察长文惩罚以及 TF 饱和的效果。");
const docs=["attention improves language understanding","attention attention attention in longer language models and other computational contexts","bm25 uses term frequency for document retrieval","language models can predict the next token"];
const input=V.input(u.controls,"搜索关键词","attention language",render);
let b=.75;V.slider(u.controls,"长度归一化 b",0,1,.75,.05,v=>{b=v;render();});
function tok(s){return(s.toLowerCase().match(/[a-z]+/g)||[]);}
function render(){
V.clear(u.output);const q=[...new Set(tok(input.value))];const tokens=docs.map(tok);
const avg=tokens.reduce((a,b)=>a+b.length,0)/docs.length;
const ranked=tokens.map((words,i)=>{const freq={};words.forEach(x=>freq[x]=(freq[x]||0)+1);
const K=1.5*(1-b+b*words.length/avg);
const parts=q.map(w=>{const df=tokens.filter(t=>t.includes(w)).length;
const idf=Math.log(1+(docs.length-df+.5)/(df+.5));const tf=freq[w]||0;
return tf?idf*tf*2.5/(tf+K):0;});
return {name:docs[i],score:parts.reduce((a,b)=>a+b,0),parts};}).sort((a,b)=>b.score-a.score);
V.table(u.output,["文档","BM25 score","每词贡献"],ranked.map(r=>[r.name,r.score.toFixed(4),r.parts.map(v=>v.toFixed(3)).join(" + ")]));
V.note(u.output,"所有分值是基于当前 toy 索引现场计算的，不是神经向量相似度。");
}render();
})();