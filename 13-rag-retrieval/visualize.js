/* 13: Evidence retrieval, scores and abstention in a fully local mini RAG. */
(()=>{const V=NanoViz,u=V.mount("RAG：从检索分数到可引用答案","修改问题并改变 Top-k；当没有正分证据时拒答，不能凭模型记忆伪造来源。");
const docs=[["D1","Transformer uses self-attention to connect tokens."],["D2","BERT uses masked language modeling."],
["D3","GPT predicts next tokens with causal attention."],["D4","BM25 ranks documents by term frequency."]];
const query=V.input(u.controls,"问题","What does Transformer use?",render);
let k=2;V.slider(u.controls,"Top-k",1,4,2,1,n=>{k=n;render();});
function tokens(s){return(s.toLowerCase().match(/[a-z0-9]+/g)||[]);}
function render(){
V.clear(u.output);const q=[...new Set(tokens(query.value))];
const all=docs.map(d=>tokens(d[1])),avg=all.reduce((s,d)=>s+d.length,0)/all.length;
const ranking=docs.map((d,i)=>{const tf={};all[i].forEach(w=>tf[w]=(tf[w]||0)+1);
const lengthK=1.5*(.25+.75*all[i].length/avg);
const score=q.reduce((sum,w)=>{const df=all.filter(a=>a.includes(w)).length;
const idf=Math.log(1+(docs.length-df+.5)/(df+.5));
return sum+(tf[w]||0)*2.5*idf/((tf[w]||0)+lengthK);},0);
return {id:d[0],text:d[1],score};}).sort((a,b)=>b.score-a.score).slice(0,k);
V.table(u.output,["来源","BM25","证据"],ranking.map(d=>[d.id,d.score.toFixed(4),d.text]));
const top=ranking[0],answer=top&&top.score>0?top.text+" ["+top.id+"]":"证据不足";
V.el("p",u.output,"回答："+answer,"deep-result");
V.note(u.output,"本例使用可追踪抽取式回答，不依赖外部 LLM。文档相关性高不一定完全支持一个复杂问题。");
}render();
})();