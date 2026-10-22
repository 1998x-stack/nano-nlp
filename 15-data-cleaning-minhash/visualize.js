/* 15: Live shingle Jaccard + probability of LSH candidate. */
(()=>{const V=NanoViz,u=V.mount("MinHash / LSH：近似去重的概率边界","更换两份文档及 n-gram 大小；使用真实集合 Jaccard 推导 LSH 候选概率。");
const left=V.input(u.controls,"文档 A","nlp models learn from a large text corpus",render);
const right=V.input(u.controls,"文档 B","nlp models learn from large text corpora",render);
let ngram=2,bands=8;
V.slider(u.controls,"词 n-gram",1,4,2,1,v=>{ngram=v;render();});
V.slider(u.controls,"LSH bands (r=3)",1,16,8,1,v=>{bands=v;render();});
function shingles(text){
const w=(text.toLowerCase().match(/[a-z]+/g)||[]),set=new Set();
for(let i=0;i+ngram<=w.length;i++)set.add(w.slice(i,i+ngram).join(" "));
return set;}
function render(){
V.clear(u.output);const a=shingles(left.value),b=shingles(right.value);
const common=[...a].filter(x=>b.has(x)),union=new Set([...a,...b]);
const similarity=union.size?common.length/union.size:1;
const hit=1-Math.pow(1-Math.pow(similarity,3),bands);
V.table(u.output,["指标","值"],[["A shingles",a.size],["B shingles",b.size],["交集",common.length],["并集",union.size],
["Jaccard",similarity.toFixed(4)],["P(LSH candidate)",hit.toFixed(5)]]);
V.note(u.output,"相似度约等于 MinHash 单次碰撞概率，LSH 概率不是硬阈值；生产需要候选后的精确相似度复核。");
}render();
})();