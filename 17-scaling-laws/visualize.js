/* 17: constrained search over token/parameter budget under explicit toy scaling law. */
(()=>{const V=NanoViz,u=V.mount("Scaling Law：固定训练预算搜索 N / D","改变预算，枚举可行的参数规模 N 与 token 规模 D；损失系数纯属示意，不是 Chinchilla 实测曲线。");
const options=[2,4,8,16,32,64,128,256];let budget=4096;
V.slider(u.controls,"计算预算 C=N×D",64,8192,4096,64,v=>{budget=v;render();});
function loss(n,d){return 1+.8/Math.pow(n,.32)+.5/Math.pow(d,.28);}
function render(){
V.clear(u.output);const candidates=[];
options.forEach(n=>options.forEach(d=>{if(n*d<=budget)candidates.push({n,d,loss:loss(n,d)});}));
if(!candidates.length){V.note(u.output,"无可行参数组合");return;}
candidates.sort((a,b)=>a.loss-b.loss);
const best=candidates[0];
V.el("p",u.output,"最佳 N="+best.n+"，D="+best.d+"，Toy loss="+best.loss.toFixed(4),"deep-result");
V.table(u.output,["候选 N","候选 D","C 使用","预测 loss"],candidates.slice(0,8).map(r=>[r.n,r.d,r.n*r.d,r.loss.toFixed(5)]));
V.note(u.output,"可行集合满足 N×D≤预算；真实 Scaling Laws 必须来自实际训练拟合及有效的外推检验。");
}render();
})();