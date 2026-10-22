/* Explicit log-space dynamic programming trace. */
(()=>{const V=NanoViz,u=V.mount("Viterbi 状态网格","自选连续两个观测词，查看每个状态最优分数和回溯指针。");
const x=V.select(u.controls,"第一个词",["dogs","run"],render);
const y=V.select(u.controls,"第二个词",["run","dogs"],render);
const start={N:.8,V:.2},trans={N:{N:.2,V:.8},V:{N:.8,V:.2}},em={N:{dogs:.8,run:.2},V:{dogs:.1,run:.9}};
function render(){
V.clear(u.output);const dp={N:Math.log(start.N*em.N[x.value]),V:Math.log(start.V*em.V[x.value])};
const score={},prev={};for(const state of ["N","V"]){
const choices=["N","V"].map(p=>[p,dp[p]+Math.log(trans[p][state])]).sort((a,b)=>b[1]-a[1]);
prev[state]=choices[0][0];score[state]=choices[0][1]+Math.log(em[state][y.value]);}
V.table(u.output,["时间","观测","log P(末态=N)","log P(末态=V)","最佳前驱"],[
["t=0",x.value,dp.N.toFixed(4),dp.V.toFixed(4),"start"],
["t=1",y.value,score.N.toFixed(4),score.V.toFixed(4),"N←"+prev.N+" / V←"+prev.V]]);
const last=score.N>=score.V?"N":"V";
V.el("p",u.output,"最优路径："+prev[last]+" → "+last,"deep-result");
V.note(u.output,"每个格子保留最高概率前驱；对数概率相加替代极小概率连乘。");
}render();
})();