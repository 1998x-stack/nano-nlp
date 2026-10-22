/* 14: Group-relative advantages and no variance edge case. */
(()=>{const V=NanoViz,u=V.mount("Verifier RL：组内奖励与优势标准化","改变组大小和正确候选数量，观察 A=(r−mean)/(std+ε)。全对或全错时没有有效相对优势。");
let n=8,correct=4;
V.slider(u.controls,"采样数量",2,16,8,1,v=>{n=v;correct=Math.min(correct,n);render();});
V.slider(u.controls,"正确候选数",0,16,4,1,v=>{correct=v;render();});
function render(){
V.clear(u.output);const c=Math.min(correct,n),p=c/n,std=Math.sqrt(p*(1-p)),eps=1e-6;
const positive=std?(1-p)/(std+eps):0,negative=std?(-p)/(std+eps):0;
const rewards=Array.from({length:n},(_,i)=>i<c?1:0);
V.table(u.output,["统计","值"],[["Rewards",rewards.join(", ")],["mean(r)",p.toFixed(4)],["std(r)",std.toFixed(4)],["A(正确)",positive.toFixed(4)],["A(错误)",negative.toFixed(4)]]);
V.bars(u.output,[["正确率",p],["负例比例",1-p]]);
V.note(u.output,"真正 RL 项目的 verifier 要独立于生成器；本章用确定性的整数运算代替主观模型评分。");
}render();
})();