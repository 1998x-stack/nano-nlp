/* Explicit LSTM cell recurrence; real neural gates are input-dependent. */
(()=>{const V=NanoViz,u=V.mount("LSTM 门控记忆逐步更新","数值演示 c_t=f_t·c_(t−1)+i_t·g_t；直接观察梯度可经过的记忆通道。");
let forget=.75,keep=.35;V.slider(u.controls,"遗忘门 f",0,1,.75,.05,v=>{forget=v;render();});
V.slider(u.controls,"输入门 i",0,1,.35,.05,v=>{keep=v;render();});
function render(){
V.clear(u.output);const candidate=[.4,-.3,.8,.1,-.6];let cell=0,rows=[];
candidate.forEach((g,t)=>{const before=cell;cell=forget*cell+keep*g;
rows.push([t+1,g.toFixed(2),before.toFixed(4),forget.toFixed(2),keep.toFixed(2),cell.toFixed(4)]);});
V.table(u.output,["t","候选 g","历史 c","f","i","新 c"],rows);
V.note(u.output,"固定门值只是方便手算；实际 LSTM 的 f、i、g 随输入 token 和上一时刻隐藏状态变化。");
}render();
})();