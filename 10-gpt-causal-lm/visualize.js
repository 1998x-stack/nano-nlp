/* 10: GPT shift targets / causal visibility / context limitation. */
(()=>{const V=NanoViz,u=V.mount("GPT Causal LM：Label Shift 与上下文窗口","更换输入字符序列，逐位置展示可见历史与下一 token 标签；末 token 右侧没有训练标签。");
const input=V.input(u.controls,"字符序列","to be or not",render);
let windowSize=6;V.slider(u.controls,"Context block_size",2,12,6,1,v=>{windowSize=v;render();});
function render(){
V.clear(u.output);const chars=[...input.value].slice(0,25);
if(chars.length<2){V.note(u.output,"请输入至少两个字符。");return;}
const rows=chars.slice(0,-1).map((c,i)=>[i,c===" "?"␣":c,chars.slice(Math.max(0,i-windowSize+1),i+1).join(" "),chars[i+1]===" "?"␣":chars[i+1]]);
V.table(u.output,["位置 t","输入 x[t]","causal 可见窗口","监督 y[t]=x[t+1]"],rows);
const m=chars.slice(0,Math.min(8,chars.length)).map((_,i)=>chars.slice(0,Math.min(8,chars.length)).map((__,j)=>j<=i?1:0));
V.heatmap(u.output,m,m.map((_,i)=>"K"+i),m.map((_,i)=>"Q"+i));
V.note(u.output,"截断历史长度不等于扩大位置编码。实际模型训练时对每个位置使用因果掩码，生成只输入真实已见 token。");
}render();
})();