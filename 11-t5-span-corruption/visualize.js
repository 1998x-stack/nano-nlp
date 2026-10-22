/* 11: T5 span corruption (real target formatting, sentinel semantics). */
(()=>{const V=NanoViz,u=V.mount("T5 Span Corruption：输入与目标序列","分别选择两个删除片段；sentinel 负责标识不同片段，而非普通 [MASK] 分类。");
const tokens=["the","cat","sat","on","mat"];let first=1,second=4;
V.slider(u.controls,"第一个 span",0,3,1,1,v=>{first=v;render();});
V.slider(u.controls,"第二个 span",1,4,4,1,v=>{second=v;render();});
function render(){
V.clear(u.output);const positions=[first,second].sort((a,b)=>a-b);
if(positions[0]===positions[1]){V.note(u.output,"两个 span 需要位于不同位置。");return;}
const corrupt=[],target=[];
tokens.forEach((token,i)=>{
const idx=positions.indexOf(i);
if(idx<0)corrupt.push(token);else{
const sentinel="<extra_id_"+idx+">";corrupt.push(sentinel);target.push(sentinel,token);
}});
target.push("<eos>");
V.table(u.output,["步骤","Tokens","Attention 模式"],[
["原始输入",tokens.join(" "),"无"],["Encoder 输入",corrupt.join(" "),"双向"],["Decoder 目标",target.join(" "),"causal + cross-attn"]]);
V.note(u.output,"预训练目标是恢复被删片段，且 Decoder 目标包含 sentinel token。真正训练需要随机连续 span 的采样和正确 target shift。");
}render();
})();