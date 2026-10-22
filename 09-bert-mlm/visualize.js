/* 09: Masked token and bidirectional visible context; do not pretend to predict. */
(()=>{const V=NanoViz,u=V.mount("Masked LM 与 Causal LM：可见性差异","移动被遮挡位置，观察双向 Encoder 可读取左右 token，而 Decoder 不能访问未来位置。");
const words=["the","cat","eats","fish"];let index=1,mode="bidirectional";
V.slider(u.controls,"MASK 位置",0,3,1,1,v=>{index=v;render();});
V.select(u.controls,"Attention 类型",["bidirectional","causal"],v=>{mode=v;render();});
function render(){
V.clear(u.output);const masked=words.map((x,i)=>i===index?"[MASK]":x);
V.table(u.output,["输入","真实监督","Loss 位置"],[[masked.join(" "),words[index],String(index)]]);
const p=words.map((a,i)=>words.map((b,j)=>mode==="bidirectional"||j<=i?1:0));
V.heatmap(u.output,p,words,words);
V.note(u.output,"这里展示输入与合法的信息路径，不伪造 masked word 预测；模型输出必须由训练后的 project.py 计算。");
}render();
})();