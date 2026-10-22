/* 08: Scaled dot product attention + causal mask. */
(()=>{const V=NanoViz,u=V.mount("Self-Attention：QKᵀ 与未来 Mask","每行是当前位置 Query 的注意力概率；causal=开时未来 token 必须严格为零。");
let scale=1,causal=true;
V.slider(u.controls,"相似度放大倍数",.25,3,1,.25,v=>{scale=v;render();});
V.select(u.controls,"因果掩码",["on","off"],v=>{causal=v==="on";render();});
const Q=[[1,0],[0,1],[1,1],[2,1]],K=[[1,0],[0,1],[1,1],[2,0]];
function render(){
V.clear(u.output);const p=Q.map((q,i)=>{const logits=K.map((k,j)=>{
if(causal&&j>i)return -Infinity;
return(q[0]*k[0]+q[1]*k[1])*scale/Math.sqrt(2);});
return V.softmax(logits);});
V.heatmap(u.output,p,["key0","key1","key2","key3"],["query0","query1","query2","query3"]);
V.note(u.output,"每行 Softmax 和为 1；未来位置填 -∞，概率等于 0。切换 off 仅用于展示双向注意力差异。");
}render();
})();