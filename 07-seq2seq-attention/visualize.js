/* 07: Cross-attention weights, contexts, and teacher forcing vs inference. */
(()=>{const V=NanoViz,u=V.mount("Encoder–Decoder 跨注意力矩阵","针对三个 Decoder 位置计算源端对齐分布；滑块是加给对应词的 attention logit 偏置。");
let focus=1,strength=2;V.slider(u.controls,"关注源位置",1,3,1,1,v=>{focus=v;render();});
V.slider(u.controls,"聚焦强度",-2,4,2,.25,v=>{strength=v;render();});
function render(){
V.clear(u.output);const source=["I","love","NLP"],target=["我","爱","语言"];
const base=[[2,0,0],[0,2,0],[0,0,2]];
const matrix=base.map(row=>V.softmax(row.map((v,j)=>v+(j+1===focus?strength:0))));
V.heatmap(u.output,matrix,source,target);
V.table(u.output,["阶段","输入 Token"],[["Encoder",source.join(" ")],["Decoder Teacher Forcing","<bos> 我 爱"],["推理时 Decoder","<bos> + 已经预测出的词"]]);
V.note(u.output,"权重只是可解释的数值模拟；实际训练由 project.py 计算 Decoder query 与 Encoder memory 的点积并更新参数。");
}render();
})();