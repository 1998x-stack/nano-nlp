/* 16: Low rank LoRA parameter savings and exact shape accounting. */
(()=>{const V=NanoViz,u=V.mount("LoRA：低秩矩阵形状与参数账本","调整输入维度、输出维度和 rank，观察 W、A、B 的参数规模与 adapter 比例。");
let din=256,dout=256,rank=8;
V.slider(u.controls,"输入维度 in",32,1024,256,32,v=>{din=v;render();});
V.slider(u.controls,"输出维度 out",32,1024,256,32,v=>{dout=v;render();});
V.slider(u.controls,"Rank r",1,64,8,1,v=>{rank=v;render();});
function render(){
V.clear(u.output);const full=din*dout,extra=rank*(din+dout),ratio=extra/full;
V.table(u.output,["矩阵","形状","参数数目"],[
["冻结 W",dout+" × "+din,full],["可训练 A",rank+" × "+din,rank*din],
["可训练 B",dout+" × "+rank,dout*rank],
["增量 BA",dout+" × "+din,extra]]);
V.el("p",u.output,"Adapter / Full 参数比例 = "+(ratio*100).toFixed(2)+"%","deep-result");
V.note(u.output,"例子中忽略 bias；当 rank 过大时低秩方式甚至可能没有参数优势。冻结 W 不意味着推理不使用 W。");
}render();
})();