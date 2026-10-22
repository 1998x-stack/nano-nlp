/* 12: DPO formula exactly using log-odds difference. */
(()=>{const V=NanoViz,u=V.mount("SFT vs DPO：偏好对数优势","用 chosen 相对于 rejected 的概率变化，计算直接偏好优化的 logistic loss。");
let p=1,r=0,beta=.5;
V.slider(u.controls,"Policy log odds",-4,4,1,.25,v=>{p=v;render();});
V.slider(u.controls,"Reference log odds",-4,4,0,.25,v=>{r=v;render();});
V.slider(u.controls,"β",.1,2,.5,.1,v=>{beta=v;render();});
function render(){
V.clear(u.output);const delta=p-r,z=beta*delta,loss=Math.log1p(Math.exp(-z));
V.table(u.output,["项","值"],[["log πθ(chosen) − log πθ(rejected)",p.toFixed(3)],
["log πref(chosen) − log πref(rejected)",r.toFixed(3)],
["策略相对参考优势",delta.toFixed(3)],["β × 优势",z.toFixed(3)],
["DPO loss = softplus(−βΔ)",loss.toFixed(5)]]);
V.note(u.output,"SFT 优化 chosen 的绝对 log likelihood；DPO 同时比较 chosen/rejected，并用冻结的 reference 约束相对变化。");
}render();
})();