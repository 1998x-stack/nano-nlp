/* 根据玩具 CFG 的真实规则解析输入，而不是使用假的预设输出。 */
(()=>{const V=NanoViz,u=V.mount("CFG 递归展开实验","替换输入 token，逐位置检查终结符、非终结符和完整消耗条件。");
const input=V.input(u.controls,"五词句子","the cat sees a dog",render);
function render(){
V.clear(u.output);
const words=input.value.toLowerCase().trim().split(/\s+/);
const rules=[["the","a"],["cat","dog"],["sees","likes"],["the","a"],["cat","dog"]];
const labels=["Det","N","V","Det","N"];
const ok=words.length===5&&words.every((w,i)=>rules[i].includes(w));
V.el("p",u.output,ok?"✓ S → NP VP 解析成功":"× 当前产生式无法完整匹配","deep-result");
V.table(u.output,["位置","Token","规则","匹配"],Array.from({length:Math.max(words.length,5)},(_,i)=>[i+1,words[i]||"∅",labels[i]||"无",rules[i]?.includes(words[i])?"✓":"×"]));
V.note(u.output,ok?"树：S(NP(Det("+words[0]+"),N("+words[1]+")), VP(V("+words[2]+"),NP(Det("+words[3]+"),N("+words[4]+"))))":"样例：a dog likes the cat。规则接受性不等于语义理解。");
}render();
})();