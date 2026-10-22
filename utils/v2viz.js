/* nano-nlp v2.0 DOM-only visualizations, no external dependencies. */
(function(){
"use strict";
function el(tag,parent,value,css){
  const n=document.createElement(tag);
  if(value!==undefined&&value!==null)n.textContent=String(value);
  if(css)n.className=css;
  if(parent)parent.append(n);
  return n;
}
function mount(title,description){
  const section=el("section",document.querySelector(".app"),null,"deep-sim");
  el("h2",section,title);
  el("p",section,description,"deep-intro");
  return {section,controls:el("div",section,null,"deep-controls"),
          output:el("div",section,null,"deep-output")};
}
function input(parent,label,value,change){
  const wrapper=el("label",parent,null,"deep-control");
  el("span",wrapper,label);
  const x=el("input",wrapper);
  x.value=value;x.addEventListener("input",()=>change(x.value));
  return x;
}
function slider(parent,label,min,max,value,step,change){
  const wrapper=el("label",parent,null,"deep-control");
  const title=el("span",wrapper,label+": "+value);
  const x=el("input",wrapper);
  x.type="range";x.min=min;x.max=max;x.value=value;x.step=step;
  x.addEventListener("input",()=>{
    title.textContent=label+": "+x.value;change(Number(x.value));
  });
  return x;
}
function select(parent,label,options,change){
  const wrapper=el("label",parent,null,"deep-control");
  el("span",wrapper,label);
  const x=el("select",wrapper);
  options.forEach(o=>{const node=el("option",x,o);node.value=o;});
  x.addEventListener("change",()=>change(x.value));return x;
}
function table(parent,heads,rows){
  const wrap=el("div",parent,null,"deep-scroll");
  const tbl=el("table",wrap,null,"deep-table");
  const header=el("tr",el("thead",tbl));
  heads.forEach(x=>el("th",header,x));
  const body=el("tbody",tbl);
  rows.forEach(row=>{const line=el("tr",body);
    row.forEach(v=>el("td",line,v));});
  return tbl;
}
function bars(parent,values){
  const wrap=el("div",parent,null,"deep-bars");
  const max=Math.max(0.00001,...values.map(pair=>Math.abs(pair[1])));
  values.forEach(pair=>{
    const row=el("div",wrap,null,"deep-bar");
    el("span",row,pair[0]);
    const bg=el("div",row,null,"deep-bar-bg");
    const fg=el("div",bg,null,"deep-bar-fg");
    fg.style.width=(100*Math.abs(pair[1])/max).toFixed(1)+"%";
    el("span",row,Number(pair[1]).toFixed(4));
  });
}
function heatmap(parent,matrix,cols,rows){
  const t=table(parent,["↓"].concat(cols),matrix.map((r,i)=>[rows[i]].concat(r.map(x=>Number(x).toFixed(3)))));
  Array.from(t.tBodies[0].rows).forEach(tr=>{
    Array.from(tr.cells).slice(1).forEach(cell=>{
      const n=parseFloat(cell.textContent);
      if(Number.isFinite(n)){
        const alpha=(Math.max(0,Math.min(1,n))*.7+.05).toFixed(2);
        cell.style.background="rgba(13,148,136,"+alpha+")";
      }
    });
  });
}
function softmax(scores){
  const max=Math.max(...scores);
  const exp=scores.map(x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
  return exp.map(x=>x/sum);
}
function note(parent,text){return el("p",parent,text,"deep-note");}
function clear(parent){parent.replaceChildren();}
window.NanoViz={el,mount,input,slider,select,table,bars,heatmap,softmax,note,clear};
})();
