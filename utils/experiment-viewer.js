/* No synthetic metrics: all results are imported from local experiment artifacts. */
(() => {
  const root=document.querySelector(".app");
  if(!root) return;
  const panel=document.createElement("section");
  panel.className="experiment-viewer";
  panel.innerHTML="<h2>真实训练 / 评估结果</h2><p>从本章 outputs 目录选择 metrics.json、history.jsonl、examples.jsonl；只在本地浏览器解析，不上传文件。未选择文件时没有虚构曲线。</p>";
  root.append(panel);
  function renderFile(label,parse){
    const wrap=document.createElement("label");
    wrap.textContent=label;
    const inp=document.createElement("input");inp.type="file";inp.accept=".json,.jsonl";
    const out=document.createElement("div");
    wrap.append(inp);panel.append(wrap,out);
    inp.addEventListener("change",async()=>{
      out.replaceChildren();
      if(!inp.files[0])return;
      try{parse(await inp.files[0].text(),out);}
      catch(e){out.textContent="读取失败："+String(e.message||e);}
    });
  }
  function pre(parent,obj){
    const p=document.createElement("pre");p.className="evidence-examples";
    p.textContent=typeof obj==="string"?obj:JSON.stringify(obj,null,2);
    parent.append(p);
  }
  renderFile("导入 metrics.json",(raw,out)=>{
    const value=JSON.parse(raw);
    if(!value.metrics)throw Error("无 metrics 字段");
    pre(out,{seed:value.seed,metrics:value.metrics});
  });
  renderFile("导入 history.jsonl",(raw,out)=>{
    const rows=raw.split(/\r?\n/).filter(Boolean).map(x=>JSON.parse(x));
    const points=rows.filter(x=>Number.isFinite(x.loss));
    if(!points.length){out.textContent="本章没有梯度优化历史（适用于规则/搜索等项目）。";return;}
    const losses=points.map(x=>x.loss),lo=Math.min(...losses),hi=Math.max(...losses),span=Math.max(hi-lo,1e-10);
    const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");
    svg.setAttribute("viewBox","0 0 640 220");svg.classList.add("evidence-chart");
    svg.setAttribute("role","img");svg.setAttribute("aria-label","从真实训练 history.jsonl 绘制的 loss 折线图");
    const line=document.createElementNS(ns,"polyline");
    line.setAttribute("points",points.map((v,i)=>{
      const x=24+592*(points.length===1?.5:i/(points.length-1));
      const y=196-172*(v.loss-lo)/span;
      return x.toFixed(2)+","+y.toFixed(2);
    }).join(" "));
    line.setAttribute("fill","none");line.setAttribute("stroke","#0f766e");line.setAttribute("stroke-width","3");
    svg.append(line);out.append(svg);
    const meta=document.createElement("p");meta.textContent="实际记录 "+points.length+" 步；起始 loss="+losses[0].toFixed(4)+"；结束 loss="+losses[losses.length-1].toFixed(4);out.append(meta);
  });
  renderFile("导入 examples.jsonl",(raw,out)=>{
    const values=raw.split(/\r?\n/).filter(Boolean).map(x=>JSON.parse(x));
    pre(out,values.slice(0,8));
    if(values.length>8){const p=document.createElement("p");p.textContent="共 "+values.length+" 条，仅预览前 8 条。";out.append(p);}
  });
})();
