/* 一个轻量可复用的可视化播放器：没有依赖、能直接从本地 HTML 打开。 */
(() => {
  const doc = document, meta = JSON.parse(doc.getElementById('lesson').textContent);
  const root = doc.getElementById('pipeline');
  const prev = doc.getElementById('prev'), next = doc.getElementById('next');
  let index = 0;
  const safe = value => String(value);
  function show(i) {
    index = Math.max(0, Math.min(meta.steps.length - 1, i));
    root.replaceChildren();
    meta.steps.forEach((step, j) => {
      const b = doc.createElement('button');
      b.textContent = (j + 1) + '  ' + step.label;
      b.className = j === index ? 'active' : (j < index ? 'done' : '');
      b.setAttribute('aria-current', String(j === index));
      b.addEventListener('click', () => show(j));
      root.append(b);
      if (j < meta.steps.length - 1) {
        const a = doc.createElement('span'); a.className = 'arrow'; a.textContent = '→'; root.append(a);
      }
    });
    doc.getElementById('counter').textContent = (index + 1) + ' / ' + meta.steps.length;
    doc.getElementById('step-name').textContent = meta.steps[index].label;
    doc.getElementById('step-desc').textContent = meta.steps[index].desc;
    doc.getElementById('step-code').textContent = meta.steps[index].code;
    prev.disabled = index === 0; next.disabled = index === meta.steps.length - 1;
  }
  prev.addEventListener('click', () => show(index - 1));
  next.addEventListener('click', () => show(index + 1));
  doc.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'ArrowLeft') show(index - 1);
    if (e.key === 'ArrowRight') show(index + 1);
  });
  show(0);

  /* 不止静态幻灯片：按主题提供一个真实公式驱动的数值实验。 */
  const sim = doc.createElement('section'); sim.className = 'sim';
  const title = doc.createElement('h2'), expl = doc.createElement('p');
  const controls = doc.createElement('div'), output = doc.createElement('div');
  title.textContent = '互动微实验';
  sim.append(title, expl, controls, output);
  doc.querySelector('.workbench').after(sim);
  function slider(label, min, max, val, step, render) {
    const line = doc.createElement('label'); line.style.display = 'block';
    const cap = doc.createElement('span'), input = doc.createElement('input');
    input.type = 'range'; input.min = min; input.max = max; input.step = step; input.value = val;
    cap.textContent = label + ': ' + val;
    input.addEventListener('input', () => {cap.textContent = label + ': ' + input.value; render(+input.value);});
    line.append(cap, input); controls.replaceChildren(line); render(+val);
  }
  function bars(items) {
    output.replaceChildren();
    const all = doc.createElement('div'); all.className = 'bars';
    items.forEach(([name, val]) => {
      const r = doc.createElement('div'); r.className = 'bar';
      const n = doc.createElement('span'); n.className = 'name'; n.textContent = name;
      const track = doc.createElement('div'); track.className = 'track';
      const fill = doc.createElement('div'); fill.className = 'fill'; fill.style.width = Math.max(0,Math.min(100,val*100))+'%';
      track.append(fill);const value=doc.createElement('span'); value.className='value';value.textContent=val.toFixed(3);
      r.append(n,track,value);all.append(r);
    });output.append(all);
  }
  function matrix(n, allow) {
    const grid=doc.createElement('div');grid.className='matrix';grid.style.gridTemplateColumns='repeat('+n+',1fr)';
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){const b=doc.createElement('div');b.className='cell'+(allow(i,j)?'':' blocked');b.textContent=allow(i,j)?'✓':'×';grid.append(b);}
    output.replaceChildren(grid);
  }
  function metric(label, value) {
    output.replaceChildren();const p=doc.createElement('div');p.className='metric';p.textContent=label+' = '+value;output.append(p);
  }
  const name = meta.title.toLowerCase();
  if (name.includes('n-gram')) {
    expl.textContent='将 likes 后的 fish 出现 2 次、milk 出现 1 次，改变 Laplace 平滑强度。';
    slider('平滑 α',0.1,2,1,.1,a=>{const V=4;bars([['fish',(2+a)/(3+V*a)],['milk',(1+a)/(3+V*a)],['unseen',a/(3+V*a)]]);});
  } else if(name.includes('bm25')){
    expl.textContent='比较两个有相同词频但篇幅不同的文档；b 增大时，长度惩罚增强。';
    slider('长度归一化 b',0,1,.75,.05,b=>{
      const idf=Math.log(1+(3-1+.5)/(1+.5)),k=1.5, avg=10;
      const score=dl=>idf*(k+1)/(1+k*(1-b+b*dl/avg));
      const a=score(5),c=score(20);bars([['短文',a/2],['长文',c/2]]);
    });
  } else if(name.includes('attention')||name.includes('gpt')||name.includes('bert')||name.includes('t5')) {
    expl.textContent='Transformer 解码器采用下三角可见性；BERT Encoder 是双向注意力。改变可见范围观察掩码结构。';
    const bidirectional=name.includes('bert');
    slider('序列长度',3,7,5,1,n=>matrix(n,(i,j)=>bidirectional||j<=i));
  } else if(name.includes('viterbi')){
    expl.textContent='模拟不同的状态转移可信度，观察一条候选路径的联合 log-score。';
    slider('N→V 概率',.1,.95,.8,.05,p=>metric('log(起始×发射×转移×发射)',(Math.log(.8)+Math.log(.8)+Math.log(p)+Math.log(.9)).toFixed(3)));
  } else if(name.includes('skip-gram')){
    expl.textContent='改变两个向量的点积。正样本损失 softplus(−dot) 越小越好。';
    slider('正样本点积',-4,4,1,.25,x=>metric('−log sigmoid(dot)',Math.log(1+Math.exp(-x)).toFixed(3)));
  } else if(name.includes('lstm')){
    expl.textContent='LSTM 的 cell 更新近似 c_new = forget × c_old + input × candidate。';
    slider('遗忘门 f',0,1,.7,.05,f=>metric('c_new (c_old=1,candidate=.5,input=.4)',(f+.4*.5).toFixed(3)));
  } else if(name.includes('seq2seq')){
    expl.textContent='Attention 将 encoder states 按概率加权，改变第一个源位置的权重。';
    slider('第一个词的注意力权重',0,1,.45,.05,a=>bars([['source 1',a],['source 2',(1-a)/2],['source 3',(1-a)/2]]));
  } else if(name.includes('span corruption')){
    expl.textContent='输入损坏文本与 decoder 恢复目标需要不同格式。';
    metric('示例','the <extra_id_0> sat on <extra_id_1> → <extra_id_0> cat <extra_id_1> mat');
  } else if(name.includes('dpo')||name.includes('sft')){
    expl.textContent='策略相对于参考模型把 chosen 对 rejected 的对数优势提高 1.5。β 是偏好强度。';
    slider('偏好 β',.1,2,.5,.1,b=>metric('DPO loss',Math.log(1+Math.exp(-b*1.5)).toFixed(4)));
  } else if(name.includes('rag')){
    expl.textContent='改变 Top-k 检索文档数，控制送给生成器的上下文数量。';
    slider('Top-k',1,3,2,1,k=>metric('证据文档数量',String(k)));
  } else if(name.includes('推理 rl')){
    expl.textContent='组内平均 reward 会影响每个候选的相对优势（这里取正例 reward=1）。';
    slider('组均奖励',0,.9,.5,.1,m=>metric('未归一化 advantage',(1-m).toFixed(2)));
  } else if(name.includes('minhash')){
    expl.textContent='两个文档的 Jaccard = 0.8。band 数越多，成为 LSH 候选的概率通常越大（r=3）。';
    slider('band 数 b',1,16,8,1,b=>metric('P(candidate)',(1-Math.pow(1-Math.pow(.8,3),b)).toFixed(3)));
  } else if(name.includes('lora')){
    expl.textContent='低秩秩数 r 控制额外可训练参数 r×(input+output)。设 in=256,out=256。';
    slider('rank',1,32,4,1,r=>metric('adapter 参数量',String(r*(256+256))));
  } else if(name.includes('scaling')){
    expl.textContent='简化幂律 L=1+.8/N^.32+.5/D^.28，固定预算 N×D=4096。';
    slider('模型规模 N',2,128,32,2,n=>metric('假设 loss',(1+.8/Math.pow(n,.32)+.5/Math.pow(4096/n,.28)).toFixed(4)));
  } else {
    expl.textContent='点击上方任一步骤并结合 demo.py 阅读：算法行为对应哪些数据、约束和中间状态？';
    const note=doc.createElement('p');note.textContent='使用 ← / → 键也可切换算法步骤。';output.append(note);
  }
})();