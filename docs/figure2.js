'use strict';

(async function figure2Explorer() {
  const root = document.querySelector('#figure2');
  if (!root) return;
  const get = id => document.getElementById(id);
  const svgNS = 'http://www.w3.org/2000/svg';
  const make = (tag, attrs = {}, text) => {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const fmt = (value, digits = 3) => {
    if (value === 0) return '0';
    return Math.abs(value) < 0.001 || Math.abs(value) >= 10000
      ? value.toExponential(2) : Number(value.toPrecision(digits)).toString();
  };
  const visibleToken = token => token.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥');
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
  const color = (stops, t) => {
    const scaled = clamp(t) * (stops.length - 1);
    const i = Math.min(Math.floor(scaled), stops.length - 2);
    const ratio = scaled - i;
    return stops[i].map((value, channel) => Math.round(value + (stops[i + 1][channel] - value) * ratio));
  };
  const coolwarm = [[59,76,192],[141,176,254],[221,220,220],[244,152,122],[180,4,38]];
  const viridis = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const sampleFiles = ['assets/figure2-data.json', 'assets/figure2-sample2.json'];
  let samples;
  try {
    samples = await Promise.all(sampleFiles.map(async path => {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Data request: ${response.status}`);
      const data = await response.json();
      if (!data.points.length || data.points.length !== data.meta.plotted_complete_clusters || data.points.some(p => p.scores.length !== data.meta.block_size || !p.scores.every(Number.isFinite))) throw new Error('Invalid Figure 2 sample');
      return data;
    }));
  } catch (error) {
    get('f2-loading').textContent = 'The interactive data could not be loaded. Reload this page or view Figure 2 in the paper above.';
    console.error('Figure 2 data unavailable:', error);
    return;
  }
  get('f2-loading').hidden = true;
  get('f2-content').hidden = false;
  const allPoints = samples.flatMap(sample => sample.points);
  const allScores = allPoints.flatMap(point => point.scores);
  const scoreMin = Math.floor(Math.min(...allScores) / 2) * 2;
  const scoreMax = Math.ceil(Math.max(...allScores) / 2) * 2;
  const errorMin = Math.min(...allPoints.map(point => point.output_error));
  const errorMax = Math.max(...allPoints.map(point => point.output_error));
  const fittedAxes = samples.map(({points}) => ({
    X: {min:Math.floor((Math.min(...points.map(p => p.delta))-.3)*2)/2, max:Math.ceil((Math.max(...points.map(p => p.delta))+.3)*2)/2, left:80, right:711},
    Y: {min:Math.floor((Math.log10(Math.min(...points.map(p => p.importance)))-.15)*2)/2, max:Math.min(0,Math.ceil((Math.log10(Math.max(...points.map(p => p.importance)))+.15)*2)/2), top:24, bottom:430}
  }));
  const sharedX = {...fittedAxes[0].X, min:Math.min(...fittedAxes.map(axes => axes.X.min)), max:Math.max(...fittedAxes.map(axes => axes.X.max))};
  const sharedY = {...fittedAxes[0].Y, min:Math.min(...fittedAxes.map(axes => axes.Y.min)), max:Math.max(...fittedAxes.map(axes => axes.Y.max))};
  const scaleFor = (meta, delta) => 2 * meta.vmax * ((Math.cosh(delta) - 1) / meta.denominator + Math.tanh(delta / 2));
  const riskLow = Math.floor(Math.min(...samples.map(({meta}) => Math.log10(10 ** sharedY.min * scaleFor(meta, sharedX.min)))));
  const riskHigh = Math.ceil(Math.max(...samples.map(({meta}) => Math.log10(10 ** sharedY.max * scaleFor(meta, sharedX.max)))));
  let sharedAxes = false;
  let activeSampleIndex = 0;
  let selectedClusterId = 0;
  let releaseSample = () => {};

  function renderSample(data, sampleIndex) {
    const listeners = [];
    const listen = (node, type, handler) => {
      node.addEventListener(type, handler);
      listeners.push(() => node.removeEventListener(type, handler));
    };
    const {points, meta} = data;
    const {X,Y} = sharedAxes ? {X:sharedX,Y:sharedY} : fittedAxes[sampleIndex];
    let selected = points.find(point => point.id === selectedClusterId) || points[0];
    let tau = 10 ** Number(get('f2-tau').value);
    const plot = get('f2-plot');
    plot.replaceChildren(
      make('title', {id:'f2-plot-title'}, `Decision boundary for query ${meta.query_position}`),
      make('desc', {id:'f2-plot-desc'}, `${points.length} real KV clusters. Select a point or use the arrow keys to inspect its QK scores. Scatter axes ${sharedAxes?'are shared between samples':'fit the selected sample'}. Color and QK score scales remain shared.`)
    );
    Object.assign(plot.dataset,{xMin:X.min,xMax:X.max,yMin:Y.min,yMax:Y.max});
    root.dataset.axes = sharedAxes ? 'shared' : 'fit';
    root.dataset.sample = String(sampleIndex + 1);
    get('f2-sample-info').textContent = `${meta.visible_tokens.toLocaleString('en-US')} tokens · head ${meta.query_head} · ${points.length} clusters`;
    get('f2-attention-note').textContent = `Attention is softmax over all ${meta.visible_tokens.toLocaleString('en-US')} visible keys, not just this cluster.`;
    get('f2-data-source').textContent = `${sampleIndex === 0 ? 'Original Figure 2 query' : 'Additional query from the same experiment'}: Llama-3.1-8B, WikiText-2, layer ${meta.layer}, query head ${meta.query_head}, KV head ${meta.kv_head}, query position ${meta.query_position} (zero-based). The plot contains ${points.length} complete clusters of ${meta.block_size} keys. The remaining ${meta.partial_cluster_size} visible keys contribute to the fixed normalization but are not plotted.`;
    get('f2-data-verification').textContent = `All ${points.length} clusters were reproduced and matched the saved geometry.${meta.qk_reference_cases_verified ? ` The ${meta.qk_reference_cases_verified} archived per-token examples were also verified against their saved QK scores.` : ' QK scores were captured from the reproduced model run.'}`;
    const px = x => X.left + (x - X.min) / (X.max - X.min) * (X.right - X.left);
    const py = importance => Y.bottom - (Math.log10(importance) - Y.min) / (Y.max - Y.min) * (Y.bottom - Y.top);
    // This is the exact Vmax/Vmax level-set formula in the Figure 2 plotting code.
    // Changing tau never recomputes the fixed query, logits, summaries, or D.
    const multiplier = delta => scaleFor(meta, delta);
    const boundaryY = (delta, threshold = tau) => threshold / multiplier(delta);
    const curve = threshold => Array.from({length:241}, (_, i) => {
      const x = X.min + (X.max-X.min) * i / 240;
      return `${i ? 'L' : 'M'}${px(x).toFixed(3)},${py(boundaryY(x, threshold)).toFixed(3)}`;
    }).join(' ');
    const defs = make('defs');
    const clip = make('clipPath', {id:'f2-plot-clip'});
    clip.append(make('rect',{x:X.left,y:Y.top,width:X.right-X.left,height:Y.bottom-Y.top}));
    defs.append(clip); plot.append(defs);

    // The background is a fixed logarithmic risk field, independent of tau.
    const canvas = document.createElement('canvas');
    canvas.width=420; canvas.height=280;
    const ctx=canvas.getContext('2d');
    const image=ctx.createImageData(canvas.width,canvas.height);
    for (let y=0;y<canvas.height;y++) {
      const importance=10 ** (Y.max - y/(canvas.height-1)*(Y.max-Y.min));
      for(let x=0;x<canvas.width;x++) {
        const delta=X.min+x/(canvas.width-1)*(X.max-X.min);
        const t=(Math.log10(importance*multiplier(delta))-riskLow)/(riskHigh-riskLow);
        const rgb=color(coolwarm,t); const offset=(y*canvas.width+x)*4;
        image.data.set([...rgb,220],offset);
      }
    }
    ctx.putImageData(image,0,0);
    plot.append(make('image',{x:X.left,y:Y.top,width:X.right-X.left,height:Y.bottom-Y.top,href:canvas.toDataURL(),preserveAspectRatio:'none'}));
    const curves=make('g',{'clip-path':'url(#f2-plot-clip)'});
    for(let power=riskLow;power<=riskHigh;power++) curves.append(make('path',{d:curve(10**power),fill:'none',stroke:'#344b57','stroke-width':.6,opacity:.18}));
    plot.append(curves);
    for (let x = Math.ceil(X.min); x <= X.max; x += 2) {
      plot.append(make('line',{x1:px(x),x2:px(x),y1:Y.top,y2:Y.bottom,class:'grid-line'}));
      plot.append(make('text',{x:px(x),y:Y.bottom+23,class:'tick','text-anchor':'middle'},x));
    }
    const superscript = value => String(value).split('').map(char=>char==='-'?'⁻':'⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(char)]).join('');
    for(let power=Math.ceil(Y.min);power<=Math.floor(Y.max);power++) {
      const y=py(10**power);
      plot.append(make('line',{x1:X.left,x2:X.right,y1:y,y2:y,class:'grid-line'}));
      plot.append(make('text',{x:X.left-12,y:y+4,class:'tick','text-anchor':'end'},`10${superscript(power)}`));
    }
    plot.append(make('path',{d:`M${X.left},${Y.top}V${Y.bottom}H${X.right}`,fill:'none',stroke:'#455d65','stroke-width':1}));
    plot.append(make('text',{x:(X.left+X.right)/2,y:483,class:'axis-label','text-anchor':'middle'},'Local distortion bound δ'));
    plot.append(make('text',{transform:`translate(22 ${(Y.top+Y.bottom)/2}) rotate(-90)`,class:'axis-label','text-anchor':'middle'},'Global importance p̂'));
    const reference=make('path',{class:'reference-boundary',d:curve(.01)});
    const boundary=make('path',{class:'selected-boundary',id:'f2-boundary'});
    const clipped=make('g',{'clip-path':'url(#f2-plot-clip)'}); clipped.append(reference,boundary);plot.append(clipped);
    const boundaryLabel=make('text',{class:'boundary-label','text-anchor':'start',id:'f2-boundary-label'});plot.append(boundaryLabel);
    const pointLayer=make('g');plot.append(pointLayer);
    const errorColor=error=>`rgb(${color(viridis,(Math.log10(error)-Math.log10(errorMin))/(Math.log10(errorMax)-Math.log10(errorMin))).join(',')})`;
    const pointElements=new Map();
    points.forEach(point=>{
      const group=make('g',{class:'plot-point',id:`f2-point-${point.id}`,transform:`translate(${px(point.delta)},${py(point.importance)})`,role:'button',tabindex:point.id===selected.id?0:-1,'aria-pressed':String(point.id===selected.id)});
      group.append(make('circle',{r:13,fill:'transparent',class:'point-hit'}));
      group.append(make('circle',{r:6,class:'point-marker point-circle',fill:errorColor(point.output_error)}));
      group.append(make('path',{d:'M0,-8.3L8,6.1L-8,6.1Z',class:'point-marker point-triangle',fill:errorColor(point.output_error)}));
      group.append(make('circle',{r:11.5,class:'point-ring'}));
      const title=make('title');group.append(title);
      const hint=()=>{get('f2-hint').textContent=`Cluster ${point.id} · keys ${point.start}–${point.end} · risk ${fmt(point.risk,5)} · ${point.risk>tau?'expand':'represent'}`;};
      group.addEventListener('mouseenter',hint);
      group.addEventListener('focus',hint);
      group.addEventListener('mouseleave',()=>{get('f2-hint').textContent='Select any point to inspect its 10 keys. Arrow keys move between clusters.';});
      group.addEventListener('click',event=>{if(event.detail===0)selectPoint(point,true);});
      group.addEventListener('keydown',event=>{
        if(event.key==='Enter'||event.key===' '){event.preventDefault();selectPoint(point,false);}
        const movement={ArrowRight:1,ArrowDown:1,ArrowLeft:-1,ArrowUp:-1}[event.key];
        if(movement){event.preventDefault();const next=points[(points.indexOf(point)+movement+points.length)%points.length];selectPoint(next,false);pointElements.get(next.id).group.focus();}
      });
      pointElements.set(point.id,{group,title});pointLayer.append(group);
    });
    // Resolve pointer clicks by distance, so overlapping hit areas cannot hide a cluster.
    // Pointer events retain subpixel coordinates; click events may round them.
    let pointerRelease = null;
    listen(plot,'pointerup',event=>{
      if(event.button===0)pointerRelease={x:event.clientX,y:event.clientY,time:event.timeStamp};
    });
    listen(plot,'click',event=>{
      if(event.detail===0)return;
      const matrix=plot.getScreenCTM();
      if(!matrix)return;
      const useRelease=pointerRelease&&event.timeStamp-pointerRelease.time<250&&Math.hypot(pointerRelease.x-event.clientX,pointerRelease.y-event.clientY)<2;
      const local=new DOMPoint(useRelease?pointerRelease.x:event.clientX,useRelease?pointerRelease.y:event.clientY).matrixTransform(matrix.inverse());
      pointerRelease=null;
      const nearest=points.reduce((best,point)=>{
        const distance=Math.hypot(px(point.delta)-local.x,py(point.importance)-local.y);
        return distance<best.distance?{point,distance}:best;
      },{point:null,distance:Infinity});
      if(nearest.distance<=Math.max(14,16/matrix.a))selectPoint(nearest.point,true);
    });
    get('f2-risk-low').textContent=fmt(10**riskLow);
    get('f2-risk-high').textContent=fmt(10**riskHigh);
    get('f2-error-low').textContent=fmt(errorMin);
    get('f2-error-high').textContent=fmt(errorMax);
    const select=get('f2-cluster-select');
    select.replaceChildren();
    points.forEach(point=>{
      const option=document.createElement('option');option.value=point.id;
      option.textContent=`Cluster ${point.id} · keys ${point.start}–${point.end}`;
      select.append(option);
    });
    listen(select,'change',()=>selectPoint(points.find(p=>p.id===Number(select.value)),false));
    listen(get('f2-prev'),'click',()=>selectPoint(points[(points.indexOf(selected)-1+points.length)%points.length],false));
    listen(get('f2-next'),'click',()=>selectPoint(points[(points.indexOf(selected)+1)%points.length],false));

    const qk=get('f2-qk');
    function renderScores() {
      // Position encodes the signed score: farther right always means higher.
      // Every cluster shares this linear scale and every token uses an equal-sized dot.
      const chart={left:55,right:308,top:27,bottom:247};
      const x=score=>chart.left+(score-scoreMin)/(scoreMax-scoreMin)*(chart.right-chart.left);
      qk.replaceChildren(make('title',{id:'f2-qk-title'},`QK scores for keys ${selected.start}–${selected.end}`));
      qk.append(make('text',{x:0,y:14,class:'qk-axis'},'Key'));
      qk.append(make('text',{x:388,y:14,class:'qk-axis','text-anchor':'end'},'Score'));
      for(let tick=Math.ceil(scoreMin/4)*4;tick<=scoreMax;tick+=4) {
        qk.append(make('line',{x1:x(tick),x2:x(tick),y1:chart.top,y2:chart.bottom,stroke:tick===0?'#7e9b8e':'#d4e1d8','stroke-width':tick===0?1.2:.7}));
        qk.append(make('text',{x:x(tick),y:267,class:'qk-axis','text-anchor':'middle'},tick));
      }
      const mean=selected.scores.reduce((sum,value)=>sum+value,0)/10;
      const tokenGroups=[];
      const showToken=i=>{
        tokenGroups.forEach((group,index)=>group.classList.toggle('is-highlighted',i===index));
        get('f2-token-readout').textContent=`K${selected.start+i} · ${JSON.stringify(visibleToken(selected.tokens[i]))} · QK ${selected.scores[i].toFixed(6)}`;
      };
      selected.scores.forEach((score,i)=>{
        const y=38+i*22;
        const group=make('g',{class:'qk-token',tabindex:'0',role:'button','aria-label':`Key ${selected.start+i}, token ${visibleToken(selected.tokens[i])}, QK score ${score.toFixed(6)}`});
        group.append(make('rect',{x:0,y:y-10,width:390,height:20,rx:3,fill:'transparent',class:'qk-hit'}));
        group.append(make('line',{x1:chart.left,x2:chart.right,y1:y,y2:y,class:'qk-row-guide','pointer-events':'none'}));
        group.append(make('text',{x:0,y:y+4,class:'qk-axis','pointer-events':'none'},`K${selected.start+i}`));
        group.append(make('circle',{cx:x(score),cy:y,r:4.2,class:'qk-dot','pointer-events':'none'}));
        group.append(make('text',{x:388,y:y+4,class:'qk-value','text-anchor':'end','pointer-events':'none'},score.toFixed(3)));
        group.append(make('title',{},`K${selected.start+i} ${JSON.stringify(selected.tokens[i])}\nQK: ${score.toFixed(6)}\nFull-context attention: ${(selected.attention[i]*100).toFixed(6)}%`));
        group.addEventListener('mouseenter',()=>showToken(i));
        group.addEventListener('focus',()=>showToken(i));
        group.addEventListener('click',()=>showToken(i));
        group.addEventListener('keydown',event=>{if(event.key===' '||event.key==='Enter'){event.preventDefault();showToken(i);}});
        tokenGroups.push(group);qk.append(group);
      });
      qk.append(make('text',{x:(chart.left+chart.right)/2,y:295,class:'qk-axis','text-anchor':'middle'},'Lower ← QK score → Higher'));
      showToken(selected.scores.indexOf(Math.max(...selected.scores)));
      get('f2-score-summary').textContent=`Mean ${mean.toFixed(3)} · score range ${(Math.max(...selected.scores)-Math.min(...selected.scores)).toFixed(3)} · shared axis across samples`;
      const rows=get('f2-token-rows');rows.replaceChildren();
      selected.scores.forEach((score,i)=>{
        const row=document.createElement('tr');
        [String(selected.start+i),visibleToken(selected.tokens[i]),score.toFixed(6),`${(selected.attention[i]*100).toFixed(4)}%`].forEach((value,index)=>{
          const cell=document.createElement(index===0?'th':'td');
          if(index===0)cell.scope='row';cell.textContent=value;cell.title=value;row.append(cell);
        });rows.append(row);
      });
    }
    function renderDecision() {
      const expanded=selected.risk>tau;
      get('f2-decision').textContent=expanded?'EXPAND':'REPRESENT';
      get('f2-decision').classList.toggle('expanded',expanded);
      get('f2-rule').textContent=`R = ${fmt(selected.risk,6)} ${expanded?'>':'≤'} τ = ${fmt(tau,3)}`;
    }
    function selectPoint(point,scroll) {
      selected=point;selectedClusterId=point.id;select.value=String(point.id);
      get('f2-cluster-title').textContent=`Keys ${point.start}–${point.end}`;
      get('f2-importance').textContent=fmt(point.importance,5);
      get('f2-distortion').textContent=point.delta.toFixed(3);
      get('f2-risk').textContent=fmt(point.risk,5);
      get('f2-error').textContent=fmt(point.output_error,5);
      get('f2-hint').textContent=`Selected: cluster ${point.id} · keys ${point.start}–${point.end} · risk ${fmt(point.risk,5)}`;
      for(const [id,{group}] of pointElements){group.setAttribute('aria-pressed',String(id===point.id));group.setAttribute('tabindex',id===point.id?'0':'-1');}
      pointLayer.append(pointElements.get(point.id).group);
      renderScores();renderDecision();
      if(scroll&&matchMedia('(max-width:800px)').matches)document.querySelector('.f2-detail').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
    }
    function updateTau() {
      tau=10**Number(get('f2-tau').value);
      get('f2-tau-value').textContent=fmt(tau,3);
      get('f2-tau').setAttribute('aria-valuetext',`Risk threshold ${fmt(tau,3)}`);
      root.dataset.tau=tau;
      boundary.setAttribute('d',curve(tau));
      reference.setAttribute('d',curve(Math.abs(Math.log10(tau)+2)<.01?.1:.01));
      const labelCandidates=Array.from({length:80},(_,i)=>X.min+.12*(X.max-X.min)+i/79*.6*(X.max-X.min));
      const lx=labelCandidates.find(x=>py(boundaryY(x))>Y.top+25&&py(boundaryY(x))<Y.bottom-20);
      boundaryLabel.style.display=lx===undefined?'none':'';
      if(lx!==undefined){boundaryLabel.setAttribute('x',px(lx));boundaryLabel.setAttribute('y',py(boundaryY(lx))-10);boundaryLabel.textContent=`τ = ${fmt(tau,3)}`;}
      let expandedCount=0;
      points.forEach(point=>{
        const expanded=point.risk>tau;expandedCount+=Number(expanded);
        const {group,title}=pointElements.get(point.id);
        group.dataset.decision=expanded?'expand':'represent';
        group.querySelector('.point-circle').style.display=expanded?'none':'';
        group.querySelector('.point-triangle').style.display=expanded?'':'none';
        group.setAttribute('aria-label',`Cluster ${point.id}, keys ${point.start} to ${point.end}, risk ${fmt(point.risk,5)}, ${expanded?'expand':'represent'}`);
        title.textContent=`Keys ${point.start}–${point.end}\nImportance ${fmt(point.importance,5)}\nDistortion ${point.delta.toFixed(3)}\nRisk ${fmt(point.risk,6)}\n${expanded?'Expand':'Represent'} at τ = ${fmt(tau,3)}`;
      });
      get('f2-count').textContent=`${expandedCount} expand · ${points.length-expandedCount} represent`;
      const status=`τ = ${fmt(tau,3)}: ${expandedCount} of ${points.length} clusters expand; ${points.length-expandedCount} use representatives. QK scores stay fixed.`;
      get('f2-status').textContent=status;
      document.querySelectorAll('.f2-presets [data-tau]').forEach(button=>button.setAttribute('aria-pressed',String(Math.abs(Math.log10(Number(button.dataset.tau))-Math.log10(tau))<.005)));
      renderDecision();
    }
    let animationId=0;
    let lastTime=0;
    let direction=1;
    let animationValue=Number(get('f2-tau').value);
    function stopAnimation() {
      cancelAnimationFrame(animationId);animationId=0;lastTime=0;
      get('f2-play').replaceChildren(Object.assign(document.createElement('span'),{textContent:'▷'}),document.createTextNode(' Animate τ'));
      get('f2-play').setAttribute('aria-pressed','false');
      get('f2-status').setAttribute('aria-live','polite');
    }
    function animate(time) {
      if(lastTime){
        animationValue+=direction*Math.min(time-lastTime,100)/1600;
        if(animationValue>1){animationValue=1;direction=-1;}if(animationValue< -4){animationValue=-4;direction=1;}
        get('f2-tau').value=animationValue;updateTau();
      }
      lastTime=time;animationId=requestAnimationFrame(animate);
    }
    listen(get('f2-play'),'click',()=>{
      if(animationId){stopAnimation();return;}
      get('f2-play').textContent='Ⅱ Pause';get('f2-play').setAttribute('aria-pressed','true');
      get('f2-status').setAttribute('aria-live','off');lastTime=0;animationValue=Number(get('f2-tau').value);animationId=requestAnimationFrame(animate);
    });
    listen(get('f2-tau'),'input',()=>{stopAnimation();updateTau();});
    document.querySelectorAll('.f2-presets [data-tau]').forEach(button=>listen(button,'click',()=>{stopAnimation();get('f2-tau').value=Math.log10(Number(button.dataset.tau));updateTau();}));
    listen(document,'visibilitychange',()=>{if(document.hidden)stopAnimation();});
    const visibilityObserver = new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stopAnimation();});
    visibilityObserver.observe(root);
    selectPoint(selected,false);updateTau();
    return () => {
      stopAnimation();
      listeners.forEach(remove => remove());
      visibilityObserver.disconnect();
    };
  }

  function switchSample(index) {
    releaseSample();
    activeSampleIndex = index;
    releaseSample = renderSample(samples[index], index);
    document.querySelectorAll('[data-f2-sample]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.f2Sample) === index)));
    document.querySelectorAll('[data-f2-axes]').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.f2Axes === (sharedAxes?'shared':'fit'))));
  }
  document.querySelectorAll('[data-f2-sample]').forEach(button => button.addEventListener('click', () => switchSample(Number(button.dataset.f2Sample))));
  document.querySelectorAll('[data-f2-axes]').forEach(button => button.addEventListener('click', () => {
    sharedAxes = button.dataset.f2Axes === 'shared';
    switchSample(activeSampleIndex);
  }));
  switchSample(0);
})();
