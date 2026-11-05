'use strict';
const $ = (s) => document.querySelector(s);
const titles = { bolt_flange: 'Bolt flange', counterbored_plate: 'Counterbored plate', stepped_bore: 'Stepped bore' };
let data, viewer;
function toast(text) { const el = $('.toast'); el.textContent = text; el.classList.add('visible'); clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove('visible'), 2200); }
$('.menu')?.addEventListener('click', () => { const open = $('#nav').classList.toggle('open'); $('.menu').setAttribute('aria-expanded', String(open)); });
$('#nav')?.addEventListener('click', (event) => { if (event.target.closest('a')) { $('#nav').classList.remove('open'); $('.menu').setAttribute('aria-expanded', 'false'); } });
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const value = document.getElementById(button.dataset.copy)?.textContent;
  if (!value) return;
  try { await navigator.clipboard.writeText(value); toast('Copied to clipboard'); } catch { toast('Clipboard unavailable. Select the code to copy.'); }
}));

class CADViewer {
  constructor(canvas) {
    this.canvas = canvas; this.gl = canvas.getContext('webgl', { antialias: true, alpha: true });
    if (!this.gl) throw new Error('WebGL is unavailable. Reference geometry is listed below.');
    const gl = this.gl;
    const vertex = `attribute vec3 a_position; attribute vec3 a_normal; uniform float u_yaw,u_pitch,u_scale,u_aspect; varying vec3 v_normal; vec3 turn(vec3 p){float c=cos(u_yaw),s=sin(u_yaw); vec3 q=vec3(p.x*c-p.y*s,p.x*s+p.y*c,p.z); return vec3(q.x,q.y*cos(u_pitch)+q.z*sin(u_pitch),-q.y*sin(u_pitch)+q.z*cos(u_pitch));} void main(){vec3 p=turn(a_position);v_normal=turn(a_normal);gl_Position=vec4(p.x*u_scale*u_aspect,p.y*u_scale-0.09,-p.z*0.24,1.0);}`;
    const fragment = `precision mediump float; varying vec3 v_normal; uniform vec3 u_color; uniform float u_shade; void main(){vec3 n=normalize(v_normal+vec3(0.00001));float light=0.48+0.42*max(dot(n,normalize(vec3(-0.5,0.7,1.2))),0.0)+0.12*max(dot(n,normalize(vec3(0.7,-0.2,0.3))),0.0);gl_FragColor=vec4(u_color*mix(1.0,light,u_shade),1.0);}`;
    const compile = (kind, source) => { const shader = gl.createShader(kind); gl.shaderSource(shader, source); gl.compileShader(shader); if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader)); return shader; };
    this.program = gl.createProgram(); gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertex)); gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('3D shader initialization failed');
    gl.useProgram(this.program); this.uniforms = Object.fromEntries(['yaw','pitch','scale','aspect','color','shade'].map(name => [name, gl.getUniformLocation(this.program, 'u_'+name)]));
    this.position = gl.getAttribLocation(this.program, 'a_position'); this.normal = gl.getAttribLocation(this.program, 'a_normal'); this.wire = false;
    this.reset(); let last;
    canvas.addEventListener('pointerdown', event => { last = [event.clientX,event.clientY]; canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener('pointermove', event => { if (!last) return; this.yaw += (event.clientX-last[0])*0.009; this.pitch = Math.max(0.12,Math.min(1.5,this.pitch+(event.clientY-last[1])*0.006)); last=[event.clientX,event.clientY]; this.draw(); });
    const release = () => { last = null; }; canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('wheel', event => { event.preventDefault(); this.zoom(event.deltaY>0 ? .94 : 1.06); }, { passive: false });
    canvas.addEventListener('keydown', event => { if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) { event.preventDefault(); this.yaw += event.key==='ArrowLeft'?-.1:event.key==='ArrowRight'?.1:0; this.pitch=Math.max(.12,Math.min(1.5,this.pitch+(event.key==='ArrowUp'?.1:event.key==='ArrowDown'?-.1:0))); this.draw(); } });
    this.observer = new ResizeObserver(() => this.draw()); this.observer.observe(canvas);
  }
  reset() { this.yaw = -.62; this.pitch = .96; this.scale = 1.08; this.draw(); }
  zoom(factor) { this.scale = Math.max(.55,Math.min(1.8,this.scale*factor)); this.draw(); }
  buffer(vertices,normals) { const gl=this.gl; const a=gl.createBuffer(),b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,a); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW); gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(normals),gl.STATIC_DRAW); return {a,b,count:vertices.length/3}; }
  dispose(shape) { if(shape){this.gl.deleteBuffer(shape.a);this.gl.deleteBuffer(shape.b);} }
  setMesh(mesh) {
    for(const key of ['triangles','edges','allEdges','grid']) this.dispose(this[key]);
    const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity]; for(const v of mesh.vertices) for(let i=0;i<3;i++){min[i]=Math.min(min[i],v[i]);max[i]=Math.max(max[i],v[i]);}
    const span=Math.max(...max.map((v,i)=>v-min[i])); if(!Number.isFinite(span)||span<=0) throw new Error('Invalid reference mesh');
    const vertices=mesh.vertices.map(v=>v.map((x,i)=>(x-(min[i]+max[i])/2)*1.65/span)); const positions=[],normals=[],edgeMap=new Map(),all=[];
    const key=v=>v.map(x=>Math.round(x*100000)).join(',');
    for(const face of mesh.faces){const a=vertices[face[0]],b=vertices[face[1]],c=vertices[face[2]],ab=b.map((v,i)=>v-a[i]),ac=c.map((v,i)=>v-a[i]);let n=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];const length=Math.hypot(...n);if(!length)continue;n=n.map(x=>x/length); for(const v of [a,b,c]){positions.push(...v);normals.push(...n);} for(const [p,q] of [[a,b],[b,c],[c,a]]){all.push(...p,...q);const id=[key(p),key(q)].sort().join('/');const entry=edgeMap.get(id);if(entry){entry.crease=entry.crease||entry.n.reduce((s,v,i)=>s+v*n[i],0)<.97;entry.shared=true;}else edgeMap.set(id,{p,q,n,shared:false,crease:false});}}
    const edges=[];for(const e of edgeMap.values())if(!e.shared||e.crease)edges.push(...e.p,...e.q);
    const grid=[],z=(min[2]-(min[2]+max[2])/2)*1.65/span-.018;for(let i=-7;i<=7;i++){const p=i*.22;grid.push(-1.54,p,z,1.54,p,z,p,-1.54,z,p,1.54,z);}
    this.triangles=this.buffer(positions,normals);this.edges=this.buffer(edges,edges.map(()=>0));this.allEdges=this.buffer(all,all.map(()=>0));this.grid=this.buffer(grid,grid.map(()=>0));this.draw();
  }
  renderShape(shape,mode,color,shade){const gl=this.gl;if(!shape)return;gl.bindBuffer(gl.ARRAY_BUFFER,shape.a);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(this.position);gl.bindBuffer(gl.ARRAY_BUFFER,shape.b);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(this.normal);gl.uniform3fv(this.uniforms.color,color);gl.uniform1f(this.uniforms.shade,shade);gl.drawArrays(mode,0,shape.count);}
  draw(){if(!this.gl||!this.program)return;const gl=this.gl,ratio=Math.min(window.devicePixelRatio||1,2),w=Math.round(this.canvas.clientWidth*ratio),h=Math.round(this.canvas.clientHeight*ratio);if(!w||!h)return;this.canvas.width=w;this.canvas.height=h;gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);gl.uniform1f(this.uniforms.yaw,this.yaw);gl.uniform1f(this.uniforms.pitch,this.pitch);gl.uniform1f(this.uniforms.scale,this.scale*Math.min(1,w/h)*0.94);gl.uniform1f(this.uniforms.aspect,h/w);gl.enable(gl.DEPTH_TEST);this.renderShape(this.grid,gl.LINES,[.80,.84,.76],0);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);this.renderShape(this.triangles,gl.TRIANGLES,[.52,.62,.43],1);gl.disable(gl.POLYGON_OFFSET_FILL);this.renderShape(this.wire?this.allEdges:this.edges,gl.LINES,[.18,.29,.16],0);}
}

async function selectPart(family) {
  const example=data.examples.find(e=>e.family===family); if(!example)return;
  document.querySelectorAll('.part').forEach(button=>{const active=button.dataset.part===family;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  $('#model-name').textContent=titles[family]||family;$('#model-subtitle').textContent='FreeCAD · validated TRAIN reference';
  $('#trajectory-code').textContent=example.actions.map((action,index)=>String(index+1).padStart(2,'0')+'  '+JSON.stringify(action)).join('\n');$('#action-count').textContent=example.actions.length+' ACTIONS · NATIVE-VERIFIED TEACHER';
  if(viewer){const token=selectPart.token=Symbol();try{const response=await fetch(example.mesh);if(!response.ok)throw new Error('Mesh unavailable');const mesh=await response.json();if(selectPart.token===token)viewer.setMesh(mesh);}catch(error){toast(error.message);}}
}
async function init(){
  try{const response=await fetch('data.json');if(!response.ok)throw new Error('Project snapshot unavailable');data=await response.json();
    const results=$('#results');for(const result of data.results){const before=100*result.before/result.tasks,after=100*result.after/result.tasks;const row=document.createElement('article');row.className='result-row';row.innerHTML=`<div><h3>${result.engine==='freecad'?'FreeCAD':'CadQuery'}</h3><strong><small>${before.toFixed(1)}%</small>${after.toFixed(1)}%</strong></div><div class="bar-track"><span class="bar-before" style="width:${before}%"></span><span class="bar-after" style="width:${after}%"></span></div><span class="mono">${result.before} → ${result.after} / ${result.tasks} TASKS · SEQUENTIAL GREEDY EVALUATION</span>`;results.append(row);}
    $('#eval-status').textContent=data.evaluation_status;if(data.evaluation_status.startsWith('Validation complete')){$('.evaluation-note').classList.add('complete');$('.evaluation-note .dot').classList.remove('amber');}
    for(const result of data.retention||[]){const cell=document.createElement('div');const label=document.createElement('span');label.textContent=result.name;const value=document.createElement('strong');value.textContent=result.before+' → '+result.after+' / '+result.tasks;const profile=document.createElement('small');profile.textContent=result.execution;cell.append(label,value,profile);$('#retention-summary').append(cell);}
    $('#snapshot-time').textContent='RECORDED SNAPSHOT / '+new Date(data.updated).toLocaleString('en-GB',{dateStyle:'medium',timeStyle:'short',timeZone:'Asia/Yekaterinburg'})+' UTC+5 · Not live telemetry';
    try{viewer=new CADViewer($('#cad-view'));}catch(error){const fallback=document.createElement('p');fallback.className='viewer-error';fallback.textContent=error.message;$('.viewport').append(fallback);}
    await selectPart('bolt_flange');
  }catch(error){$('#trajectory-code').textContent='Run a local HTTP server to load the project snapshot.\nSee web/README.md.';toast(error.message);}
}
document.querySelectorAll('[data-part]').forEach(button=>button.addEventListener('click',()=>selectPart(button.dataset.part)));
$('#wireframe')?.addEventListener('click',()=>{if(!viewer)return;viewer.wire=!viewer.wire;$('#wireframe').setAttribute('aria-pressed',String(viewer.wire));viewer.draw();});
$('#zoom-in')?.addEventListener('click',()=>viewer?.zoom(1.12));$('#zoom-out')?.addEventListener('click',()=>viewer?.zoom(1/1.12));$('#reset-view')?.addEventListener('click',()=>viewer?.reset());
init();
