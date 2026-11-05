const {readFileSync}=require('node:fs'), {runInNewContext}=require('node:vm');
const assert=require('node:assert/strict');
function boot(denied=false){
 let document;
 class Element {
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.attrs={};this.classes=new Set();this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x)};this.hidden=false;this.open=false;}
  setAttribute(k,v){this.attrs[k]=v;}
  append(...nodes){for(const n of nodes){if(n.parent)n.parent.children.splice(n.parent.children.indexOf(n),1);this.children.push(n);n.parent=this;}}
  before(n){n.parent=this.parent;this.parent.children.splice(this.parent.children.indexOf(this),0,n);}
  replaceWith(n){const parent=this.parent;if(n.parent)n.parent.children.splice(n.parent.children.indexOf(n),1);parent.children[parent.children.indexOf(this)]=n;n.parent=parent;this.parent=null;}
  replaceChildren(){this.children=[];}
  closest(){return this.parent;}
  addEventListener(k,f){(this.events[k]??=[]).push(f);}
  async fire(k,event={}){for(const f of this.events[k]||[])await f(event);}
  focus(){document.activeElement=this;}
  showModal(){this.open=true;}
  close(){this.open=false;}
 }
 const body=new Element('body'),header=new Element('header'),main=new Element('main'),root=new Element('html');
 const ids={};body.append(header,main);
 for(const id of ['pov','map','cinematic','alog']){const panel=new Element('section'),el=new Element('view');el.id=id;ids[id]=el;panel.append(el);main.append(panel);}
 ids.cmd=new Element('input');main.append(ids.cmd);
 const original=[...main.children];
 function find(n,id){if(n.id===id)return n;for(const c of n.children){const got=find(c,id);if(got)return got;}}
 document={body,documentElement:root,activeElement:ids.cmd,fullscreenElement:null,events:{},getElementById:id=>ids[id]||find(body,id),querySelector:()=>header,createElement:t=>new Element(t),createComment:()=>new Element('comment'),addEventListener:Element.prototype.addEventListener,fire:Element.prototype.fire};
 root.requestFullscreen=async()=>{if(denied)throw Error('denied');document.fullscreenElement=root;await document.fire('fullscreenchange');};
 document.exitFullscreen=async()=>{document.fullscreenElement=null;await document.fire('fullscreenchange');};
 const window={};runInNewContext(readFileSync('engine/all_views.js','utf8'),{document,window});
 return {document,window,ids,main,original,button:find(body,'allViewsButton'),overlay:find(body,'allViews')};
}
(async()=>{
 for(const denied of [false,true]){
  const a=boot(denied);await a.button.fire('click');
  assert.equal(a.overlay.hidden,false);assert.deepEqual(a.overlay.children.slice(1).map(p=>p.children[0].id),['pov','map','cinematic','alog']);
  assert.equal(a.ids.pov.parent,a.overlay.children[1]);
  await a.document.fire('keydown',{key:'Escape',preventDefault(){}});
  assert.equal(a.overlay.hidden,true);assert.deepEqual(a.main.children,a.original);
  await a.button.fire('click');await a.overlay.children[0].children[1].fire('click');
  assert.deepEqual(a.main.children,a.original);assert.equal(a.document.fullscreenElement,null);
 }
 const a=boot();await a.button.fire('click');await a.document.exitFullscreen();assert.deepEqual(a.main.children,a.original);
 a.window.showSelectionIssue({code:'ambiguous',message:'2 purple cones. Please be specific.',candidates:['small purple cone','large purple cone']});
 const d=a.document.getElementById('selectionIssue');assert(d.open);assert.equal(d.children[2].children.length,2);await d.children[3].fire('click');assert(!d.open);assert.equal(a.document.activeElement,a.ids.cmd);
 console.log('All views: existing streams reused, three views + logs, native/fallback exit, Escape, restore order and clarification editing passed.');
})().catch(e=>{console.error(e);process.exitCode=1});
