const {readFileSync}=require('node:fs');
const {runInNewContext}=require('node:vm');
const assert=require('node:assert/strict');
function boot(supported=true) {
  const elements=[];
  const make=()=>({value:'Bring the tallest object',disabled:false,events:{},textContent:'',
    setAttribute(){},insertAdjacentElement(){},closest(){return {appendChild(){}}},focus(){},
    addEventListener(k,f){(this.events[k]??=[]).push(f)},
    fire(k,e={}){for(const f of this.events[k]||[])f(e)},click(){this.fire('click');}});
  const input=make(),run=make();let calls=0,rec;
  run.click=()=>{run.fire('click');calls++};
  class Speech {constructor(){rec=this}start(){}stop(){this.onend()}abort(){}}
  const window={addEventListener(){},...(supported?{SpeechRecognition:Speech}:{})};
  const document={getElementById:id=>id==='cmd'?input:run,createElement(){const e=make();elements.push(e);return e}};
  runInNewContext(readFileSync('engine/voice_input.js','utf8'),{window,document});
  return {input,run,button:elements[0],status:elements[1],get rec(){return rec},get calls(){return calls},
    enter(){input.fire('keydown',{key:'Enter',preventDefault(){}})}};
}
let app=boot();app.button.click();app.rec.onresult({results:[[{transcript:'bring the blue cube'}]]});
assert.equal(app.input.value,'bring the blue cube');assert.equal(app.calls,0);app.enter();assert.equal(app.calls,1);
app=boot();app.button.click();app.rec.onerror({error:'not-allowed'});app.rec.onend();assert.equal(app.calls,0);assert.match(app.status.textContent,/denied/);
app=boot();app.button.click();app.enter();assert.equal(app.calls,0);assert.equal(app.input.value,'Bring the tallest object');
app=boot(false);app.button.click();assert.match(app.status.textContent,/unavailable/);app.input.value='bring the red ball';app.enter();assert.equal(app.calls,1);
app=boot();app.run.disabled=true;app.enter();assert.equal(app.calls,0);
app=boot();app.button.click();app.input.value='my edited draft';app.input.fire('input');app.rec.onresult({results:[[{transcript:'late result'}]]});assert.equal(app.input.value,'my edited draft');
console.log('Voice input: transcript draft, explicit Enter, denied permission, silence, unsupported browser, lock, and late-result checks passed.');
