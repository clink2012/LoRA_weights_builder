'use strict';

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const option = (id, title, description, visual = '') => ({id,title,description,visual});
const swatches = (colors) => `<span class="option-swatch" aria-hidden="true">${colors.map(c=>`<i style="background:${c}"></i>`).join('')}</span>`;
const miniLayout = (type) => `<span class="option-mini mini-${type}" aria-hidden="true"><span class="mini-sidebar"></span><span class="mini-body">${[0,1,2].map(()=>'<span class="mini-col"><i class="mini-line"></i><i class="mini-line"></i><i class="mini-line"></i></span>').join('')}</span></span>`;
const miniBars = () => '<span class="mini-visual-bars" aria-hidden="true">'+[14,22,12,24,17,11,8,19,25,16,21,12,7,10,19,22,17,11,6,14].map(h=>`<i style="height:${h}px"></i>`).join('')+'</span>';
const miniHeat = () => '<span class="mini-visual-heat" aria-hidden="true">'+Array.from({length:24},(_,i)=>`<i style="opacity:${.15+(i%7)/9}"></i>`).join('')+'</span>';
const miniRows = () => '<span class="mini-visual-rows" aria-hidden="true">'+[70,45,85].map(v=>`<i style="--fill:${v}%"></i>`).join('')+'</span>';

const steps = [
  {name:'Look & feel',title:'A workspace with personality.',intro:'Start with the atmosphere. Each direction changes the live app example. A preview starts in Prism; that is not a chosen answer.',questions:[
    {id:'direction',title:'Which design direction feels closest?',help:'Choose a starting point. We can blend the details later.',options:[
      option('prism','Prism · violet glass','Layered plum panels, violet light and cyan details. Expressive without hiding the controls.',swatches(['#161022','#2b2240','#b291f6','#67dae7'])),
      option('orbit','Orbit · deep blue','A blue creative studio with crisp luminous controls and a calmer, spacious feel.',swatches(['#121b34','#233352','#818aff','#57c9ff'])),
      option('atelier','Atelier · light & emerald','Soft white surfaces, warm spacing and elegant green details. Clear and less visually intense.',swatches(['#eef1ec','#fbfcf9','#2c805d','#47a996'])),
      option('carbon','Carbon · graphite & mint','A focused instrument panel with charcoal surfaces and restrained mint highlights.',swatches(['#111517','#232e32','#75d8c9','#b8b1ff']))]},
    {id:'palette',title:'Which accent colours would you like?',help:'Try a different accent within the design above. LoRA rows also use text labels, so colour is never the only cue.',options:[
      option('violet','Violet + cyan','Creative, cool and bright.',swatches(['#b291f6','#67dae7','#f0a9a3'])),
      option('blue','Ice blue + lilac','Clean, technical and luminous.',swatches(['#83a9ff','#71e1fa','#caa8fd'])),
      option('mint','Mint + warm gold','Calm, fresh and warm.',swatches(['#83d9b6','#e6be87','#bca7ef'])),
      option('ember','Apricot + lavender','A warmer creative palette.',swatches(['#efb287','#c7b0ed','#8fd1c6']))]}
  ]},
  {name:'Workspace',title:'How should the app flow?',intro:'Imagine choosing three LoRAs, comparing their individual blocks and copying each result into its loader. Try the arrangements to see what suits you.',questions:[
    {id:'layout',title:'Which arrangement feels easiest?',options:[
      option('studio','Studio · library beside the editor','Keep the selected LoRAs in view while inspecting one full block vector.',miniLayout('studio')),
      option('guided','Guided · one stage at a time','A clear Choose → Inspect → Copy path, with fewer competing areas.',miniLayout('guided')),
      option('lanes','Compare · one lane per LoRA','See the sample block patterns together, then open one LoRA for exact edits.',miniLayout('lanes'))]},
    {id:'density',title:'How much should fit on screen?',options:[
      option('roomy','Spacious','Larger panels and more breathing room; more scrolling.'),
      option('balanced','Balanced','Comfortable spacing with the main controls close together.'),
      option('compact','Compact','A denser workspace for comparing more information at once.')]},
    {id:'thumbnails',title:'How should LoRAs look in the library?',help:'This example uses abstract placeholders. Real previews would come from your local library where available.',options:[
      option('small','Small thumbnail + name','A visual cue with the filename and role still prominent.'),
      option('large','Larger preview cards','More visual browsing, using more screen space.'),
      option('none','Names and details','A tidy list for finding familiar files quickly.')]}
  ]},
  {name:'Block controls',title:'Individual blocks, clearly.',intro:'The core output is the complete vector for each LoRA. The sample has 58 slots: BASE, 19 double blocks and 38 single blocks. Other families need their own verified maps.',questions:[
    {id:'block_view',title:'How would you first inspect the blocks?',help:'All three keep the complete comma-separated export. Click a block in the preview to inspect its exact value.',options:[
      option('bars','Bars · see each weight','Compare the height of all block values, split into labelled groups.',miniBars()),
      option('heat','Tiles · a colour map','A numbered grid with intensity indicating each value. Selected blocks show their exact number.',miniHeat()),
      option('curve','Curve + numbered controls','A connected view of the weight pattern, with individual slots underneath.',miniRows())]},
    {id:'editing',title:'How much manual control should be visible?',help:'Default stays unchanged. Edits belong to My variant. Try “Edit a copy” in the preview.',options:[
      option('focused','One block at a time','Select a block, then adjust its exact value. The rest stays uncluttered.'),
      option('table','Exact values available together','Keep a full numbered value grid available for careful edits.'),
      option('advanced','Advanced tools on demand','Start with one block; expand range, group and batch controls when needed.') ]}
  ]},
  {name:'Guidance',title:'Useful advice, with context.',intro:'Block overlap can guide experiments, but a weight graph alone cannot predict image quality. Choose how the app should explain what it knows and what still needs render testing.',questions:[
    {id:'explanations',title:'How much explanation would help?',options:[
      option('plain','Plain advice first','Short suggestions with the reason and uncertainty visible.'),
      option('layered','Short advice + expandable evidence','Start simply, then open block details and supporting information.'),
      option('technical','Detailed inspection view','Show structural details, settings and evidence alongside the advice.')]},
    {id:'comparison',title:'How would you compare stacked LoRAs?',help:'In this design sample, the lines and bars show entered block values only. They are not measured semantic influence or conflict scores.',options:[
      option('patterns','Side-by-side block patterns','Compare each LoRA’s values and locations without a single combined score.'),
      option('overlay','An overlay with separate colours','Place the entered patterns on one chart to inspect the same block positions.'),
      option('notes','A simple explanation beside the vectors','Keep the graphs secondary and read where experiments may be useful.') ]},
    {id:'ab',title:'How should A/B experiments be presented?',help:'A/B are shared values substituted into selected slots. The example ranges are illustrative, not validated model recommendations.',options:[
      option('numeric','Exact numeric values first','Start with the full numeric vector. Keep A/B controls tucked away.'),
      option('guided','A/B with visible bounds','Highlight chosen slots and show A/B values with suggested min/max and their evidence.'),
      option('custom','A/B with custom experiment bounds','Show the suggested bounds and let a personal variant record your own bounds.') ]}
  ]},
  {name:'Save & copy',title:'Keep the good experiments.',intro:'Default is the stable reference. Personal edits become their own entry, so you can always return to it. Choose how to organise variants and the copy area.',questions:[
    {id:'variants',title:'How should your saved experiments be organised?',help:'All options preserve Default, earlier versions and the actual per-block values. This choice changes how history is presented. Saving in this preview is only a demonstration.',options:[
      option('named','Named variants · history tucked away','Recognisable names under each LoRA, with earlier versions available when needed.'),
      option('history','Named variants · history alongside','Keep the change history close, so prior settings are easy to inspect and restore.'),
      option('recipe','Recipes · history per saved stack','Organise each LoRA’s settings by stack and loader order, with earlier recipe versions retained.')]},
    {id:'export',title:'What should the final copy area emphasise?',help:'Every choice includes all required slots. Any separate model/CLIP strength is secondary to the block vector.',options:[
      option('vector','The complete vector first','A large full-value field and an obvious Copy button for each LoRA.'),
      option('loader','A card per loader in chain order','Clearly labelled LoRA, loader number, full vector and supporting settings.'),
      option('checklist','A short transfer checklist','Show which loader each vector belongs in and let you mark it copied.')]}
  ]},
  {name:'Finishing touches',title:'Make it comfortable to use.',intro:'A few details make daily use nicer. You can skip anything you are unsure about and still save the questionnaire.',questions:[
    {id:'motion',title:'How lively should the interface feel?',options:[
      option('subtle','Subtle movement','Gentle state changes and short transitions; nothing continuously moving.'),
      option('off','Mostly still','Instant changes, with clear selection states instead of animation.'),
      option('expressive','A little more expressive','More noticeable transitions when exploring panels; keep editing steady.')]},
    {id:'text_size',title:'How large should everyday text be?',options:[
      option('standard','Standard desktop size','Comfortable labels with room for information.'),
      option('large','Larger and easier to scan','Bigger labels and explanations, taking more space.')]},
    {id:'references',title:'Which of your references should guide us?',help:'Optional: choose any that appeal to you. Click an image to inspect it larger; choose its checkbox to record your preference.',multiple:true,reference:true,options:Array.from({length:7},(_,i)=>option(String(i+1),`Reference ${i+1}`,''))}
  ]},
  {name:'Review',title:'Your design brief.',intro:'Review the choices that felt right. Unanswered questions stay open; they will not be filled in from the preview.',questions:[]}
];
const allQuestions = steps.flatMap(step=>step.questions);
const questionLookup = Object.fromEntries(allQuestions.map(q=>[q.id,q]));
const names = ['Prism studio','Orbit studio','Atelier workspace','Carbon workspace'];
const blocks = [{label:'BASE',group:'base',number:null},...Array.from({length:19},(_,i)=>({label:`D${String(i).padStart(2,'0')}`,group:'double',number:i})),...Array.from({length:38},(_,i)=>({label:`S${String(i).padStart(2,'0')}`,group:'single',number:i}))];
const sampleNames = [{name:'Person example',role:'Identity / person',color:'var(--accent)',symbol:'◎'},{name:'Clothing example',role:'Clothing',color:'var(--accent2)',symbol:'◇'},{name:'Style example',role:'Style / pose',color:'var(--accent3)',symbol:'✳'}];
const sampleDefaults = Array.from({length:3},(_,lora)=>blocks.map((block,index)=>{
  if(index===0)return 1;
  if(lora===0)return block.group==='double'?Number((.35+.4*Math.sin((index+1)*.27)**2).toFixed(2)):Number((.16+.39*Math.cos(index*.16)**2).toFixed(2));
  if(lora===1)return index<20?Number((.12+.45*Math.sin(index*.38)**2).toFixed(2)):Number((.12+.63*Math.sin(index*.12+.7)**2).toFixed(2));
  return index<20?Number((.15+.2*Math.cos(index*.31)**2).toFixed(2)):Number((.07+.35*Math.cos(index*.21)**2).toFixed(2));
}));
const variants = sampleDefaults.map(values=>({values:[...values],tokens:{},edited:false}));
let answers = {}, notes = '', submitted = false, savedAt = null, receipt = null, currentStep = 0;
let currentLora = 0, selectedBlock = 25, currentVariant = 'default', abPreview = false, exportMode = 'numeric';
let abValues = {A:.45,B:.7};
const abBounds = {A:[.2,.8],B:[.3,.9]};
let saveTimer = null, saveChain = Promise.resolve(), revision = 0, savedRevision = -1, submitting = false, ready = false, toastTimer;
let saveError = false, loadError = false;

function getOptionLabel(id,value){const q=questionLookup[id];return q?.options.find(o=>o.id===value)?.title??String(value);}
function validAnswers(input){const result={};if(!input||typeof input!=='object'||Array.isArray(input))return result;for(const q of allQuestions){const value=input[q.id];const permitted=new Set(q.options.map(o=>o.id));if(q.multiple&&Array.isArray(value)){const valid=[...new Set(value)].filter(v=>typeof v==='string'&&permitted.has(v));if(valid.length)result[q.id]=valid;}else if(!q.multiple&&typeof value==='string'&&permitted.has(value)){result[q.id]=value;}}return result;}
function filled(q){return q.multiple?Array.isArray(answers[q.id])&&answers[q.id].length>0:typeof answers[q.id]==='string';}
function completedCount(){return allQuestions.filter(filled).length;}
function setStatus(message,error=false){const el=$('#save-status');el.classList.toggle('error',error);el.textContent=message;if(error){const retry=document.createElement('button');retry.textContent=loadError?'Retry load':'Retry save';retry.type='button';retry.addEventListener('click',()=>{if(loadError)initialize();else queueSave(true).then(()=>renderQuestions()).catch(()=>{});});el.append(retry);}}
function timeLabel(time){if(!time)return '';const date=new Date(time);return Number.isNaN(date.getTime())?'':date.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',timeZone:'Europe/London'});}
function scheduleSave(){if(!ready)return;revision++;submitted=false;receipt=null;clearTimeout(saveTimer);setStatus('Changes waiting to save…');saveTimer=setTimeout(()=>queueSave(),450);}
function payload(){return {schema_version:1,answers:JSON.parse(JSON.stringify(answers)),notes,submitted};}
function queueSave(force=false){
  if(!ready)return Promise.reject(new Error('Saved answers must load before writing'));
  clearTimeout(saveTimer);saveTimer=null;
  const capturedRevision=revision;
  const data=payload();
  saveChain=saveChain.catch(()=>{}).then(async()=>{
    if(!force&&capturedRevision<revision)return;
    if(!force&&capturedRevision===savedRevision&&!saveError)return;
    setStatus(data.submitted?'Saving your completed questionnaire…':'Saving on this computer…');
    try{
      const response=await fetch('/api/answers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
      if(!response.ok)throw new Error(`Save returned ${response.status}`);
      const result=await response.json();
      if(!result.ok)throw new Error('Save was not confirmed');
      savedAt=result.saved_at; savedRevision=capturedRevision; saveError=false;
      if(capturedRevision===revision){receipt=result.receipt??null;setStatus(`${data.submitted?'Questionnaire saved':'Draft saved'}${timeLabel(savedAt)?` · ${timeLabel(savedAt)}`:''}`);}
      return result;
    }catch(error){saveError=true;setStatus('Could not save locally. ',true);throw error;}
  });
  saveChain.catch(()=>{});
  return saveChain;
}
function toast(message){clearTimeout(toastTimer);$('#toast').textContent=message;$('#toast').classList.add('visible');toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),3000);}
function selectAnswer(id,value){const q=questionLookup[id];if(!q||submitting)return;if(q.multiple){const selected=new Set(answers[id]||[]);if(selected.has(value))selected.delete(value);else selected.add(value);if(selected.size)answers[id]=[...selected];else delete answers[id];}else answers[id]=value;
  if(id==='ab'){abPreview=value!=='numeric';if(abPreview)enableAB();else exportMode='numeric';}
  scheduleSave();render();
}
function renderNav(){
  $('#step-nav').innerHTML=steps.map((step,index)=>{const complete=step.questions.length>0&&step.questions.every(filled);return `<button type="button" class="step-link ${index===currentStep?'active':''} ${complete?'complete':''}" data-step="${index}" ${index===currentStep?'aria-current="step"':''}><span class="step-number">${complete?'✓':index+1}</span>${step.name}</button>`;}).join('');
}
function focusDescriptor(){const element=document.activeElement;if(!element)return null;if(element.id)return {id:element.id};for(const key of ['answer','clear','block','lora','variant','token','exportMode','step']){if(element.dataset?.[key]!==undefined)return {key,value:element.dataset[key],answerValue:element.dataset.value};}return null;}
function restoreFocus(descriptor){if(!descriptor)return;let target=descriptor.id?document.getElementById(descriptor.id):null;if(descriptor.key){const candidates=document.querySelectorAll('button');target=[...candidates].find(element=>element.dataset[descriptor.key]===descriptor.value&&(descriptor.answerValue===undefined||element.dataset.value===descriptor.answerValue));}if(target&&!target.disabled)target.focus({preventScroll:true});}
function renderQuestion(q){
  const selected=answers[q.id];
  const selectedOption=(value)=>q.multiple?Array.isArray(selected)&&selected.includes(value):selected===value;
  let options;
  if(q.reference){options=`<div class="reference-grid">${q.options.map(o=>`<div class="reference-card ${selectedOption(o.id)?'selected':''}"><button type="button" class="reference-inspect" data-reference="${o.id}" aria-label="Open reference ${o.id}"><img src="/references/${o.id}.png" alt="Your design reference ${o.id}" loading="lazy"></button><button type="button" class="reference-pick" data-answer="${q.id}" data-value="${o.id}" aria-pressed="${selectedOption(o.id)}"><input type="checkbox" tabindex="-1" aria-hidden="true" ${selectedOption(o.id)?'checked':''}>${o.title}</button></div>`).join('')}</div>`;}
  else options=`<div class="answer-options" role="group" aria-label="${escapeHtml(q.title)}">${q.options.map(o=>`<button type="button" class="answer-option ${selectedOption(o.id)?'selected':''}" data-answer="${q.id}" data-value="${o.id}" aria-pressed="${selectedOption(o.id)}"><span class="choice-circle" aria-hidden="true"></span><span class="option-copy"><span class="option-title">${escapeHtml(o.title)}</span><span class="option-description">${escapeHtml(o.description)}</span>${o.visual}</span></button>`).join('')}</div>`;
  return `<fieldset class="question"><legend>${escapeHtml(q.title)}</legend>${q.help?`<p class="question-help">${escapeHtml(q.help)}</p>`:''}${options}${filled(q)?`<button type="button" class="clear-answer" data-clear="${q.id}">Leave this unanswered</button>`:''}</fieldset>`;
}
function renderQuestions(){
  const step=steps[currentStep];
  let content=`<div class="step-overline">${currentStep===6?'Ready when you are':`Step ${currentStep+1} of 6`}</div><h1>${step.title}</h1><p class="step-intro">${step.intro}</p>`;
  if(currentStep===6){content+=`<div class="review-progress">${completedCount()} of ${allQuestions.length} choices answered · ${allQuestions.length-completedCount()} left open</div>`;
    for(let index=0;index<6;index++){content+=`<section class="review-section"><h3>${steps[index].name}</h3>${steps[index].questions.map(q=>{let label='Open — no preference yet';if(filled(q))label=q.multiple?answers[q.id].map(v=>getOptionLabel(q.id,v)).join(', '):getOptionLabel(q.id,answers[q.id]);return `<div class="review-row ${filled(q)?'':'unanswered'}"><span>${escapeHtml(q.title)}</span><b><button type="button" data-step="${index}" aria-label="Change ${escapeHtml(q.title)}">${escapeHtml(label)}</button></b></div>`;}).join('')}</section>`;}
    content+='<fieldset class="question"><legend>Anything else? <span class="question-help">Optional</span></legend><label class="question-help" for="notes">A detail you like, dislike or want us to combine.</label><textarea class="optional-note" id="notes" maxlength="10000" placeholder="For example: blue layout, violet accents, fewer rounded panels…">'+escapeHtml(notes)+'</textarea></fieldset><div class="review-tools"><button type="button" class="button" id="download-answers">Download answers as JSON</button></div><p class="finish-note">Finish saves a local design brief for review. It is a preference record; implementation details and untested block recommendations still need review.</p>';
    if(submitted&&!saveError&&!submitting&&savedRevision===revision){content+=`<div class="submit-success"><strong>Your questionnaire is saved.</strong><br>We can use this design brief in the management chat. You can still return and revise a choice.${receipt?`<br><span>Saved receipt: ${escapeHtml(receipt)}</span>`:''}</div>`;}
  }else content+=step.questions.map(renderQuestion).join('');
  $('#question-content').innerHTML=content;
  $('#question-actions').innerHTML=`<button type="button" class="button" data-step="${Math.max(0,currentStep-1)}" ${currentStep===0||submitting?'disabled':''}>← Back</button><span class="step-counter">${currentStep===6?'Review':`${currentStep+1} / 6`}</span>${currentStep===6?`<button type="button" class="button primary" id="submit-answers" ${submitting?'disabled':''}>${submitting?'Saving…':submitted?'Save updated brief':'Finish & save'} →</button>`:`<button type="button" class="button primary" data-step="${currentStep+1}">Next →</button>`}`;
  if(submitting)$('#question-content').querySelectorAll('button,input,textarea').forEach(element=>element.disabled=true);
  if(!ready){$('#question-content').querySelectorAll('[data-answer], [data-clear], #notes, #download-answers').forEach(element=>element.disabled=true);const submit=$('#submit-answers');if(submit)submit.disabled=true;if(loadError)$('#question-content').insertAdjacentHTML('afterbegin','<p class="submit-success" style="margin-bottom:18px">Your saved preferences could not be loaded. Use “Retry load” above before answering. Your existing file will not be overwritten.</p>');}
}
function icon(type){const common='viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';const paths={grid:'<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',layers:'<path d="m12 3 10 5-10 5L2 8zM2 12l10 5 10-5M2 16l10 5 10-5"/>',chart:'<path d="M4 4v16h16M8 15l4-6 4 3 4-7"/>',folder:'<path d="M3 7h7l2 3h9v10H3zM3 7V4h7l2 3h7v3"/>',settings:'<circle cx="12" cy="12" r="3"/><path d="m9 3-1 3-3 1-2 5 2 5 3 1 1 3h6l1-3 3-1 2-5-2-5-3-1-1-3z"/>'};return `<svg ${common}>${paths[type]??paths.grid}</svg>`;}
function activeData(){return currentVariant==='default'?{values:sampleDefaults[currentLora],tokens:{}}:variants[currentLora];}
function valueAt(index,data=activeData()){const token=data.tokens[index];return token&&abPreview?abValues[token]:data.values[index];}
function valueForLora(lora,index){return valueAt(index,currentVariant==='default'?{values:sampleDefaults[lora],tokens:{}}:variants[lora]);}
function formatNumber(value){return Number(value.toFixed(3)).toString();}
function fullCSV(template=false){const data=activeData();return blocks.map((_,i)=>template&&abPreview&&data.tokens[i]?data.tokens[i]:formatNumber(valueAt(i,data))).join(',');}
function blockLabel(index){const b=blocks[index];return b.group==='base'?'BASE':`${b.group==='double'?'Double':'Single'} ${String(b.number).padStart(2,'0')}`;}
function editCopy(){currentVariant='mine';variants[currentLora].edited=true;}
function enableAB(){editCopy();const variant=variants[currentLora];if(!Object.values(variant.tokens).includes('A'))variant.tokens[14]='A';if(!Object.values(variant.tokens).includes('B'))variant.tokens[40]='B';}
function libraryHTML(){return `<aside class="library-card"><div class="library-title"><b>Selected LoRAs</b><span>03</span></div>${sampleNames.map((lora,index)=>`<button type="button" class="sample-lora ${index===currentLora?'active':''}" data-lora="${index}" style="--lora-color:${lora.color}"><span class="lora-sample-row"><span class="lora-art" aria-hidden="true">${lora.symbol}</span><span><span class="lora-name">${lora.name}</span><span class="lora-role">${lora.role}</span></span></span><span class="lora-footer"><span>Loader ${String(index+1).padStart(2,'0')}</span><span>58 slots</span></span></button>`).join('')}<button type="button" class="library-button" data-demo="Library search and role selection will use your local LoRA library in the finished app.">+ Browse library</button><p class="library-note">Sample LoRAs and entered patterns. No files are loaded or analysed here.</p></aside>`;}
function chartHTML(){
  const view=answers.block_view||'bars',data=activeData();
  const groupColor=(group)=>group==='double'?'var(--accent)':group==='single'?'var(--accent2)':'var(--accent3)';
  const blockButton=(index,cssClass,extra='')=>{const value=valueAt(index),token=abPreview?data.tokens[index]:null;return `<button type="button" class="${cssClass} ${index===selectedBlock?'selected':''}" data-block="${index}" aria-label="${blockLabel(index)}, ${formatNumber(value)}${token?`, uses ${token}`:''}" aria-pressed="${index===selectedBlock}" title="${blockLabel(index)} · ${formatNumber(value)}${token?` · ${token}`:''}" ${token?`data-token="${token}"`:''} style="--block-color:${groupColor(blocks[index].group)};${extra}">${cssClass==='block-bar'?'':blocks[index].label}</button>`;};
  if(view==='heat')return `<div class="heat-groups">${['base','double','single'].map(group=>{const indexes=blocks.map((b,i)=>b.group===group?i:-1).filter(i=>i>=0);return `<div><div class="heat-group-label">${group==='base'?'BASE · 1 slot':group==='double'?'Double blocks · D00–D18':'Single blocks · S00–S37'}</div><div class="heat-cells">${indexes.map(i=>blockButton(i,'heat-cell',`--intensity:${Math.min(85,Math.max(10,Math.abs(valueAt(i))*85))}%;`)).join('')}</div></div>`;}).join('')}</div>`;
  if(view==='curve')return `<div class="curve-wrap">${curveSVG([currentLora])}</div><div class="block-mini-grid">${blocks.map((_,i)=>blockButton(i,'number-block')).join('')}</div>`;
  return `<div class="block-chart">${[['base',0,1],['double',1,20],['single',20,58]].map(([group,start,end])=>`<div class="chart-group ${group}" style="--group-size:${end-start}"><div class="group-bars">${blocks.slice(start,end).map((b,j)=>blockButton(start+j,'block-bar',`height:${Math.min(100,Math.max(2,Math.abs(valueAt(start+j))/1.2*100))}%;`)).join('')}</div><div class="group-caption"><strong>${group==='base'?'BASE':group==='double'?'Double · 19':'Single · 38'}</strong>${group==='base'?'':`<span>${group==='double'?'00–18':'00–37'}</span>`}</div></div>`).join('')}</div>`;
}
function curveSVG(loras){const points=(lora)=>blocks.map((_,i)=>`${10+i*(480/57)},${107-Math.abs(valueForLora(lora,i))*70}`).join(' ');return `<svg viewBox="0 0 500 128" role="img" aria-label="Sample entered block magnitudes; exact fields preserve signs, not measured image influence"><path d="M10 20H490M10 50H490M10 80H490M10 110H490" stroke="var(--line)" fill="none"/><path d="M173 12V112" stroke="var(--line)" stroke-dasharray="3 4"/>${loras.map(lora=>`<polyline points="${points(lora)}" fill="none" stroke="${sampleNames[lora].color}" stroke-width="2"/>`).join('')}<text x="12" y="126" fill="var(--muted)" font-size="8">BASE + double</text><text x="181" y="126" fill="var(--muted)" font-size="8">Single blocks</text></svg>`;}
function lanesHTML(){return `<div class="lanes">${sampleNames.map((lora,index)=>`<button type="button" class="lane-card ${index===currentLora?'active':''}" data-lora="${index}" style="--lora-color:${lora.color}"><span><strong>${lora.name}</strong><small>Loader ${index+1}</small></span><span class="lane-spark" aria-hidden="true">${blocks.map((_,i)=>`<i style="height:${Math.min(100,Math.abs(valueForLora(index,i))*95)}%"></i>`).join('')}</span><span class="lane-end">58 values<br>Open →</span></button>`).join('')}</div>`;}
function comparisonHTML(){const mode=answers.comparison||'notes';if(mode==='overlay')return `<div class="insight-card"><div style="width:100%"><h4>Compare entered patterns</h4>${curveSVG([0,1,2])}<p>Person / clothing / style colours correspond to the sample rows. These curves show entered magnitudes; exact fields preserve signs. They are not measured semantic effects.</p></div></div>`;if(mode==='patterns'&&answers.layout!=='lanes')return lanesHTML();return `<div class="insight-card"><span class="insight-icon" aria-hidden="true">✦</span><div><h4>${answers.explanations==='technical'?'Structural evidence, then render evidence':'Room to experiment'}</h4><p>Shared block positions are a place to investigate. These sample values do not establish conflict, compatibility or image quality.${answers.explanations==='layered'?' An evidence panel in the finished app can explain the basis of a recommendation.':''}</p></div></div>`;}
function editorHTML(){const editable=currentVariant==='mine',data=activeData(),token=abPreview?data.tokens[selectedBlock]:null;return `<div class="editor-card"><div class="editor-label">Selected block<strong>${blockLabel(selectedBlock)}</strong></div><input class="weight-range" id="block-range" type="range" min="-1" max="2" step="0.01" value="${valueAt(selectedBlock)}" ${editable?'':'disabled'} aria-label="${blockLabel(selectedBlock)} sample weight"><input id="block-number" class="weight-number" type="number" min="-1" max="2" step="0.01" value="${formatNumber(valueAt(selectedBlock))}" ${editable?'':'disabled'} aria-label="Exact sample weight for ${blockLabel(selectedBlock)}"><div class="editor-tokens">${editable?`${abPreview?`<span style="margin-left:0">Use</span>${['number','A','B'].map(t=>`<button type="button" class="small-button ${(token||'number')===t?'selected':''}" data-token="${t}">${t==='number'?'Number':t}</button>`).join('')}`:'<span style="margin-left:0">Exact value · preview edit range −1 to 2</span>'}<button type="button" class="small-button" data-reset-block="true">Reset block</button><span>Edits stay in My variant</span>`:'<span style="margin-left:0">Default is preserved.</span><button type="button" class="small-button" id="edit-copy">Edit a copy →</button>'}</div></div>`;}
function valueGridHTML(){return `<div class="ab-demo"><h4>Every slot, available together</h4><p>Inspect exact values across all groups. Editing stays with the selected block above.</p><div class="block-mini-grid">${blocks.map((b,i)=>`<button type="button" class="${i===selectedBlock?'selected':''}" data-block="${i}" title="${blockLabel(i)}">${b.label}<br>${formatNumber(valueAt(i))}</button>`).join('')}</div></div>`;}
function abHTML(){return `<div class="ab-demo"><h4>A/B · a bounded experiment</h4><p>A and B substitute into marked blocks. The min/max below are example experiment bounds, not validated recommendations. Numeric export resolves every token.</p><div class="ab-controls">${['A','B'].map(token=>`<div class="ab-control"><label for="ab-${token}">${token} · ${Object.values(activeData().tokens).filter(t=>t===token).length} slot(s)</label><input id="ab-${token}" type="number" min="${abBounds[token][0]}" max="${abBounds[token][1]}" step="0.01" value="${abValues[token]}"><small>Illustrative min ${abBounds[token][0]} / max ${abBounds[token][1]}</small></div>`).join('')}</div></div>`;}
function exportHTML(){const mode=answers.export||'vector',data=activeData(),hasTokens=abPreview&&Object.keys(data.tokens).length>0;
  const title=mode==='vector'?'Full block vector':`Loader ${currentLora+1} · ${sampleNames[currentLora].name}`;
  return `<section class="export-card"><div class="export-header"><h4>${title}</h4><span>58 / 58 slots · BASE + 19 D + 38 S</span></div>${mode==='loader'?'<p class="block-help" style="margin-bottom:9px">Illustrative supporting settings: model strength 1 · CLIP strength 1. The vector is the primary output.</p>':''}${hasTokens?`<div class="export-tabs"><button type="button" class="small-button ${exportMode==='numeric'?'selected':''}" data-export-mode="numeric">Resolved numeric values</button><button type="button" class="small-button ${exportMode==='template'?'selected':''}" data-export-mode="template">A/B template + values</button></div>`:''}<label class="panel-label" for="full-vector">${exportMode==='template'&&hasTokens?`Full A/B template · A = ${abValues.A}, B = ${abValues.B}`:'Complete comma-separated values'}</label><textarea id="full-vector" class="csv-output" readonly spellcheck="false" aria-label="Complete 58-slot block vector">${fullCSV(exportMode==='template'&&hasTokens)}</textarea><div class="export-footer"><p>${currentVariant==='default'?'Default · unchanged':'My variant · sample edits'}${exportMode==='template'&&hasTokens?` · Set A = ${abValues.A} and B = ${abValues.B} in the loader.`:' · All slots are explicit; no abbreviated ranges.'}</p><button type="button" class="save-variant" data-demo="In the finished app, this will save your exact block values as a personal entry. This questionnaire only saves your design answers.">Save variant</button><button type="button" class="copy-button" id="copy-vector">${mode==='checklist'?'Copy for this loader ✓':'Copy full vector'}</button></div>${hasTokens?`<div class="preview-switcher"><button type="button" id="copy-numeric">Copy numeric CSV</button><button type="button" id="copy-template">Copy A/B template</button><button type="button" id="copy-ab-settings">Copy A/B settings</button></div>`:''}</section>`;
}
function renderPreview(){
  const previousFocus=focusDescriptor();
  const direction=answers.direction||'prism',layout=answers.layout||'studio',density=answers.density||'balanced',thumbnails=answers.thumbnails||'small',view=answers.block_view||'bars';
  const preview=$('#app-preview');
  preview.className=`app-preview direction-${direction} ${answers.palette?`palette-${answers.palette}`:''} layout-${layout} density-${density} thumbnails-${thumbnails} ${answers.text_size==='large'?'text-large':''} ${answers.motion==='off'?'motion-off':''}`;
  $('#preview-title').textContent=names[['prism','orbit','atelier','carbon'].indexOf(direction)];
  preview.innerHTML=`<div class="app-topbar"><div class="app-wordmark"><span class="app-symbol" aria-hidden="true">◈</span>LoRA / COMBINER</div><span class="app-model">FLUX.1 · sample</span><span class="local-badge">Bender only</span></div><div class="app-shell"><aside class="app-rail" aria-hidden="true"><span class="rail-item active">${icon('grid')}</span><span class="rail-item">${icon('layers')}</span><span class="rail-item">${icon('chart')}</span><span class="rail-item">${icon('folder')}</span><span class="rail-item rail-bottom">${icon('settings')}</span></aside><div class="app-main"><div class="app-hero"><div><span class="app-eyebrow">YOUR CREATIVE WORKBENCH</span><h3>Give every LoRA room.</h3><p>Inspect individual blocks. Keep your defaults. Take a complete vector to each loader.</p></div><div class="recipe-label">Three-LoRA sample stack</div></div>${layout==='guided'?'<div class="guided-stage"><span>01 · Choose LoRAs</span><span class="active">02 · Inspect blocks</span><span>03 · Copy results</span></div>':''}<div class="app-toolbar"><button type="button" class="app-tab ${currentVariant==='default'?'active':''}" data-variant="default">Default</button><button type="button" class="app-tab ${currentVariant==='mine'?'active':''}" data-variant="mine">My variant ${variants[currentLora].edited?'· edited':'+'}</button><span class="demo-label">Sample patterns · no analysis scores</span></div>${layout==='lanes'?lanesHTML():''}<div class="workbench">${libraryHTML()}<div class="main-controls"><section class="block-card"><div class="block-heading"><div><span class="panel-label">LOADER ${String(currentLora+1).padStart(2,'0')} · ${sampleNames[currentLora].role}</span><h4>${sampleNames[currentLora].name} / block weights</h4></div><span class="variant-pill">${currentVariant==='default'?'Default · preserved':'My variant'}</span></div><div class="block-legend"><span><i style="--legend-color:var(--accent3)"></i>BASE</span><span><i style="--legend-color:var(--accent)"></i>Double blocks</span><span><i style="--legend-color:var(--accent2)"></i>Single blocks</span></div>${chartHTML()}<p class="block-help">${view==='bars'?'Select any bar to inspect its exact slot. Bar height shows magnitude; the exact field preserves the sign.':view==='heat'?'Select any tile to inspect its exact slot and value.':'Select a numbered slot below the sample curve. Curve height shows magnitude; exact fields preserve the sign.'} All 58 slots are retained in the export.</p><div class="preview-switcher"><button type="button" id="toggle-ab" class="${abPreview?'active':''}">${abPreview?'A/B demo on · turn off':'Try A/B experiment'}</button>${answers.editing==='advanced'?'<button type="button" id="expand-grid">Open exact-value grid</button>':''}</div></section>${editorHTML()}${answers.editing==='table'?valueGridHTML():''}${abPreview?abHTML():''}${comparisonHTML()}</div></div>${exportHTML()}</div></div>`;
  restoreFocus(previousFocus);
}
function render(){const previousFocus=focusDescriptor();renderNav();renderQuestions();renderPreview();restoreFocus(previousFocus);}
function goToStep(index){if(submitting)return;currentStep=Math.min(6,Math.max(0,Number(index)));renderNav();renderQuestions();if(window.innerWidth<850)$('#step-nav').scrollIntoView({behavior:answers.motion==='off'?'instant':'smooth',block:'start'});}
async function copy(text){try{await navigator.clipboard.writeText(text);toast(`Copied ${text.includes(',')?'all 58 slots':'A/B settings'}.`);}catch{let output=$('#clipboard-fallback');if(!output){output=document.createElement('textarea');output.id='clipboard-fallback';output.className='csv-output';output.setAttribute('aria-label','Text to copy manually');output.readOnly=true;$('.export-card').append(output);}output.value=text;output.focus();output.select();toast('Clipboard access is unavailable. The text is selected below the vector: press Ctrl+C.');}}
function changeWeight(value){if(currentVariant==='default')return;const number=Number(value);if(!Number.isFinite(number)||number< -1||number>2){toast('The sample editor accepts values from −1 to 2.');renderPreview();return;}variants[currentLora].values[selectedBlock]=number;delete variants[currentLora].tokens[selectedBlock];variants[currentLora].edited=true;renderPreview();}
function downloadAnswers(){const blob=new Blob([JSON.stringify({...payload(),saved_at:savedAt,receipt},null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const anchor=document.createElement('a');anchor.href=url;anchor.download='lora-gui-preferences.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function submitAnswers(){if(submitting||!ready)return;clearTimeout(saveTimer);saveTimer=null;submitting=true;submitted=true;revision++;renderQuestions();try{await queueSave(true);toast('Your design brief is saved on this computer.');}catch{toast('The brief has not been confirmed saved. Retry or download your answers.');}finally{submitting=false;renderQuestions();}}

document.addEventListener('click',event=>{
  const element=event.target.closest('button');if(!element)return;
  if(element.dataset.step!==undefined){goToStep(element.dataset.step);return;}
  if(element.dataset.answer){selectAnswer(element.dataset.answer,element.dataset.value);return;}
  if(element.dataset.clear){delete answers[element.dataset.clear];scheduleSave();render();return;}
  if(element.dataset.reference){const number=element.dataset.reference;$('#reference-title').textContent=`Reference ${number}`;$('#reference-image').src=`/references/${number}.png`;$('#reference-image').alt=`Your design reference ${number}`;$('#reference-dialog').showModal();return;}
  if(element.dataset.lora!==undefined){currentLora=Number(element.dataset.lora);if(currentVariant==='mine'&&abPreview)enableAB();renderPreview();return;}
  if(element.dataset.block!==undefined){selectedBlock=Number(element.dataset.block);renderPreview();return;}
  if(element.dataset.variant){currentVariant=element.dataset.variant;if(currentVariant==='mine')editCopy();renderPreview();return;}
  if(element.dataset.token){editCopy();if(element.dataset.token==='number'){variants[currentLora].values[selectedBlock]=valueAt(selectedBlock);delete variants[currentLora].tokens[selectedBlock];}else variants[currentLora].tokens[selectedBlock]=element.dataset.token;renderPreview();return;}
  if(element.dataset.resetBlock){variants[currentLora].values[selectedBlock]=sampleDefaults[currentLora][selectedBlock];delete variants[currentLora].tokens[selectedBlock];renderPreview();return;}
  if(element.dataset.exportMode){exportMode=element.dataset.exportMode;renderPreview();return;}
  if(element.dataset.demo){toast(element.dataset.demo);return;}
  const actions={'open-review':()=>goToStep(6),'edit-copy':()=>{editCopy();renderPreview();},'toggle-ab':()=>{abPreview=!abPreview;if(abPreview)enableAB();else exportMode='numeric';renderPreview();},'copy-vector':()=>copy(fullCSV(exportMode==='template'&&abPreview)),'copy-numeric':()=>copy(fullCSV()),'copy-template':()=>copy(fullCSV(true)),'copy-ab-settings':()=>copy(`A=${abValues.A}\nB=${abValues.B}`),'close-reference':()=>$('#reference-dialog').close(),'download-answers':downloadAnswers,'submit-answers':submitAnswers,'expand-grid':()=>{const target=$('.main-controls');const existing=$('#expanded-grid');if(existing){existing.remove();element.textContent='Open exact-value grid';}else{const container=document.createElement('div');container.id='expanded-grid';container.innerHTML=valueGridHTML();target.append(container);element.textContent='Close exact-value grid';}}};
  actions[element.id]?.();
});
document.addEventListener('input',event=>{if(event.target.id==='notes'){notes=event.target.value;scheduleSave();}if(event.target.id==='block-range'){$('#block-number').value=event.target.value;}});
document.addEventListener('change',event=>{
  if(['block-range','block-number'].includes(event.target.id)){changeWeight(event.target.value);return;}
  if(event.target.id.startsWith('ab-')){const token=event.target.id.slice(3),value=Number(event.target.value),bounds=abBounds[token];if(!bounds)return;if(!Number.isFinite(value)||value<bounds[0]||value>bounds[1]){toast(`Use an illustrative ${token} value between ${bounds[0]} and ${bounds[1]}.`);}else abValues[token]=value;renderPreview();}
});
$('#reference-dialog').addEventListener('click',event=>{if(event.target===$('#reference-dialog'))$('#reference-dialog').close();});
window.addEventListener('beforeunload',event=>{if(ready&&savedRevision!==revision){event.preventDefault();event.returnValue='';}});

async function initialize(){
  loadError=false;setStatus('Loading your answers…');
  try{const response=await fetch('/api/answers');if(!response.ok)throw new Error('Cannot load preferences');const data=await response.json();answers=validAnswers(data.answers);notes=typeof data.notes==='string'?data.notes:'';submitted=data.submitted===true;savedAt=data.saved_at;receipt=data.receipt??null;if(answers.ab&&answers.ab!=='numeric'){abPreview=true;enableAB();}ready=true;saveError=false;savedRevision=0;setStatus(savedAt?`${submitted?'Questionnaire saved':'Draft restored'}${timeLabel(savedAt)?` · ${timeLabel(savedAt)}`:''}`:'No answers chosen yet · saves locally');render();}
  catch{ready=false;loadError=true;setStatus('Could not load saved answers. ',true);render();toast('Saved answers could not be loaded. Retry load before choosing preferences.');}
}
render();initialize();
