const assert = require('node:assert/strict');
const t = require('../src/launcher_static/telemetry.js');
let count = 0;
const equal = (a,b) => { assert.deepEqual(a,b); count++; };
const sample = (overrides={}) => ({session:'a',sample_ms:1000,active:[],prefill_estimated_tokens_total:100,output_tokens_total:10,...overrides});
equal(t.observe(null,sample()).decode,null);
let next=sample({sample_ms:2000,prefill_estimated_tokens_total:600,output_tokens_total:50});
equal(t.observe(sample(),next).decode,40);
equal(t.observe(sample(),next).prefill,500);
equal(t.observe(next,sample({sample_ms:3000,prefill_estimated_tokens_total:600,output_tokens_total:50})).decode,0);
equal(t.observe(sample(),{...next,session:'restarted'}).decode,null);
equal(t.observe(sample(),{...next,sample_ms:6000}).decode,null);
equal(t.observe(sample(),{...next,sample_ms:500}).decode,null);
equal(t.observe(sample(),{...next,output_tokens_total:2}).decode,null);
equal(t.observe(sample(),{...next,active:[{prefill_reported:false}]}).prefill,null);
equal(t.observe(sample(),{...next,active:[{prefill_reported:true},{prefill_reported:true}]}).requests,2);
equal(t.acceptance({drafters:{mtp:{accepted:90,proposed:100},chain:{accepted:1,proposed:10}}}),91/110);
equal(t.acceptance({drafters:{serial:{accepted:0,proposed:0}}}),null);
equal(t.acceptance(null),null);
equal(t.acceptance({drafters:{bad:{accepted:101,proposed:100}}}),null);
for(const invalid of [null,{},sample({sample_ms:NaN}),sample({output_tokens_total:-1}),sample({active:{}}),sample({session:1})]) {
  assert.throws(()=>t.observe(null,invalid)); count++;
}
console.log(`PASS ${count} telemetry semantics assertions`);
