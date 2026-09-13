"use strict";
const $ = (id) => document.getElementById(id);
let actions = [], current, busy = false;
async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
  return result;
}
function lock(value) {
  busy = value;
  document.querySelectorAll("button,select").forEach((node) => { node.disabled = value; });
  if (!value && current?.outcome.done) {
    $("model-step").disabled = true;
    $("actions").querySelectorAll("button").forEach((node) => { node.disabled = true; });
  }
}
function show() {
  $("observation").textContent = current.observation;
  const result = current.outcome;
  $("phase").textContent = result.done ? "Episode finished" : `Step ${actions.length + 1}`;
  $("verdict").className = result.done && !result.success ? "failed" : "";
  $("verdict").textContent = result.done ? `${result.success ? "Outcome verified" : "Episode needs correction"}. Wrote: ${result.disposition || "no advisory"}. Expected: ${result.expected}. Protected repository: ${result.protected_intact ? "unchanged" : "modified"}.${result.violations.length ? " " + result.violations.join("; ") : ""}` : "No advisory written yet. The final outcome is ungraded.";
  $("trail").replaceChildren(...current.trace.map((row) => {const li=document.createElement("li");li.textContent=row.action;return li;}));
}
async function replay() {
  current = await api("/api/practice/step", {scenario_id:$("scenario").value, actions});
  show();
}
async function perform(operation) {
  if (busy) return;
  lock(true);
  try { await operation(); }
  catch (error) { $("verdict").textContent = error.message; $("verdict").className="failed"; }
  finally { lock(false); }
}
async function reset() { actions=[];$("raw").textContent="No model call yet.";await replay(); }
$("reset").onclick=()=>perform(reset);
$("scenario").onchange=()=>perform(reset);
$("model-step").onclick=()=>perform(async()=>{
  const selected=$("policy").value;
  const result=await api("/api/infer", {skill_id:"ci-diagnostic",input:current.observation,model:selected==="base"?"base":"specialist",checkpoint:selected==="base"?null:selected});
  $("raw").textContent=result.raw;
  if (!result.valid || !result.evidence_valid) throw new Error("Invalid model output; no action executed. Inspect the raw response or reset.");
  actions.push(result.output.bucket);await replay();
});
perform(async()=>{
  const [catalog, comparisons, runs]=await Promise.all([api("/api/practice"),api("/api/workflow-experiments"),api("/api/runs")]);
  catalog.scenarios.forEach(row=>{const option=document.createElement("option");option.value=row.id;option.textContent=row.family;$("scenario").append(option);});
  Object.keys(catalog.actions).forEach(action=>{const button=document.createElement("button");button.textContent=action;button.title=catalog.actions[action];button.onclick=()=>perform(async()=>{actions.push(action);await replay();});$("actions").append(button);});
  runs.filter(row=>row.skill_id==="ci-diagnostic"&&row.kind==="train"&&row.status==="completed").forEach(row=>{const option=document.createElement("option");option.value=row.id;option.textContent=`Trained adapter · ${row.id.slice(-6)}`;$("policy").append(option);});
  comparisons.forEach(row=>{
    const m=row.metrics, tr=document.createElement("tr"), count=m.count;
    const correct=row.skill_id==="ci-diagnostic"?m.success:Math.round(m.accuracy*count);
    [row.skill_id==="ci-diagnostic"?"CI evidence":"Blox arithmetic",row.policy,`${correct} / ${count}`,m.harmful===undefined?"—":m.harmful,`${m.median_ms.toFixed(1)} ms`].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});
    tr.tabIndex=0;tr.onclick=()=>{const failures=row.rows.filter(item=>item.success===false||item.correct===false);$("case-data").textContent=JSON.stringify({policy:row.policy,checkpoint:row.checkpoint?.id,failures:failures.length?failures:row.rows.slice(0,1)},null,2);$("case-detail").open=true;};tr.onkeydown=event=>{if(event.key==="Enter")tr.click();};$("results").append(tr);
  });
  await reset();
});
