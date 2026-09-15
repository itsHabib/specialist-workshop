const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const pct = (n) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);
const state = {
  skills: [],
  skill: null,
  runs: [],
  status: {},
  checkpoint: "",
  page: "workshop",
  detail: null,
  result: null,
  input: "",
  polling: false,
};
let toastTimer;

async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) {
    const message = Array.isArray(data.detail)
      ? data.detail.map((x) => x.msg).join("; ")
      : data.detail;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return data;
}
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").className = `toast${error ? " error" : ""}`;
  toastTimer = setTimeout(
    () => $("toast").classList.add("hidden"),
    error ? 11000 : 6500,
  );
}
function page(name) {
  state.page = name;
  document
    .querySelectorAll(".page")
    .forEach((x) => x.classList.toggle("hidden", x.id !== `page-${name}`));
  document
    .querySelectorAll("[data-page]")
    .forEach((x) => x.classList.toggle("active", x.dataset.page === name));
  if (name === "api") updateCode();
  window.scrollTo({ top: 0, behavior: "instant" });
}
function modelName(model) {
  return {
    base: "Base model",
    specialist: "Your specialist",
    reference: "Reference provider",
  }[model];
}
function setInput(text) {
  $("input").value = text;
  $("input-size").textContent = `${text.length.toLocaleString()} characters`;
  state.result = null;
}

async function selectSkill(id) {
  state.skill = await api(`/api/skills/${encodeURIComponent(id)}`);
  state.checkpoint = "";
  state.result = null;
  $("skill-name").textContent = state.skill.name;
  $("crumb").textContent = state.skill.name;
  $("skill-description").textContent = state.skill.description;
  $("skill-origin").textContent =
    state.skill.id === "ci-triage" ? "WORKBENCH" : "YOUR SKILL";
  $("label-count").textContent =
    `${Object.keys(state.skill.labels).length} labels · one contract`;
  $("example-count").textContent =
    `${state.skill.train.length} practice · ${state.skill.validation.length} validation`;
  const summary = state.skills.find((s) => s.id === state.skill.id);
  $("eval-caption").textContent =
    `${state.skill.test.length} reserved test cases · Majority-label baseline ${pct(summary?.majority_baseline)}`;
  $("correct-label").innerHTML = Object.keys(state.skill.labels)
    .map((x) => `<option value="${esc(x)}">${esc(x)}</option>`)
    .join("");
  const sample = state.skill.train[0];
  setInput(sample.input);
  $("correct-evidence").value = "";
  $("output").innerHTML =
    '<div class="empty-output"><span class="empty-glyph">⤷</span><h4>A little model. A specific answer.</h4><p>Run an input to see its classification,<br>supporting quote, and validation.</p></div>';
  renderCheckpoints();
  renderMetrics();
  renderLibrary();
  updateCode();
}

function renderCheckpoints() {
  const summary = state.skills.find((s) => s.id === state.skill?.id);
  const checkpoints = state.runs.filter(
    (r) =>
      r.skill_id === state.skill?.id &&
      r.kind === "train" &&
      r.status === "completed" &&
      r.model_id === state.status.model &&
      r.model_revision === state.status.revision &&
      r.prompt_fingerprint === summary?.prompt_fingerprint,
  );
  if (!checkpoints.find((r) => r.id === state.checkpoint))
    state.checkpoint = checkpoints[0]?.id || "";
  $("checkpoint").innerHTML = checkpoints.length
    ? checkpoints
        .map(
          (r) =>
            `<option value="${esc(r.id)}">${esc(r.id)} · ${r.iterations} steps</option>`,
        )
        .join("")
    : '<option value="">No trained checkpoint yet</option>';
  $("checkpoint").value = state.checkpoint;
  $("model-select").querySelector('[value="specialist"]').disabled =
    !state.checkpoint;
  $("model-select").querySelector('[value="reference"]').disabled =
    !state.status.reference_configured;
}

function metricFor(model) {
  const summary = state.skills.find((s) => s.id === state.skill.id);
  return state.runs.find(
    (r) =>
      r.kind === "evaluate" &&
      r.status === "completed" &&
      r.skill_id === state.skill.id &&
      r.model === model &&
      r.test_fingerprint === summary?.test_fingerprint &&
      r.prompt_fingerprint === summary?.prompt_fingerprint &&
      r.grader_version === state.status.grader_version &&
      (model === "reference" ||
        (r.model_id === state.status.model &&
          r.model_revision === state.status.revision)) &&
      (model !== "specialist" || r.checkpoint === state.checkpoint),
  );
}
function renderMetrics() {
  if (!state.skill) return;
  $("scorecards").innerHTML = ["base", "specialist", "reference"]
    .map((model) => {
      const run = metricFor(model),
        m = run?.metrics;
      const subtitle = {
        base: "Qwen 2.5 · 1.5B",
        specialist: "Same model + your adapter",
        reference: state.status.reference_model || "Connect a provider",
      }[model];
      const footer = m
        ? `${m.count} cases evaluated`
        : {
            base: "Run the baseline evaluation",
            specialist: state.checkpoint
              ? "Checkpoint ready to evaluate"
              : "Train your first checkpoint",
            reference: "No benchmark result yet",
          }[model];
      return `<div class="scorecard ${model}"><div class="card-top"><span>${modelName(model)}</span><span class="model-badge">${model === "specialist" ? "SPECIALIZED" : model === "base" ? "STARTING POINT" : "OPTIONAL"}</span></div><strong>${pct(m?.accuracy)}</strong><div class="metric-label">${m ? "label accuracy" : subtitle}</div><div class="score-foot"><span>${footer}</span><span>${m ? `${(m.median_ms / 1000).toFixed(2)}s median` : "Not measured"}</span></div><div class="spark-bar"><span data-width="${m ? m.accuracy * 100 : 0}"></span></div></div>`;
    })
    .join("");
  document
    .querySelectorAll("[data-width]")
    .forEach((el) => (el.style.width = `${Number(el.dataset.width)}%`));
}

function updateCode() {
  if (!state.skill) return;
  const request = {
    skill_id: state.skill.id,
    input: "Paste your input here",
    model: state.checkpoint ? "specialist" : "base",
  };
  if (state.checkpoint) request.checkpoint = state.checkpoint;
  $("mini-code").textContent =
    `POST /api/infer\n${JSON.stringify(request, null, 2)}`;
  $("api-code").textContent =
    `curl ${location.origin}/api/infer \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(request, null, 2)}'`;
}

async function infer() {
  const input = $("input").value;
  if (!input.trim()) return toast("Provide an input first.", true);
  const model = $("model-select").value;
  const button = $("infer");
  button.disabled = true;
  $("output").innerHTML =
    `<div class="empty-output"><span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span><h4>Running ${esc(modelName(model).toLowerCase())}…</h4><p>The first call includes model loading.<br>This is a real inference request.</p></div>`;
  const started = performance.now();
  try {
    const result = await api("/api/infer", {
      skill_id: state.skill.id,
      input,
      model,
      checkpoint: model === "specialist" ? state.checkpoint : null,
    });
    state.result = result;
    state.input = input;
    $("output-timing").textContent =
      `${(result.elapsed_ms / 1000).toFixed(2)}s generation`;
    if (!result.output) {
      $("output").innerHTML =
        `<div class="result-label">OUTPUT NEEDS ATTENTION</div><h4>Could not parse the response</h4><pre class="raw-output">${esc(result.raw)}</pre>`;
      return;
    }
    $("output").innerHTML =
      `<div class="result-label">PREDICTED LABEL</div><div class="result-bucket">${esc(result.output.bucket || "No label")}</div><p class="result-description">${esc(state.skill.labels[result.output.bucket] || "The response did not match the skill contract.")}</p><div class="evidence-quote">${esc(result.output.evidence || "No evidence returned")}</div><div class="validation"><span class="${result.valid ? "" : "bad"}">${result.valid ? "✓" : "×"} Output contract</span><span class="${result.evidence_valid ? "" : "bad"}">${result.evidence_valid ? "✓" : "×"} Quote found in input</span></div>`;
    $("output-caption").textContent =
      `${modelName(model)} · ${((performance.now() - started) / 1000).toFixed(1)}s including loading`;
  } catch (error) {
    $("output").innerHTML =
      `<div class="empty-output"><h4>Inference could not complete</h4><p>${esc(error.message)}</p></div>`;
    toast(error.message, true);
  } finally {
    button.disabled = false;
    await refresh();
  }
}

async function startRun(kind) {
  const model = kind === "train" ? "base" : $("model-select").value;
  const iterations = Number($("iterations").value);
  const request = {
    skill_id: state.skill.id,
    kind,
    model,
    iterations,
    checkpoint: model === "specialist" ? state.checkpoint : null,
  };
  try {
    const job = await api("/api/runs", request);
    state.detail = job.id;
    toast(
      kind === "train"
        ? "Training started. You can inspect the live log in Experiments."
        : "Evaluation started on the reserved test set.",
    );
    await refresh();
    page("experiments");
    await showRun(job.id);
  } catch (error) {
    toast(error.message, true);
  }
}

function renderExperiments() {
  $("run-count").textContent = state.runs.length;
  const runs = state.runs.filter((r) => r.skill_id === state.skill?.id);
  $("experiment-list").innerHTML = runs.length
    ? runs
        .map(
          (r) =>
            `<button class="experiment-row" data-run="${esc(r.id)}"><span class="experiment-icon">${r.kind === "train" ? "✳" : "◷"}</span><span class="experiment-name"><strong>${r.kind === "train" ? "Train specialist" : `Evaluate ${modelName(r.model).toLowerCase()}`}</strong><small>${esc(r.id)} · ${esc(new Date(r.created_at * 1000).toLocaleString())}</small></span><span class="status-chip ${esc(r.status)}">${esc(r.status)}${r.status === "running" && r.kind === "evaluate" ? ` · ${r.progress}/${r.total}` : ""}</span><span class="experiment-metric">${r.metrics ? pct(r.metrics.accuracy) : `${r.iterations} steps`}</span><span>↗</span></button>`,
        )
        .join("")
    : '<div class="panel import-panel"><h3>Your first experiment starts here.</h3><p>Run a baseline evaluation or train a checkpoint from the workshop.</p></div>';
  document
    .querySelectorAll("[data-run]")
    .forEach(
      (el) =>
        (el.onclick = () =>
          showRun(el.dataset.run).catch((e) => toast(e.message, true))),
    );
}

function lossChart(log) {
  const points = [...log.matchAll(/Iter (\d+): Train loss ([\d.]+)/g)].map(
    (m) => [+m[1], +m[2]],
  );
  if (points.length < 2) return "";
  const maxX = Math.max(...points.map((p) => p[0])),
    maxY = Math.max(...points.map((p) => p[1]));
  const coordinates = points
    .map(([x, y]) => `${10 + (x / maxX) * 780},${125 - (y / maxY) * 110}`)
    .join(" ");
  return `<div class="tiny">Training loss · supervised token prediction · lower is better</div><svg class="loss-chart" viewBox="0 0 800 140" role="img" aria-label="Measured training loss"><line x1="10" y1="125" x2="790" y2="125"/><line x1="10" y1="65" x2="790" y2="65"/><polyline points="${coordinates}"/></svg>`;
}
async function showRun(id) {
  state.detail = id;
  const r = await api(`/api/runs/${id}`);
  const detail = $("run-detail");
  detail.classList.remove("hidden");
  const m = r.metrics;
  detail.innerHTML = `<div class="run-detail-head"><div><div class="eyebrow">${esc(r.id)}</div><h3>${r.kind === "train" ? "Supervised adapter training" : `${modelName(r.model)} evaluation`}</h3></div><span class="status-chip ${esc(r.status)}">${esc(r.status)}</span></div><p class="run-notes">${esc(r.model_id)}<br>${m ? `${m.count} cases · Label accuracy ${pct(m.accuracy)} · Verbatim evidence ${pct(m.evidence_rate)} · Correct label + quote ${pct(m.accepted_rate)}` : `${r.training_count} practice examples · ${r.validation_count} validation examples · ${r.iterations} steps`}<br>Skill snapshot: ${esc(r.skill_fingerprint?.slice(0, 16))} · Test set: ${esc(r.test_fingerprint?.slice(0, 16))}</p>${r.error ? `<p class="run-notes">${esc(r.error)}</p>` : ""}${r.kind === "train" ? lossChart(r.log) : ""}${r.log ? `<details><summary class="tiny">Inspect raw run log</summary><pre>${esc(r.log)}</pre></details>` : ""}${r.adapter_sha256 ? `<p class="tiny">Adapter SHA-256: ${esc(r.adapter_sha256)}</p>` : ""}${r.rows?.length ? `<table class="results-table"><thead><tr><th>CASE</th><th>EXPECTED</th><th>MODEL</th><th>LABEL</th><th>QUOTE</th></tr></thead><tbody>${r.rows.map((row, i) => `<tr data-case="${i}" tabindex="0" role="button"><td class="case-source">${esc(row.source)}</td><td>${esc(row.expected)}</td><td>${esc(row.output?.bucket || "invalid")}</td><td>${row.correct ? "✓" : "×"}</td><td>${row.evidence_valid ? "✓" : "×"}</td></tr>`).join("")}</tbody></table><div id="case-inspector"></div>` : ""}`;
  document.querySelectorAll("[data-case]").forEach((el) => {
    const inspect = () => {
      const row = r.rows[Number(el.dataset.case)];
      $("case-inspector").innerHTML =
        `<div class="case-inspector"><h4>${esc(row.source)}</h4><p>Reserved evaluation input. Do not add this case to training.</p><pre>${esc(row.input)}</pre><h4>Raw model response</h4><pre>${esc(row.raw)}</pre></div>`;
    };
    el.onclick = inspect;
    el.onkeydown = (event) => {
      if (event.key === "Enter") inspect();
    };
  });
}

function renderLibrary() {
  $("skill-list").innerHTML = state.skills
    .map(
      (s) =>
        `<button class="skill-tile ${s.id === state.skill?.id ? "active" : ""}" data-skill="${esc(s.id)}"><strong>${esc(s.name)}</strong><p>${esc(s.description)}</p><small>${s.train_count} practice · ${s.test_count} test cases ${s.id === state.skill?.id ? " · ACTIVE" : " →"}</small></button>`,
    )
    .join("");
  document.querySelectorAll("[data-skill]").forEach(
    (el) =>
      (el.onclick = async () => {
        try {
          await selectSkill(el.dataset.skill);
          renderExperiments();
          toast("Active skill changed.");
        } catch (e) {
          toast(e.message, true);
        }
      }),
  );
  $("provenance").innerHTML =
    Object.entries(state.skill?.provenance || {})
      .map(
        ([key, value]) =>
          `<dt>${esc(key.replaceAll("_", " "))}</dt><dd>${esc(value)}</dd>`,
      )
      .join("") ||
    "<dd>User-provided skill. Review its examples and labels before using the results.</dd>";
}

async function refresh() {
  if (state.polling) return;
  state.polling = true;
  try {
    [state.status, state.runs, state.skills] = await Promise.all([
      api("/api/status"),
      api("/api/runs"),
      api("/api/skills"),
    ]);
    $("runtime-state").className = `runtime${state.status.busy ? " busy" : ""}`;
    $("runtime-state").innerHTML =
      `<span class="dot"></span> ${state.status.busy ? "Experiment running" : "Local runtime ready"}`;
    if (!state.skill && state.skills.length)
      await selectSkill(state.skills[0].id);
    renderCheckpoints();
    renderMetrics();
    renderExperiments();
    updateCode();
    const active = state.runs.find(
      (r) => r.status === "running" || r.status === "queued",
    );
    $("train-feedback").textContent = active
      ? `${active.kind === "train" ? "Training" : "Evaluating"} · ${active.id}`
      : state.checkpoint
        ? "Checkpoint ready. Test it before relying on it."
        : "Training runs locally; no cloud training bill.";
    ["train", "evaluate", "infer"].forEach(
      (id) => ($(id).disabled = state.status.busy),
    );
    if (state.page === "experiments" && state.detail) {
      const selected = state.runs.find((r) => r.id === state.detail);
      if (selected && ["running", "queued"].includes(selected.status))
        await showRun(state.detail);
      if (selected && $("run-detail").dataset.lastStatus !== selected.status) {
        await showRun(state.detail);
        $("run-detail").dataset.lastStatus = selected.status;
      }
    }
  } finally {
    state.polling = false;
  }
}

document
  .querySelectorAll("[data-page]")
  .forEach((el) => (el.onclick = () => page(el.dataset.page)));
$("api-shortcut").onclick = () => page("api");
$("show-experiments").onclick = () => page("experiments");
$("import-shortcut").onclick = () => page("library");
$("input").oninput = () => {
  state.result = null;
  $("input-size").textContent =
    `${$("input").value.length.toLocaleString()} characters`;
};
$("load-example").onclick = () => {
  const r =
    state.skill.train[Math.floor(Math.random() * state.skill.train.length)];
  setInput(r.input);
  toast(
    "Loaded a practice example. Use the test set for generalization evidence.",
  );
};
$("infer").onclick = infer;
$("train").onclick = () => startRun("train");
$("evaluate").onclick = () => startRun("evaluate");
$("refresh-runs").onclick = () =>
  refresh().catch((e) => toast(e.message, true));
$("checkpoint").onchange = () => {
  state.checkpoint = $("checkpoint").value;
  renderMetrics();
  updateCode();
};
$("correct-toggle").onclick = () => {
  $("correction-panel").classList.remove("hidden");
  if (state.result && state.input === $("input").value) {
    const out = state.result.output || {};
    $("correct-label").value = out.bucket || Object.keys(state.skill.labels)[0];
    $("correct-evidence").value = out.evidence || "";
  }
  $("correction-panel").scrollIntoView({
    block: "center",
    behavior: "instant",
  });
};
$("close-correction").onclick = () =>
  $("correction-panel").classList.add("hidden");
$("save-correction").onclick = async () => {
  try {
    await api(`/api/skills/${state.skill.id}/corrections`, {
      input: $("input").value,
      expected: $("correct-label").value,
      evidence: $("correct-evidence").value,
    });
    state.skill = await api(`/api/skills/${state.skill.id}`);
    $("example-count").textContent =
      `${state.skill.train.length} practice · ${state.skill.validation.length} validation`;
    $("correction-panel").classList.add("hidden");
    await refresh();
    toast("Practice example saved. Train a new checkpoint to learn from it.");
  } catch (e) {
    toast(e.message, true);
  }
};
$("download-skill").onclick = () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.skill, null, 2)], {
      type: "application/json",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.skill.id}.json`;
  a.click();
  URL.revokeObjectURL(url);
};
$("skill-file").onchange = async () => {
  const f = $("skill-file").files[0];
  if (!f) return;
  if (f.size > 4000000) return toast("File exceeds 4 MB.", true);
  $("skill-json").value = await f.text();
};
$("load-template").onclick = async () => {
  try {
    $("skill-json").value = JSON.stringify(
      await api("/static/skill-template.json"),
      null,
      2,
    );
  } catch (e) {
    toast(e.message, true);
  }
};
$("import-skill").onclick = async () => {
  try {
    const skill = JSON.parse($("skill-json").value);
    const saved = await api("/api/skills", skill);
    await refresh();
    await selectSkill(saved.id);
    $("import-feedback").textContent =
      "Skill imported. Try an input, then evaluate and train.";
    page("workshop");
    toast("Your skill is ready.");
  } catch (e) {
    toast(e.message, true);
  }
};
await refresh().catch((e) =>
  toast(`Could not connect to the workshop: ${e.message}`, true),
);
setInterval(
  () =>
    refresh().catch(() => {
      $("runtime-state").textContent = "Runtime disconnected";
    }),
  4000,
);
