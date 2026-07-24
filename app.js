const DTYPE_BYTES = { fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5, int2: 0.25 };

const HARDWARE_PRESETS = {
  "atlas-a3-64x16": { gpuMemoryGiB: 64, gpuCount: 16, gpuUtil: 0.8 },
  "atlas-a3-128x8": { gpuMemoryGiB: 128, gpuCount: 8, gpuUtil: 0.9 },
  "atlas-a2-64x8": { gpuMemoryGiB: 64, gpuCount: 8, gpuUtil: 0.8 },
  "h200-141": { gpuMemoryGiB: 141, gpuCount: 8, gpuUtil: 0.95 },
};

const MODEL_PRESETS = {
  "glm52-w4a8c8": { hardwarePreset: "atlas-a3-64x16", layers: 78, kvHeads: 64, kDim: 256, vDim: 256, maxModelLen: 1024000, fullContextSeqs: 1, schedulerMaxSeqs: 32, blockSize: 16, tp: 16, pp: 1, dp: 1, kvCp: 16, totalParamsB: 743, weightBytes: 0.5, kvBytes: 1, activationBytes: 1, runtimeReserveGiB: 8, expertWeightPercent: 85, expertParallel: true },
  "glm52-w8a8": { hardwarePreset: "atlas-a3-64x16", layers: 78, kvHeads: 64, kDim: 256, vDim: 256, maxModelLen: 20480, fullContextSeqs: 1, schedulerMaxSeqs: 48, blockSize: 16, tp: 8, pp: 1, dp: 2, kvCp: 1, totalParamsB: 743, weightBytes: 1, kvBytes: 1, activationBytes: 1, runtimeReserveGiB: 8, expertWeightPercent: 85, expertParallel: true },
  "glm52-fp8": { hardwarePreset: "h200-141", layers: 78, kvHeads: 64, kDim: 256, vDim: 256, maxModelLen: 131072, fullContextSeqs: 1, schedulerMaxSeqs: 32, blockSize: 16, tp: 8, pp: 1, dp: 1, kvCp: 1, totalParamsB: 743, weightBytes: 1, kvBytes: 1, activationBytes: 2, runtimeReserveGiB: 8, expertWeightPercent: 85, expertParallel: false },
  "glm52-bf16": { hardwarePreset: "atlas-a3-128x8", layers: 78, kvHeads: 64, kDim: 256, vDim: 256, maxModelLen: 131072, fullContextSeqs: 1, schedulerMaxSeqs: 32, blockSize: 16, tp: 8, pp: 1, dp: 1, kvCp: 1, totalParamsB: 743, weightBytes: 2, kvBytes: 2, activationBytes: 2, runtimeReserveGiB: 10, expertWeightPercent: 85, expertParallel: false },
};

const deployFields = {
  modelPreset: document.querySelector("#model-preset"), hardwarePreset: document.querySelector("#hardware-preset"), gpuMemoryGiB: document.querySelector("#gpu-memory-gib"), gpuCount: document.querySelector("#gpu-count"), gpuUtil: document.querySelector("#gpu-util"), runtimeReserveGiB: document.querySelector("#runtime-reserve-gib"), tp: document.querySelector("#tp-size"), pp: document.querySelector("#pp-size"), dp: document.querySelector("#dp-size"), kvCp: document.querySelector("#kv-cp-size"), expertParallel: document.querySelector("#expert-parallel"), layers: document.querySelector("#deploy-layers"), kvHeads: document.querySelector("#deploy-kv-heads"), kDim: document.querySelector("#k-dim"), vDim: document.querySelector("#v-dim"), maxModelLen: document.querySelector("#max-model-len"), fullContextSeqs: document.querySelector("#max-num-seqs"), blockSize: document.querySelector("#block-size"), schedulerMaxSeqs: document.querySelector("#scheduler-max-seqs"), totalParamsB: document.querySelector("#total-params-b"), weightBytes: document.querySelector("#weight-bytes"), kvBytes: document.querySelector("#kv-bytes"), activationBytes: document.querySelector("#activation-bytes"), expertWeightPercent: document.querySelector("#expert-weight-percent"), checkpointGiB: document.querySelector("#checkpoint-gib"),
};

const deployOutput = { error: document.querySelector("#deployment-error"), statusCard: document.querySelector("#fit-status-card"), fitStatus: document.querySelector("#fit-status"), fitSummary: document.querySelector("#fit-summary"), total: document.querySelector("#deploy-total-gib"), headroom: document.querySelector("#deploy-headroom-gib"), weight: document.querySelector("#deploy-weight-gib"), kv: document.querySelector("#deploy-kv-gib"), maxTokens: document.querySelector("#deploy-max-tokens"), maxConcurrency: document.querySelector("#deploy-max-concurrency"), gpuRequired: document.querySelector("#gpu-required"), effectiveTokens: document.querySelector("#effective-tokens"), availableKv: document.querySelector("#available-kv-gib"), quantSummary: document.querySelector("#quant-summary") };
const basicFields = { layers: document.querySelector("#basic-layers"), kvHeads: document.querySelector("#basic-kv-heads"), context: document.querySelector("#basic-context"), batch: document.querySelector("#basic-batch"), headDim: document.querySelector("#basic-head-dim"), dtype: document.querySelector("#basic-dtype") };
const basicOutput = { error: document.querySelector("#basic-error"), bytes: document.querySelector("#basic-bytes"), mib: document.querySelector("#basic-mib"), gib: document.querySelector("#basic-gib") };

function bytesToGiB(bytes) { return bytes / 1024 ** 3; }
function bytesToMiB(bytes) { return bytes / 1024 ** 2; }
function gibToBytes(gib) { return gib * 1024 ** 3; }
function formatNumber(value, digits = 0) { return new Intl.NumberFormat("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value); }
function formatGiB(value) { return `${formatNumber(value, 2)} GiB`; }
function getDtypeBytes(dtype) { const bytes = DTYPE_BYTES[dtype]; if (bytes === undefined) throw new Error(`不支持的 dtype: ${dtype}`); return bytes; }

function readNumber(field, label, { integer = false, min = 0, max = Infinity, optional = false } = {}) {
  const raw = field.value.trim(); field.classList.remove("is-invalid");
  if (optional && raw === "") return null;
  const value = Number(raw);
  if (raw === "" || Number.isNaN(value) || !Number.isFinite(value)) { field.classList.add("is-invalid"); throw new Error(`${label} 必须是数字`); }
  if (integer && !Number.isInteger(value)) { field.classList.add("is-invalid"); throw new Error(`${label} 必须是整数`); }
  if (value < min || value > max) { field.classList.add("is-invalid"); throw new Error(`${label} 必须在 ${min} 到 ${max} 之间`); }
  return value;
}

function calculateBasicKvCache({ batch, layers, kvHeads, context, headDim, dtype }) { return 2 * batch * layers * kvHeads * context * headDim * getDtypeBytes(dtype); }

function calculateDeploymentMemory(inputs) {
  const alignedLen = Math.ceil(inputs.maxModelLen / inputs.blockSize) * inputs.blockSize;
  const logicalTokens = inputs.fullContextSeqs * alignedLen;
  const effectiveTokensPerCard = Math.ceil(logicalTokens / inputs.kvCp);
  const layersPerCard = Math.ceil(inputs.layers / inputs.pp);
  const kvHeadsPerCard = Math.ceil(inputs.kvHeads / inputs.tp);
  const kvPerCardBytes = effectiveTokensPerCard * layersPerCard * kvHeadsPerCard * (inputs.kDim + inputs.vDim) * inputs.kvBytes;
  const totalWeightBytes = inputs.checkpointGiB !== null ? gibToBytes(inputs.checkpointGiB) : inputs.totalParamsB * 1e9 * inputs.weightBytes;
  const expertRatio = inputs.expertWeightPercent / 100;
  const expertBytes = totalWeightBytes * expertRatio;
  const nonExpertBytes = totalWeightBytes - expertBytes;
  const weightPerCardBytes = inputs.expertParallel
    ? nonExpertBytes / (inputs.tp * inputs.pp) + expertBytes / (inputs.tp * inputs.dp * inputs.pp)
    : totalWeightBytes / (inputs.tp * inputs.pp);
  const budgetGiB = inputs.gpuMemoryGiB * inputs.gpuUtil;
  const weightGiB = bytesToGiB(weightPerCardBytes);
  const kvGiB = bytesToGiB(kvPerCardBytes);
  const totalGiB = weightGiB + kvGiB + inputs.runtimeReserveGiB;
  const headroomGiB = budgetGiB - totalGiB;
  const availableKvGiB = budgetGiB - weightGiB - inputs.runtimeReserveGiB;
  const perTokenFullModelKvBytes = inputs.layers * inputs.kvHeads * (inputs.kDim + inputs.vDim) * inputs.kvBytes;
  const maxCachedTokens = Math.max(0, Math.floor(gibToBytes(Math.max(availableKvGiB, 0)) * inputs.tp * inputs.pp * inputs.kvCp / perTokenFullModelKvBytes));
  const maxFullContextSeqs = maxCachedTokens / inputs.maxModelLen;
  const cardRequired = inputs.tp * inputs.pp * inputs.dp;
  let status = "Fit";
  if (headroomGiB < 0) status = "OOM Risk";
  else if (headroomGiB / budgetGiB < 0.08) status = "Tight";
  return { alignedLen, logicalTokens, effectiveTokensPerCard, cardRequired, budgetGiB, weightGiB, kvGiB, totalGiB, headroomGiB, availableKvGiB, maxCachedTokens, maxFullContextSeqs, status, hasEnoughCardCount: inputs.gpuCount >= cardRequired };
}

function readDeploymentInputs() {
  return { gpuMemoryGiB: readNumber(deployFields.gpuMemoryGiB, "单卡显存", { min: 1 }), gpuCount: readNumber(deployFields.gpuCount, "卡数", { integer: true, min: 1 }), gpuUtil: readNumber(deployFields.gpuUtil, "memory utilization", { min: 0.01, max: 1 }), runtimeReserveGiB: readNumber(deployFields.runtimeReserveGiB, "Runtime reserve", { min: 0 }), tp: readNumber(deployFields.tp, "TP", { integer: true, min: 1 }), pp: readNumber(deployFields.pp, "PP", { integer: true, min: 1 }), dp: readNumber(deployFields.dp, "DP", { integer: true, min: 1 }), kvCp: readNumber(deployFields.kvCp, "KV CP / DCP shard", { integer: true, min: 1 }), expertParallel: deployFields.expertParallel.checked, layers: readNumber(deployFields.layers, "Layers", { integer: true, min: 1 }), kvHeads: readNumber(deployFields.kvHeads, "KV heads", { integer: true, min: 1 }), kDim: readNumber(deployFields.kDim, "K dim", { integer: true, min: 1 }), vDim: readNumber(deployFields.vDim, "V dim", { integer: true, min: 1 }), maxModelLen: readNumber(deployFields.maxModelLen, "max_model_len", { integer: true, min: 1 }), fullContextSeqs: readNumber(deployFields.fullContextSeqs, "Full-context seqs", { integer: true, min: 1 }), blockSize: readNumber(deployFields.blockSize, "block_size", { integer: true, min: 1 }), schedulerMaxSeqs: readNumber(deployFields.schedulerMaxSeqs, "vLLM max_num_seqs", { integer: true, min: 1 }), totalParamsB: readNumber(deployFields.totalParamsB, "Total params B", { min: 0.1 }), weightBytes: readNumber(deployFields.weightBytes, "Weight bytes/param", { min: 0.01 }), kvBytes: readNumber(deployFields.kvBytes, "KV bytes/element", { min: 0.01 }), activationBytes: readNumber(deployFields.activationBytes, "Activation bytes", { min: 0.01 }), expertWeightPercent: readNumber(deployFields.expertWeightPercent, "Expert weight %", { min: 0, max: 100 }), checkpointGiB: readNumber(deployFields.checkpointGiB, "Checkpoint size GiB", { min: 0, optional: true }) };
}

function renderDeploymentInvalid(message) {
  deployOutput.error.textContent = message; deployOutput.fitStatus.textContent = "--"; deployOutput.fitSummary.textContent = "请修正参数"; deployOutput.statusCard.className = "status-card";
  [deployOutput.total, deployOutput.headroom, deployOutput.weight, deployOutput.kv, deployOutput.maxTokens, deployOutput.maxConcurrency, deployOutput.gpuRequired, deployOutput.effectiveTokens, deployOutput.availableKv, deployOutput.quantSummary].forEach((el) => { el.textContent = "--"; });
}

function renderDeployment(result, inputs) {
  deployOutput.error.textContent = "";
  deployOutput.statusCard.className = `status-card ${result.status === "Fit" ? "fit" : result.status === "Tight" ? "tight" : "risk"}`;
  deployOutput.fitStatus.textContent = result.status;
  deployOutput.fitSummary.textContent = result.hasEnoughCardCount ? `需要 ${result.cardRequired} 张卡，当前卡数满足并行配置` : `需要 ${result.cardRequired} 张卡，但当前只填了 ${inputs.gpuCount} 张`;
  deployOutput.total.textContent = `${formatGiB(result.totalGiB)} / ${formatGiB(result.budgetGiB)}`;
  deployOutput.headroom.textContent = formatGiB(result.headroomGiB);
  deployOutput.weight.textContent = formatGiB(result.weightGiB);
  deployOutput.kv.textContent = formatGiB(result.kvGiB);
  deployOutput.maxTokens.textContent = formatNumber(result.maxCachedTokens, 0);
  deployOutput.maxConcurrency.textContent = `${formatNumber(result.maxFullContextSeqs, 2)}x`;
  deployOutput.gpuRequired.textContent = `${result.cardRequired} = TP ${inputs.tp} × PP ${inputs.pp} × DP ${inputs.dp}`;
  deployOutput.effectiveTokens.textContent = `${formatNumber(result.effectiveTokensPerCard, 0)} / card (${formatNumber(result.logicalTokens, 0)} ÷ CP ${inputs.kvCp})`;
  deployOutput.availableKv.textContent = formatGiB(result.availableKvGiB);
  deployOutput.quantSummary.textContent = `W=${inputs.weightBytes} B/param, A=${inputs.activationBytes} B, KV/C=${inputs.kvBytes} B/elem, scheduler max_num_seqs=${inputs.schedulerMaxSeqs}`;
}

function updateDeploymentCalculator() { try { Object.values(deployFields).forEach((field) => field.classList?.remove("is-invalid")); const inputs = readDeploymentInputs(); renderDeployment(calculateDeploymentMemory(inputs), inputs); } catch (error) { renderDeploymentInvalid(error.message); } }
function applyHardwarePreset(name) { if (name === "custom") return; const preset = HARDWARE_PRESETS[name]; if (!preset) return; deployFields.gpuMemoryGiB.value = preset.gpuMemoryGiB; deployFields.gpuCount.value = preset.gpuCount; deployFields.gpuUtil.value = preset.gpuUtil; }
function applyModelPreset(name) { if (name === "custom") return; const preset = MODEL_PRESETS[name]; if (!preset) return; deployFields.hardwarePreset.value = preset.hardwarePreset; applyHardwarePreset(preset.hardwarePreset); deployFields.layers.value = preset.layers; deployFields.kvHeads.value = preset.kvHeads; deployFields.kDim.value = preset.kDim; deployFields.vDim.value = preset.vDim; deployFields.maxModelLen.value = preset.maxModelLen; deployFields.fullContextSeqs.value = preset.fullContextSeqs; deployFields.schedulerMaxSeqs.value = preset.schedulerMaxSeqs; deployFields.blockSize.value = preset.blockSize; deployFields.tp.value = preset.tp; deployFields.pp.value = preset.pp; deployFields.dp.value = preset.dp; deployFields.kvCp.value = preset.kvCp; deployFields.totalParamsB.value = preset.totalParamsB; deployFields.weightBytes.value = preset.weightBytes; deployFields.kvBytes.value = preset.kvBytes; deployFields.activationBytes.value = preset.activationBytes; deployFields.runtimeReserveGiB.value = preset.runtimeReserveGiB; deployFields.expertWeightPercent.value = preset.expertWeightPercent; deployFields.expertParallel.checked = preset.expertParallel; deployFields.checkpointGiB.value = ""; }

function readBasicInputs() { return { layers: readNumber(basicFields.layers, "Layers", { integer: true, min: 1 }), kvHeads: readNumber(basicFields.kvHeads, "KV heads", { integer: true, min: 1 }), context: readNumber(basicFields.context, "Context", { integer: true, min: 1 }), batch: readNumber(basicFields.batch, "Batch", { integer: true, min: 1 }), headDim: readNumber(basicFields.headDim, "Head dim", { integer: true, min: 1 }), dtype: basicFields.dtype.value }; }
function renderBasicInvalid(message) { basicOutput.error.textContent = message; basicOutput.bytes.textContent = "--"; basicOutput.mib.textContent = "--"; basicOutput.gib.textContent = "--"; }
function updateBasicCalculator() { try { Object.values(basicFields).forEach((field) => field.classList.remove("is-invalid")); basicOutput.error.textContent = ""; const bytes = calculateBasicKvCache(readBasicInputs()); basicOutput.bytes.textContent = formatNumber(bytes, 0); basicOutput.mib.textContent = formatNumber(bytesToMiB(bytes), 2); basicOutput.gib.textContent = formatNumber(bytesToGiB(bytes), 4); } catch (error) { renderBasicInvalid(error.message); } }
function setMode(mode) { document.querySelectorAll(".mode-tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.mode === mode)); document.querySelectorAll(".mode-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `${mode}-mode`)); }

document.querySelectorAll(".mode-tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
deployFields.modelPreset.addEventListener("change", () => { applyModelPreset(deployFields.modelPreset.value); updateDeploymentCalculator(); });
deployFields.hardwarePreset.addEventListener("change", () => { applyHardwarePreset(deployFields.hardwarePreset.value); deployFields.modelPreset.value = "custom"; updateDeploymentCalculator(); });
document.querySelector("#deployment-form").addEventListener("input", (event) => { if (event.target !== deployFields.modelPreset && event.target !== deployFields.hardwarePreset) deployFields.modelPreset.value = "custom"; updateDeploymentCalculator(); });
document.querySelector("#deployment-form").addEventListener("change", updateDeploymentCalculator);
document.querySelector("#basic-form").addEventListener("input", updateBasicCalculator);
document.querySelector("#basic-form").addEventListener("change", updateBasicCalculator);

applyModelPreset("glm52-w4a8c8");
updateDeploymentCalculator();
updateBasicCalculator();
