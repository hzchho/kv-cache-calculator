const DTYPE_BYTES = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
  int2: 0.25,
};

const PRESETS = {
  gqa: { layers: 32, kvHeads: 8, context: 4096, batch: 4, headDim: 128, dtype: "fp16" },
  mha: { layers: 32, kvHeads: 32, context: 4096, batch: 1, headDim: 128, dtype: "fp16" },
  long: { layers: 32, kvHeads: 8, context: 32768, batch: 1, headDim: 128, dtype: "fp16" },
  batch: { layers: 32, kvHeads: 8, context: 4096, batch: 16, headDim: 128, dtype: "fp16" },
};

const form = document.querySelector("#calculator-form");
const fields = {
  layers: document.querySelector("#layers"),
  kvHeads: document.querySelector("#kv-heads"),
  context: document.querySelector("#context"),
  batch: document.querySelector("#batch"),
  headDim: document.querySelector("#head-dim"),
  dtype: document.querySelector("#dtype"),
};

const output = {
  bytes: document.querySelector("#result-bytes"),
  mib: document.querySelector("#result-mib"),
  gib: document.querySelector("#result-gib"),
  error: document.querySelector("#error-message"),
  insight: document.querySelector("#insight-box"),
  dtypeChart: document.querySelector("#dtype-chart"),
  headsComparison: document.querySelector("#heads-comparison"),
};

function calculateKvCacheBytes({ batch, layers, kvHeads, context, headDim, dtype }) {
  const bytesPerElement = DTYPE_BYTES[dtype];

  if (bytesPerElement === undefined) {
    throw new Error(`不支持的 dtype: ${dtype}`);
  }

  // KV Cache 同时保存 Key 和 Value 两份张量，所以公式最前面要乘以 2。
  // 其余变量 B、L、Hkv、T、d_head 都会线性放大 KV Cache 的理论显存。
  return 2 * batch * layers * kvHeads * context * headDim * bytesPerElement;
}

function bytesToMiB(bytes) {
  return bytes / 1024 ** 2;
}

function bytesToGiB(bytes) {
  return bytes / 1024 ** 3;
}

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function readPositiveInteger(field, label) {
  const rawValue = field.value.trim();
  const value = Number(rawValue);

  field.classList.remove("is-invalid");

  if (rawValue === "" || !Number.isInteger(value) || value <= 0) {
    field.classList.add("is-invalid");
    throw new Error(`${label} 必须是正整数`);
  }

  return value;
}

function readInputs() {
  return {
    layers: readPositiveInteger(fields.layers, "模型层数 L"),
    kvHeads: readPositiveInteger(fields.kvHeads, "KV head 数 Hkv"),
    context: readPositiveInteger(fields.context, "上下文长度 T"),
    batch: readPositiveInteger(fields.batch, "Batch / 并发 B"),
    headDim: readPositiveInteger(fields.headDim, "Head dimension d_head"),
    dtype: fields.dtype.value,
  };
}

function renderResults(result) {
  output.bytes.textContent = formatNumber(result.bytes, 0);
  output.mib.textContent = formatNumber(bytesToMiB(result.bytes), 2);
  output.gib.textContent = formatNumber(bytesToGiB(result.bytes), 4);
}

function renderInvalidResults() {
  output.bytes.textContent = "--";
  output.mib.textContent = "--";
  output.gib.textContent = "--";
  output.insight.textContent = "请先修正输入参数，结果会在参数合法后自动更新。";
  output.dtypeChart.innerHTML = "";
  output.headsComparison.innerHTML = "";
}

function renderComparisons(inputs) {
  const dtypeRows = Object.entries(DTYPE_BYTES).map(([dtype]) => {
    const bytes = calculateKvCacheBytes({ ...inputs, dtype });
    return { dtype, bytes, gib: bytesToGiB(bytes) };
  });
  const maxBytes = Math.max(...dtypeRows.map((row) => row.bytes));

  output.dtypeChart.innerHTML = dtypeRows
    .map((row) => {
      const width = Math.max((row.bytes / maxBytes) * 100, 2);
      return `
        <div class="bar-row">
          <span class="bar-label">${row.dtype}</span>
          <div class="bar-track" aria-label="${row.dtype} ${formatNumber(row.gib, 4)} GiB">
            <div class="bar-fill" style="width: ${width}%"></div>
          </div>
          <span class="bar-value">${formatNumber(row.gib, 4)} GiB</span>
        </div>
      `;
    })
    .join("");

  const mhaHeads = Math.max(inputs.kvHeads, 32);
  const currentBytes = calculateKvCacheBytes(inputs);
  const mhaBytes = calculateKvCacheBytes({ ...inputs, kvHeads: mhaHeads });
  const ratio = currentBytes / mhaBytes;

  output.headsComparison.innerHTML = `
    <article class="comparison-card">
      <strong>当前配置：Hkv = ${inputs.kvHeads}</strong>
      <span>${formatNumber(bytesToGiB(currentBytes), 4)} GiB。减少 KV heads 会直接降低每层缓存的 K/V 张量规模。</span>
    </article>
    <article class="comparison-card">
      <strong>MHA 参考：Hkv = ${mhaHeads}</strong>
      <span>${formatNumber(bytesToGiB(mhaBytes), 4)} GiB。当前配置约为该参考的 ${formatNumber(ratio * 100, 1)}%。</span>
    </article>
  `;
}

function renderInsight(inputs, bytes) {
  const dtypeBytes = DTYPE_BYTES[inputs.dtype];
  output.insight.textContent =
    `当前 batch/concurrency=${inputs.batch}、context=${inputs.context}、dtype=${inputs.dtype} ` +
    `(${dtypeBytes} bytes/element)。KV Cache 主要随并发数、上下文长度和 KV heads 线性增长；GQA/MQA 的收益正来自减少 Hkv。`;
}

function clearValidationState() {
  Object.values(fields).forEach((field) => field.classList.remove("is-invalid"));
  output.error.textContent = "";
}

function updateCalculator() {
  try {
    clearValidationState();
    const inputs = readInputs();
    const bytes = calculateKvCacheBytes(inputs);
    renderResults({ bytes });
    renderComparisons(inputs);
    renderInsight(inputs, bytes);
  } catch (error) {
    renderInvalidResults();
    output.error.textContent = error.message;
  }
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }

  fields.layers.value = preset.layers;
  fields.kvHeads.value = preset.kvHeads;
  fields.context.value = preset.context;
  fields.batch.value = preset.batch;
  fields.headDim.value = preset.headDim;
  fields.dtype.value = preset.dtype;
  updateCalculator();
}

form.addEventListener("input", updateCalculator);
form.addEventListener("change", updateCalculator);

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

updateCalculator();

