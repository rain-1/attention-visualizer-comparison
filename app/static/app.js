const HEATMAPS_TO_DISPLAY = 1;
const ZERO_CUTOFF_FACTOR = 1e-6;
const MIN_ZERO_CUTOFF = 1e-9;
const heatmapContainer = document.getElementById("heatmap-container");
const template = document.getElementById("heatmap-template");
const headSlider = document.getElementById("head-slider");
const headValue = document.getElementById("head-value");
const layerSlider = document.getElementById("layer-slider");
const layerValue = document.getElementById("layer-value");
const promptText = document.getElementById("prompt-text");
const contrastSlider = document.getElementById("contrast-slider");
const contrastValue = document.getElementById("contrast-value");

let metadata = null;
const attentionCache = new Map();

function escapeHtml(str) {
    if (typeof str !== "string") return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function clampLayer(value) {
    if (!metadata) return 0;
    const numericValue = Number(value);
    const safeValue = Number.isNaN(numericValue) ? 0 : numericValue;
    const maxLayer = Math.max(0, metadata.num_layers - 1);
    return Math.min(Math.max(0, safeValue), maxLayer);
}

async function fetchMetadata() {
    const response = await fetch("/api/metadata");
    if (!response.ok) {
        throw new Error("Failed to load metadata");
    }
    return response.json();
}

async function fetchAttention(layer, head) {
    const key = `${layer}-${head}`;
    if (attentionCache.has(key)) {
        return attentionCache.get(key);
    }
    const response = await fetch(`/api/attention?layer=${layer}&head=${head}`);
    if (!response.ok) {
        throw new Error(`Failed to load attention for layer ${layer}, head ${head}`);
    }
    const data = await response.json();
    attentionCache.set(key, data);
    return data;
}

function ensureHeatmapRows() {
    while (heatmapContainer.children.length > HEATMAPS_TO_DISPLAY) {
        heatmapContainer.removeChild(heatmapContainer.lastElementChild);
    }

    const existing = heatmapContainer.children.length;
    for (let i = existing; i < HEATMAPS_TO_DISPLAY; i += 1) {
        const clone = template.content.cloneNode(true);
        heatmapContainer.appendChild(clone);
    }
}

function updateControls() {
    headSlider.max = metadata.num_heads - 1;
    headSlider.value = Math.min(headSlider.value, metadata.num_heads - 1);
    headSlider.step = 1;
    headValue.textContent = headSlider.value;

    const maxLayer = Math.max(0, metadata.num_layers - 1);
    layerSlider.max = maxLayer;
    layerSlider.value = clampLayer(layerSlider.value);
    layerSlider.step = 1;
    layerValue.textContent = layerSlider.value;

    const contrast = Number.parseFloat(contrastSlider.value) || 1;
    contrastValue.textContent = `${contrast.toFixed(1)}×`;

    if (metadata.models) {
        const promptLines = [];
        const modelEntries = [
            ["A", metadata.models.A],
            ["B", metadata.models.B],
        ];
        for (const [label, model] of modelEntries) {
            if (model && model.prompt) {
                promptLines.push(`<strong>Model ${label} Prompt:</strong> ${escapeHtml(model.prompt)}`);
            }
        }
        promptText.innerHTML = promptLines.join("<br />");
    } else {
        promptText.textContent = "";
    }
}

function buildAxisLabels(tokens, matrixSize) {
    if (Array.isArray(tokens) && tokens.length === matrixSize) {
        return tokens;
    }
    return Array.from({ length: matrixSize }, (_, idx) => idx.toString());
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}

function formatAttentionValue(value) {
    if (!Number.isFinite(value)) return "0";
    const abs = Math.abs(value);
    if (abs === 0) {
        return "0";
    }
    if (abs >= 1) {
        return value.toFixed(2);
    }
    if (abs >= 0.01) {
        return value.toFixed(3);
    }
    return value.toExponential(2);
}

function buildColorbar(range, contrast) {
    const tickPositions = [0, 0.25, 0.5, 0.75, 1];
    const span = range.max - range.min;
    const safeContrast = Math.max(contrast, 0.01);
    const inverseContrast = 1 / safeContrast;
    const tickTexts = tickPositions.map((position) => {
        const normalized = Math.pow(position, inverseContrast);
        const actualValue = normalized * span + range.min;
        return formatAttentionValue(actualValue);
    });
    return {
        title: "Attention",
        tickmode: "array",
        tickvals: tickPositions,
        ticktext: tickTexts,
    };
}

function sanitizeValue(value, zeroCutoff) {
    if (!Number.isFinite(value)) return 0;
    if (value <= 0) return 0;
    if (value <= zeroCutoff) return 0;
    return value;
}

function sanitizeMatrix(matrix, zeroCutoff) {
    return matrix.map((row) => row.map((value) => sanitizeValue(value, zeroCutoff)));
}

function transformMatrix(matrix, range, contrast) {
    const span = range.max - range.min;
    const safeSpan = span === 0 ? 1 : span;
    const safeContrast = Math.max(contrast, 0.01);
    return matrix.map((row) =>
        row.map((value) => {
            if (value <= 0) {
                return 0;
            }
            const normalized = clamp01((value - range.min) / safeSpan);
            return Math.pow(normalized, safeContrast);
        })
    );
}

function renderHeatmap(element, tokens, matrix, titleSuffix, zRange, contrast) {
    const size = element.clientWidth || element.clientHeight || 600;

    const rowCount = Array.isArray(matrix) ? matrix.length : 0;
    const colCount = rowCount > 0 && Array.isArray(matrix[0]) ? matrix[0].length : rowCount;
    const xAxisLabels = buildAxisLabels(tokens, colCount);
    const yAxisLabels = buildAxisLabels(tokens, rowCount);
    const xTickvals = xAxisLabels.map((_, idx) => idx);
    const yTickvals = yAxisLabels.map((_, idx) => idx);

    const layout = {
        margin: { t: 30, l: 150, r: 10, b: 120 },
        xaxis: {
            tickmode: "array",
            tickvals: xTickvals,
            ticktext: xAxisLabels,
            tickangle: -45,
            automargin: true,
            constrain: "domain",
        },
        yaxis: {
            tickmode: "array",
            tickvals: yTickvals,
            ticktext: yAxisLabels,
            automargin: true,
            scaleanchor: "x",
            scaleratio: 1,
            autorange: "reversed",
        },
        title: { text: titleSuffix, font: { size: 14 } },
        height: size,
    };

    const transformedMatrix = transformMatrix(matrix, zRange, contrast);

    const trace = {
        z: transformedMatrix,
        x: xAxisLabels,
        y: yAxisLabels,
        type: "heatmap",
        colorscale: "Viridis",
        colorbar: buildColorbar(zRange, contrast),
        zmin: 0,
        zmax: 1,
        customdata: matrix,
        hovertemplate: "Target: %{y}<br>Source: %{x}<br>Attention: %{customdata:.4f}<extra></extra>",
    };

    Plotly.react(element, [trace], layout, { displaylogo: false, responsive: true });
}

function computeRange(matrixA, matrixB) {
    let maxVal = 0;

    const update = (matrix) => {
        for (const row of matrix) {
            for (const value of row) {
                if (!Number.isFinite(value)) {
                    continue;
                }
                if (value <= 0) {
                    continue;
                }
                if (value > maxVal) {
                    maxVal = value;
                }
            }
        }
    };

    update(matrixA);
    update(matrixB);

    const zeroCutoff = Math.max(maxVal * ZERO_CUTOFF_FACTOR, MIN_ZERO_CUTOFF);

    return {
        min: 0,
        max: Number.isFinite(maxVal) && maxVal > 0 ? maxVal : 0,
        zeroCutoff,
    };
}

async function render() {
    const head = Number(headSlider.value);
    const layer = clampLayer(Number(layerSlider.value));
    const contrast = Number.parseFloat(contrastSlider.value) || 1;
    layerSlider.value = layer;
    layerValue.textContent = layer;
    headValue.textContent = head;
    contrastValue.textContent = `${contrast.toFixed(1)}×`;

    ensureHeatmapRows();

    const rows = heatmapContainer.querySelectorAll(".layer-group");
    const tokensByModel = {
        A: metadata.models?.A?.tokens ?? null,
        B: metadata.models?.B?.tokens ?? null,
    };

    for (let i = 0; i < rows.length; i += 1) {
        const layerIndex = layer + i;
        const rowElement = rows[i];
        const headerSpan = rowElement.querySelector(".layer-index");
        const panels = rowElement.querySelectorAll(".heatmap-panel");

        if (layerIndex >= metadata.num_layers) {
            rowElement.style.display = "none";
            continue;
        }

        rowElement.style.display = "block";
        headerSpan.textContent = layerIndex;

        try {
            const data = await fetchAttention(layerIndex, head);
            const range = computeRange(data.model_a, data.model_b);

            panels.forEach((panel) => {
                const heatmapElement = panel.querySelector(".heatmap");
                const model = heatmapElement.dataset.model;
                const rawMatrix = model === "A" ? data.model_a : data.model_b;
                const sanitizedMatrix = sanitizeMatrix(rawMatrix, range.zeroCutoff);
                const tokens = tokensByModel[model];
                const titleSuffix = `Head ${head}`;
                renderHeatmap(
                    heatmapElement,
                    tokens,
                    sanitizedMatrix,
                    titleSuffix,
                    range,
                    contrast,
                );
            });
        } catch (error) {
            panels.forEach((panel) => {
                const heatmapElement = panel.querySelector(".heatmap");
                heatmapElement.innerHTML = `<div class="error">${error.message}</div>`;
            });
        }
    }
}

async function init() {
    try {
        metadata = await fetchMetadata();
        updateControls();
        await render();
    } catch (error) {
        heatmapContainer.innerHTML = `<p class="error">${error.message}</p>`;
    }
}

headSlider.addEventListener("input", () => {
    render();
});

layerSlider.addEventListener("input", () => {
    render();
});

contrastSlider.addEventListener("input", () => {
    render();
});

window.addEventListener("resize", () => {
    render();
});

init();
