const HEATMAPS_TO_DISPLAY = 1;
const heatmapContainer = document.getElementById("heatmap-container");
const template = document.getElementById("heatmap-template");
const headSlider = document.getElementById("head-slider");
const headValue = document.getElementById("head-value");
const layerSlider = document.getElementById("layer-slider");
const layerValue = document.getElementById("layer-value");
const promptText = document.getElementById("prompt-text");

let metadata = null;
const attentionCache = new Map();

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

    if (metadata.prompt) {
        promptText.textContent = `Prompt: ${metadata.prompt}`;
    } else {
        promptText.textContent = "";
    }
}

function renderHeatmap(element, tokens, matrix, titleSuffix, zRange) {
    const size = element.clientWidth || element.clientHeight || 600;

    const layout = {
        margin: { t: 30, l: 150, r: 10, b: 120 },
        xaxis: {
            tickmode: "array",
            tickvals: tokens.map((_, idx) => idx),
            ticktext: tokens,
            tickangle: -45,
            automargin: true,
            constrain: "domain",
        },
        yaxis: {
            tickmode: "array",
            tickvals: tokens.map((_, idx) => idx),
            ticktext: tokens,
            automargin: true,
            scaleanchor: "x",
            scaleratio: 1,
        },
        title: { text: titleSuffix, font: { size: 14 } },
        height: size,
    };

    const trace = {
        z: matrix,
        type: "heatmap",
        colorscale: "Viridis",
        colorbar: { title: "Attention" },
        zmin: zRange.min,
        zmax: zRange.max,
    };

    Plotly.react(element, [trace], layout, { displaylogo: false, responsive: true });
}

function computeRange(matrixA, matrixB) {
    let minVal = Number.POSITIVE_INFINITY;
    let maxVal = Number.NEGATIVE_INFINITY;

    const update = (matrix) => {
        for (const row of matrix) {
            for (const value of row) {
                if (value < minVal) minVal = value;
                if (value > maxVal) maxVal = value;
            }
        }
    };

    update(matrixA);
    update(matrixB);

    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
        minVal = 0;
        maxVal = 1;
    } else if (minVal === maxVal) {
        maxVal = minVal + 1e-6;
    }

    return { min: minVal, max: maxVal };
}

async function render() {
    const head = Number(headSlider.value);
    const layer = clampLayer(Number(layerSlider.value));
    layerSlider.value = layer;
    layerValue.textContent = layer;
    headValue.textContent = head;

    ensureHeatmapRows();

    const rows = heatmapContainer.querySelectorAll(".layer-group");
    const tokens = metadata.tokens;

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
                const model = panel.querySelector(".heatmap").dataset.model;
                const heatmapElement = panel.querySelector(".heatmap");
                const matrix = model === "A" ? data.model_a : data.model_b;
                const titleSuffix = `Head ${head}`;
                renderHeatmap(heatmapElement, tokens, matrix, titleSuffix, range);
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

window.addEventListener("resize", () => {
    render();
});

init();
