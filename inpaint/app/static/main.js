const CANVAS_SIZE = 256;

const AppState = {
    imageId: null,
    faceData: null,
    isProcessing: false,
    brushWidth: 18,
    brushMode: 'mask',
    mousePressed: false,
    lastX: 0,
    lastY: 0,
    isDrawingAllowed: false
};

const History = {
    stack: [],
    step: -1,

    push(data) {
        this.step += 1;
        this.stack = this.stack.slice(0, this.step);
        this.stack.push(data);
        return this.step;
    },

    update(stepId, key, value) {
        const index = Number(stepId);
        if (this.stack[index]) {
            this.stack[index][key] = value;
        }
    },

    undo() {
        if (this.step > 0) {
            this.step -= 1;
        }
        return this.stack[this.step];
    },

    redo() {
        if (this.step < this.stack.length - 1) {
            this.step += 1;
        }
        return this.stack[this.step];
    },

    clear() {
        this.stack = [];
        this.step = -1;
    }
};

function byId(id) {
    return document.getElementById(id);
}

function showToast(message, type = 'success') {
    const container = byId('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        setTimeout(() => toast.remove(), 180);
    }, 3200);
}

function setStatus(message) {
    byId('status-line').textContent = message;
}

function setModelStatus(message) {
    byId('model-status').textContent = message;
}

function fillCanvas(canvas, img) {
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    if (img) {
        ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    }
}

function drawImageCover(ctx, img, size) {
    const scale = Math.max(size / img.width, size / img.height);
    const width = img.width * scale;
    const height = img.height * scale;
    const x = (size - width) / 2;
    const y = (size - height) / 2;
    ctx.drawImage(img, x, y, width, height);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Image could not be loaded'));
        img.src = src;
    });
}

function dataURItoBlob(dataURI) {
    const binary = atob(dataURI.split(',')[1]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: 'image/png' });
}

function blobToDataURI(blob) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = event => resolve(event.target.result);
        reader.readAsDataURL(blob);
    });
}

async function requestJson(response) {
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || response.statusText);
    }
    if (data.error) {
        throw new Error(data.error);
    }
    return data;
}

const Api = {
    async sendImage(canvas) {
        const formData = new FormData();
        formData.append('image', dataURItoBlob(canvas.toDataURL('image/png')), 'image.png');
        return requestJson(await fetch('/add_image', { method: 'POST', body: formData }));
    },

    async sendMask(imageId, stepId, canvas) {
        const formData = new FormData();
        formData.append('image_id', imageId);
        formData.append('step_id', String(stepId));
        formData.append('mask', dataURItoBlob(canvas.toDataURL('image/png')), 'mask.png');
        return requestJson(await fetch('/apply_mask', { method: 'POST', body: formData }));
    },

    async detectFace(canvas) {
        const formData = new FormData();
        formData.append('image', dataURItoBlob(canvas.toDataURL('image/png')), 'image.png');
        return requestJson(await fetch('/detect_face', { method: 'POST', body: formData }));
    }
};

function computeMaskCoverage() {
    const canvas = byId('canvas-mask');
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3] > 0 && Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) > 0) {
            painted += 1;
        }
    }
    return painted / (canvas.width * canvas.height);
}

function updateMaskCoverage(coverage = computeMaskCoverage()) {
    const pct = Math.round(coverage * 100);
    byId('mask-coverage').textContent = `Mask ${pct}%`;
    if (coverage >= 0.995) {
        setStatus('Full mask: the model has no source pixels to copy');
    }
}

function updateCompare(value) {
    const compare = byId('canvas-compare-source');
    const handle = byId('compare-handle');
    compare.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    handle.style.left = `${value}%`;
}

function setProcessing(active) {
    AppState.isProcessing = active;
    byId('loading-overlay').classList.toggle('active', active);
    byId('result-panel').classList.toggle('processing', active);
    setModelStatus(active ? 'Running' : 'Idle');
}

function setBrushMode(mode) {
    AppState.brushMode = mode;
    document.querySelectorAll('.tool-btn[data-mode]').forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === mode);
    });
}

const DrawEngine = {
    init(maskCanvas, outCanvas) {
        this.maskCanvas = maskCanvas;
        this.maskCtx = maskCanvas.getContext('2d');
        this.outCanvas = outCanvas;
        this.bindEvents();
    },

    bindEvents() {
        const canvas = this.maskCanvas;
        canvas.addEventListener('mousedown', event => this.onStart(event));
        canvas.addEventListener('mousemove', event => this.onMove(event));
        canvas.addEventListener('mouseup', () => this.onEnd());
        canvas.addEventListener('mouseleave', () => this.onEnd());
        canvas.addEventListener('touchstart', event => {
            event.preventDefault();
            this.onStart(event);
        }, { passive: false });
        canvas.addEventListener('touchmove', event => {
            event.preventDefault();
            this.onMove(event);
        }, { passive: false });
        canvas.addEventListener('touchend', event => {
            event.preventDefault();
            this.onEnd();
        }, { passive: false });
    },

    getCoords(event) {
        const rect = this.maskCanvas.getBoundingClientRect();
        const point = event.touches && event.touches.length ? event.touches[0] : event;
        return {
            x: (point.clientX - rect.left) / rect.width * this.maskCanvas.width,
            y: (point.clientY - rect.top) / rect.height * this.maskCanvas.height
        };
    },

    withBrush(mode, draw) {
        this.maskCtx.save();
        this.maskCtx.globalCompositeOperation = mode === 'erase' ? 'destination-out' : 'source-over';
        this.maskCtx.fillStyle = 'white';
        this.maskCtx.strokeStyle = 'white';
        draw();
        this.maskCtx.restore();
    },

    onStart(event) {
        if (!AppState.isDrawingAllowed || AppState.isProcessing) {
            return;
        }
        AppState.mousePressed = true;
        const coords = this.getCoords(event);
        this.drawDot(coords.x, coords.y, AppState.brushMode);
        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
        updateMaskCoverage();
    },

    onMove(event) {
        if (!AppState.mousePressed) {
            return;
        }
        const coords = this.getCoords(event);
        this.drawLine(AppState.lastX, AppState.lastY, coords.x, coords.y, AppState.brushMode);
        AppState.lastX = coords.x;
        AppState.lastY = coords.y;
        updateMaskCoverage();
    },

    onEnd() {
        if (!AppState.mousePressed) {
            return;
        }
        AppState.mousePressed = false;
        this.applyMask();
    },

    drawDot(x, y, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.arc(x, y, AppState.brushWidth / 2, 0, Math.PI * 2);
            this.maskCtx.fill();
        });
    },

    drawLine(x1, y1, x2, y2, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.lineWidth = AppState.brushWidth;
            this.maskCtx.lineJoin = 'round';
            this.maskCtx.lineCap = 'round';
            this.maskCtx.moveTo(x1, y1);
            this.maskCtx.lineTo(x2, y2);
            this.maskCtx.stroke();
        });
    },

    drawEllipse(cx, cy, rx, ry, mode = 'mask') {
        this.withBrush(mode, () => {
            this.maskCtx.beginPath();
            this.maskCtx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
            this.maskCtx.fill();
        });
        updateMaskCoverage();
    },

    async applyMask() {
        if (!AppState.imageId || AppState.isProcessing) {
            return;
        }

        const coverage = computeMaskCoverage();
        const stepId = History.push({ mask: this.maskCanvas.toDataURL('image/png') });
        updateMaskCoverage(coverage);
        setProcessing(true);
        setStatus(coverage >= 0.995 ? 'Reconstructing from learned face prior' : 'Reconstructing masked region');

        try {
            const data = await Api.sendMask(AppState.imageId, stepId, this.maskCanvas);
            if (AppState.imageId !== data.image_id) {
                return;
            }
            History.update(data.step_id, 'result', data.result);
            History.update(data.step_id, 'coverage', data.mask_coverage);
            if (History.step === Number(data.step_id)) {
                const img = await loadImage(data.result);
                fillCanvas(this.outCanvas, img);
                setModelStatus('Ready');
            }
            updateMaskCoverage(data.mask_coverage);
            setStatus('Reconstruction ready');
        } catch (error) {
            showToast(`Reconstruction failed: ${error.message}`, 'error');
            setStatus('Reconstruction failed');
        } finally {
            setProcessing(false);
        }
    }
};

async function applyInitialState(img) {
    const canvasIn = byId('canvas-in');
    const canvasMask = byId('canvas-mask');
    const canvasOut = byId('canvas-out');
    const canvasCompare = byId('canvas-compare-source');

    fillCanvas(canvasIn, img);
    fillCanvas(canvasOut, img);
    fillCanvas(canvasCompare, img);
    fillCanvas(canvasMask, null);

    byId('empty-state').classList.add('hidden');
    byId('face-tools').classList.add('hidden');
    AppState.faceData = null;
    AppState.imageId = null;
    AppState.isDrawingAllowed = false;

    History.clear();
    History.push({
        mask: canvasMask.toDataURL('image/png'),
        result: canvasIn.toDataURL('image/png'),
        coverage: 0
    });
    updateMaskCoverage(0);
    setStatus('Uploading source image');

    try {
        const data = await Api.sendImage(canvasIn);
        AppState.imageId = data.image_id;
        AppState.isDrawingAllowed = true;
        setStatus('Ready');
        detectFaceRegions();
    } catch (error) {
        showToast(`Upload failed: ${error.message}`, 'error');
        setStatus('Upload failed');
    }
}

async function pickRandom() {
    setStatus('Loading sample image');
    try {
        const img = await loadImage(`/pick_random?t=${Date.now()}`);
        await applyInitialState(img);
    } catch (error) {
        showToast(`Sample failed: ${error.message}`, 'error');
        setStatus('Upload an image to begin');
    }
}

async function loadFromFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('Please choose an image file', 'error');
        return;
    }

    const dataURI = await blobToDataURI(file);
    const img = await loadImage(dataURI);
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    drawImageCover(canvas.getContext('2d'), img, CANVAS_SIZE);
    const resized = await loadImage(canvas.toDataURL('image/png'));
    await applyInitialState(resized);
}

async function detectFaceRegions() {
    try {
        const data = await Api.detectFace(byId('canvas-in'));
        if (data.faces && data.faces.length > 0) {
            AppState.faceData = data.faces[0];
            byId('face-tools').classList.remove('hidden');
        }
    } catch (error) {
        console.warn('Face detection failed:', error);
    }
}

function applyFaceRegionMask(regionName) {
    if (!AppState.faceData || !AppState.isDrawingAllowed || AppState.isProcessing) {
        return;
    }
    const region = AppState.faceData.regions[regionName];
    if (!region) {
        return;
    }
    const cx = (region.x + region.w / 2) * CANVAS_SIZE;
    const cy = (region.y + region.h / 2) * CANVAS_SIZE;
    const rx = (region.w / 2) * CANVAS_SIZE;
    const ry = (region.h / 2) * CANVAS_SIZE;
    DrawEngine.drawEllipse(cx, cy, rx, ry, 'mask');
    DrawEngine.applyMask();
}

async function restoreHistoryState(data) {
    if (!data) {
        return;
    }
    const maskImg = await loadImage(data.mask);
    fillCanvas(byId('canvas-mask'), maskImg);
    if (data.result) {
        const resultImg = await loadImage(data.result);
        fillCanvas(byId('canvas-out'), resultImg);
    }
    updateMaskCoverage(data.coverage);
}

async function doUndo() {
    await restoreHistoryState(History.undo());
}

async function doRedo() {
    await restoreHistoryState(History.redo());
}

async function doClear() {
    const canvasMask = byId('canvas-mask');
    canvasMask.getContext('2d').clearRect(0, 0, canvasMask.width, canvasMask.height);
    const source = await loadImage(byId('canvas-in').toDataURL('image/png'));
    fillCanvas(byId('canvas-out'), source);
    History.clear();
    History.push({
        mask: canvasMask.toDataURL('image/png'),
        result: byId('canvas-in').toDataURL('image/png'),
        coverage: 0
    });
    updateMaskCoverage(0);
    setStatus('Mask cleared');
}

function doFillMask() {
    if (!AppState.isDrawingAllowed || AppState.isProcessing) {
        return;
    }
    const canvasMask = byId('canvas-mask');
    const ctx = canvasMask.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasMask.width, canvasMask.height);
    updateMaskCoverage(1);
    showToast('Full mask uses learned prior only', 'warning');
    DrawEngine.applyMask();
}

function saveResult() {
    if (!AppState.imageId) {
        showToast('Load an image first', 'error');
        return;
    }
    const link = document.createElement('a');
    link.href = byId('canvas-out').toDataURL('image/png');
    link.download = 'facerestore_result.png';
    link.click();
}

function initDragDrop() {
    let dragCounter = 0;
    const dropZone = byId('drop-zone');

    document.addEventListener('dragenter', (event) => {
        event.preventDefault();
        dragCounter += 1;
        dropZone.classList.add('active');
    });

    document.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dragCounter -= 1;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dropZone.classList.remove('active');
        }
    });

    document.addEventListener('dragover', event => event.preventDefault());

    document.addEventListener('drop', (event) => {
        event.preventDefault();
        dragCounter = 0;
        dropZone.classList.remove('active');
        const file = event.dataTransfer.files[0];
        loadFromFile(file);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    DrawEngine.init(byId('canvas-mask'), byId('canvas-out'));

    document.querySelectorAll('.btn-load').forEach((button) => {
        button.addEventListener('click', () => byId('file-upload').click());
    });
    byId('file-upload').addEventListener('change', (event) => {
        loadFromFile(event.target.files[0]);
        event.target.value = '';
    });

    document.querySelectorAll('.btn-pick-random').forEach((button) => {
        button.addEventListener('click', pickRandom);
    });
    document.querySelectorAll('.btn-save').forEach((button) => {
        button.addEventListener('click', saveResult);
    });

    document.querySelectorAll('.tool-btn[data-mode]').forEach((button) => {
        button.addEventListener('click', () => setBrushMode(button.dataset.mode));
    });

    byId('brush-slider').addEventListener('input', (event) => {
        AppState.brushWidth = Number(event.target.value);
        byId('brush-size').textContent = event.target.value;
    });

    byId('mask-opacity').addEventListener('input', (event) => {
        byId('canvas-mask').style.opacity = String(Number(event.target.value) / 100);
        byId('opacity-value').textContent = event.target.value;
    });

    byId('compare-slider').addEventListener('input', event => updateCompare(Number(event.target.value)));
    updateCompare(Number(byId('compare-slider').value));

    byId('btn-undo').addEventListener('click', doUndo);
    byId('btn-redo').addEventListener('click', doRedo);
    byId('btn-clear').addEventListener('click', doClear);
    byId('btn-fill').addEventListener('click', doFillMask);

    document.querySelectorAll('.region-btn').forEach((button) => {
        button.addEventListener('click', () => applyFaceRegionMask(button.dataset.region));
    });

    document.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
            event.preventDefault();
            doUndo();
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
            event.preventDefault();
            doRedo();
        }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
            event.preventDefault();
            saveResult();
        }
    });

    initDragDrop();
    pickRandom();
});
