'use strict';

const canvas = document.getElementsByTagName('canvas')[0];
canvas.width = canvas.clientWidth;
canvas.height = canvas.clientHeight;

Array.prototype.getRandom = function () {
    return this[Math.floor(Math.random() * this.length)];
};

let idleSplats;

function idleSplatsFunction() {
    if (audioLessFrames > 10) {
        multipleSplats(Math.random() * config.RANDOM_AMOUNT + (config.RANDOM_AMOUNT / 2) + 1);
    }
}

let config = {
    SIM_RESOLUTION: 256,
    DYE_RESOLUTION: 1024,
    DENSITY_DISSIPATION: 0.97,
    VELOCITY_DISSIPATION: 0.98,
    PRESSURE_DISSIPATION: 0.8,
    PRESSURE_ITERATIONS: 20,
    CURL: 30,
    SPLAT_RADIUS: 0.3,
    SHADING: true,
    PAUSED: false,
    BACK_COLOR: {r: 0, g: 0, b: 0},
    TRANSPARENT: false,
    BLOOM: true,
    BLOOM_ITERATIONS: 8,
    BLOOM_RESOLUTION: 256,
    BLOOM_INTENSITY: 0.8,
    BLOOM_THRESHOLD: 0.6,
    BLOOM_SOFT_KNEE: 0.7,
    POINTER_COLOR: [{r: 0, g: 0.15, b: 0}],
    IDLE_SPLATS: false,
    RANDOM_AMOUNT: 10,
    RANDOM_INTERVAL: 1,
    SPLAT_ON_CLICK: true,
    SHOW_MOUSE_MOVEMENT: true,

    // audio stuff
    AUDIO_RESPONSE_ENABLED: true,
    AUDIO_SPLATS: true,
    AUDIO_SENSITIVITY: .1,
    AUDIO_THRESHOLD: .2,
    AUDIO_N_BINS: 10,
    AUDIO_N_SAMPLES: 0,
    AUDIO_FREQUENCY_SCALE: 2,
    // splat size
    AUDIO_SPLAT_SIZE_BASE: .2,
    AUDIO_SPLAT_SIZE_AMP: .2,
    // splat brightness
    AUDIO_SPLAT_BRIGHTNESS_BASE: 2,
    AUDIO_SPLAT_BRIGHTNESS_AMP: 2,
    // high/low filter
    AUDIO_CUTOFF_HIGH_LO: .7,
    // high response brightness mods
    AUDIO_COLOR_FILTER_AMP: 1,
    AUDIO_COLOR_FILTER_EXP: 1,
};

const audioEpsilon = .002;
const nFreq = 64; // number of audio frequencies per channel
let peakFilter = Array(config.AUDIO_N_BINS).fill(0);
let audioHigh = 0;
let audioLessFrames = 0;

function fetchProp(props, propName, cfgName, ifPresentCallback = null) {
    if (props[propName]) {
        const value = props[propName].value
        config[cfgName] = value;
        if (ifPresentCallback != null) ifPresentCallback(value);
        // console.log(`Prop: ${cfgName}: ${value}`);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    // noinspection JSUnusedGlobalSymbols
    window.wallpaperPropertyListener = {
        applyUserProperties: (properties) => {
            // audio props
            fetchProp(properties, "audio_response_enabled", "AUDIO_RESPONSE_ENABLED");
            fetchProp(properties, "audio_response_splats", "AUDIO_SPLATS");
            fetchProp(properties, "audio_response_sensitivity", "AUDIO_SENSITIVITY");
            fetchProp(properties, "audio_response_threshold", "AUDIO_THRESHOLD");
            fetchProp(properties, "audio_response_n_bins", "AUDIO_N_BINS", (nBins) => {
                peakFilter = Array(nBins).fill(0);
            });
            fetchProp(properties, "audio_response_frequency_scale", "AUDIO_FREQUENCY_SCALE");
            fetchProp(properties, "audio_response_n_samples", "AUDIO_N_SAMPLES");
            fetchProp(properties, "audio_response_splat_size_base", "AUDIO_SPLAT_SIZE_BASE");
            fetchProp(properties, "audio_response_splat_size_amp", "AUDIO_SPLAT_SIZE_AMP");
            fetchProp(properties, "audio_response_splat_brightness_base", "AUDIO_SPLAT_BRIGHTNESS_BASE");
            fetchProp(properties, "audio_response_splat_brightness_amp", "AUDIO_SPLAT_BRIGHTNESS_AMP");
            fetchProp(properties, "audio_response_cutoff_high_lo", "AUDIO_CUTOFF_HIGH_LO");
            fetchProp(properties, "audio_response_color_filter_amp", "AUDIO_COLOR_FILTER_AMP");
            fetchProp(properties, "audio_response_color_filter_exp", "AUDIO_COLOR_FILTER_EXP");

            fetchProp(properties, "bloom_intensity", "BLOOM_INTENSITY");
            fetchProp(properties, "bloom_threshold", "BLOOM_THRESHOLD");
            fetchProp(properties, "density_diffusion", "DENSITY_DISSIPATION");
            fetchProp(properties, "enable_bloom", "BLOOM");
            fetchProp(properties, "paused", "PAUSED");
            fetchProp(properties, "pressure_diffusion", "PRESSURE_DISSIPATION");
            fetchProp(properties, "shading", "SHADING");
            fetchProp(properties, "splat_radius", "SPLAT_RADIUS");
            fetchProp(properties, "velocity_diffusion", "VELOCITY_DISSIPATION");
            fetchProp(properties, "vorticity", "CURL");
            fetchProp(properties, "simulation_resolution", "SIM_RESOLUTION", initFrameBuffers);
            fetchProp(properties, "dye_resolution", "DYE_RESOLUTION", initFrameBuffers);
            fetchProp(properties, "background_color", null, (rawColor) => {
                let c = rawColor.split(" "),
                    r = Math.floor(c[0] * 255),
                    g = Math.floor(c[1] * 255),
                    b = Math.floor(c[2] * 255);
                document.body.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
                config.BACK_COLOR.r = r;
                config.BACK_COLOR.g = g;
                config.BACK_COLOR.b = b;
            });
            fetchProp(properties, "idle_random_splats", "IDLE_SPLATS", (v) => {
                if (v) {
                    idleSplats = setInterval(idleSplatsFunction, config.RANDOM_INTERVAL * 1000);
                } else {
                    clearInterval(idleSplats);
                }
            });
            fetchProp(properties, "random_splat_interval", "RANDOM_INTERVAL", () => {
                if (config.IDLE_SPLATS) {
                    clearInterval(idleSplats);
                    idleSplats = setInterval(idleSplatsFunction, config.RANDOM_INTERVAL * 1000);
                }
            });
            fetchProp(properties, "random_splat_amount", "RANDOM_AMOUNT", () => {
                if (config.IDLE_SPLATS) {
                    clearInterval(idleSplats);
                    idleSplats = setInterval(idleSplatsFunction, config.RANDOM_INTERVAL * 1000);
                }
            });
            fetchProp(properties, "splat_on_click", "SPLAT_ON_CLICK");
            fetchProp(properties, "show_mouse_movement", "SHOW_MOUSE_MOVEMENT");
        }
    };

    // noinspection JSUnresolvedFunction
    window.wallpaperRegisterAudioListener((audioArray) => {
        if (!config.AUDIO_RESPONSE_ENABLED) {
            audioLessFrames += 1;
            return;
        }

        let bins = Array(config.AUDIO_N_BINS).fill(0);

        for (let i = 0; i < config.AUDIO_N_BINS; i++) {
            const start = Math.max(0, Math.floor(nFreq * Math.pow((i - 1) / config.AUDIO_N_BINS, 2)));
            const end = Math.min(nFreq, Math.floor(nFreq * Math.pow((i + 2) / config.AUDIO_N_BINS, 2)));
            for (let j = start; j < end; j++) {
                bins[i] += (audioArray[j] + audioArray[nFreq + j]);
            }

            const binNFreq = (end - start)
            if (0 < binNFreq) bins[i] /= binNFreq;
            else console.warn(`Bin ${i} has frequency range of 0!`);
        }

        const avgAmp = bins.reduce((c, v) => c + v) / config.AUDIO_N_BINS;
        bins = bins.map((v) => Math.max(0, v - avgAmp));

        let max = Math.max.apply(null, bins);
        let frameAudioHigh = 0;

        if (0 < max) {
            bins = bins.map((v) => v / max);
            for (let i = 0; i < config.AUDIO_N_BINS; i++) {
                const v = bins[i];
                let thOrg = (v - peakFilter[i] - config.AUDIO_THRESHOLD);
                let th = Math.max(0, thOrg * (1 - Math.min(1, (peakFilter[i] + config.AUDIO_THRESHOLD))));
                if (th < audioEpsilon) th = 0;
                bins[i] = th;

                if (config.AUDIO_CUTOFF_HIGH_LO <= i / config.AUDIO_N_BINS) {
                    frameAudioHigh += v;
                }
            }
        }
        for (let i = 0; i < bins.length; i++) {
            if (0 < config.AUDIO_N_SAMPLES) {
                peakFilter[i] = (peakFilter[i] * config.AUDIO_N_SAMPLES + bins[i]) / (config.AUDIO_N_SAMPLES + 1);
            }
        }

        let isAudio = false;
        max = Math.max.apply(null, bins);
        if (0 < max) {
            for (let i = 0; i < config.AUDIO_N_BINS; i++) {
                const th = Math.min(1, bins[i] * config.AUDIO_SENSITIVITY);
                if (0 < th && config.AUDIO_SPLATS) {
                    const brightness = th + config.AUDIO_SPLAT_BRIGHTNESS_BASE * (1 - th);
                    const rBase = config.AUDIO_SPLAT_SIZE_BASE / 10;
                    const rVar = 1 - Math.pow(i / config.AUDIO_N_BINS, config.AUDIO_SPLAT_SIZE_AMP);
                    const radius = rBase * rVar;
                    const color = generateColor(brightness);
                    const x = canvas.width * Math.random();
                    const y = canvas.height * Math.random();
                    const dx = 1000 * (Math.random() - 0.5);
                    const dy = 1000 * (Math.random() - 0.5);
                    splat(x, y, dx, dy, color, radius);
                    isAudio = true;
                }
            }
        }

        if (isAudio) audioLessFrames = 0;
        const norm = Math.floor(config.AUDIO_N_BINS * (1 - config.AUDIO_CUTOFF_HIGH_LO));
        if (0 < norm) frameAudioHigh /= norm;
        audioHigh = (audioHigh * config.AUDIO_N_SAMPLES + frameAudioHigh) / (config.AUDIO_N_SAMPLES + 1);
    });
});

class pointerPrototype {
    constructor() {
        this.id = -1;
        this.x = 0;
        this.y = 0;
        this.dx = 0;
        this.dy = 0;
        this.down = false;
        // this.moved = false;
        this.color = generateColor(.15);
    }
}

let pointers = [];
let splatStack = [];
let bloomFrameBuffers = [];
pointers.push(new pointerPrototype());

const {gl, ext} = getWebGLContext(canvas);

if (isMobile()) {
    config.SHADING = false;
}
if (!ext.supportLinearFiltering) {
    config.SHADING = false;
    config.BLOOM = false;
}

function getWebGLContext(canvas) {
    const params = {alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false};

    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2)
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float');
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2) {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    } else {
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

function getSupportedFormat(gl, internalFormat, format, type) {
    if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
        switch (internalFormat) {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }

    return {
        internalFormat,
        format
    }
}

function supportRenderTextureFormat(gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status === gl.FRAMEBUFFER_COMPLETE;

}

function isMobile() {
    return /Mobi|Android/i.test(navigator.userAgent);
}

class GLProgram {
    constructor(vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = gl.createProgram();

        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
            throw gl.getProgramInfoLog(this.program);

        const uniformCount = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < uniformCount; i++) {
            const uniformName = gl.getActiveUniform(this.program, i).name;
            this.uniforms[uniformName] = gl.getUniformLocation(this.program, uniformName);
        }
    }

    bind() {
        gl.useProgram(this.program);
    }

    uniform(fun, name, ...args) {
        let id = this.uniforms[name];
        if (id != null) {
            gl[`uniform${fun}`](id, ...args);
        }
    }
}

function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        console.log(info);
        throw info;
    }

    return shader;
}

const shaderIncludeColorConversion = `
    vec3 rgb2hsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }
    
    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
`;

const shaderIncludeAudioResponseHelper = `
    uniform float uAudioHigh;
    uniform float uColorExp; // determines how much bright colors react to high peaks
    uniform float uColorAmp; // brightness amplitude multiplier after exponentiation
    
    ${shaderIncludeColorConversion}
    
    vec3 mapColor(vec3 color) {
        vec3 hsvColor = rgb2hsv(color);
        // try only high for now
        float v = hsvColor.z;
        v = pow(v, uColorExp) * uColorAmp * uAudioHigh;
        return vec3(color.xy, v);
    }
`;

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
    precision highp float;

    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;

    void main () {
        vUv = aPosition * 0.5 + 0.5;
        vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`);

const clearShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;

    void main () {
        gl_FragColor = value * texture2D(uTexture, vUv);
    }
`);

const colorShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;

    uniform vec4 color;

    void main () {
        gl_FragColor = color;
    }
`);

const backgroundShader = compileShader(gl.FRAGMENT_SHADER, `
    void main () {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
`);

const displayShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;
    
    ${shaderIncludeAudioResponseHelper}

    varying vec2 vUv;
    uniform sampler2D uTexture;

    void main () {
        vec3 C = texture2D(uTexture, vUv).rgb;
        float a = max(C.r, max(C.g, C.b));
        gl_FragColor = vec4(mapColor(C), a);
    }
`);

const displayBloomShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;
    
    ${shaderIncludeAudioResponseHelper}

    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform sampler2D uBloom;
    uniform sampler2D uDithering;
    uniform vec2 ditherScale;

    void main () {
        vec3 C = texture2D(uTexture, vUv).rgb;
        vec3 bloom = texture2D(uBloom, vUv).rgb;
        bloom = pow(bloom.rgb, vec3(1.0 / 2.2));
        C += bloom;
        float a = max(C.r, max(C.g, C.b));
        gl_FragColor = vec4(mapColor(C), a);
    }
`);

const displayShadingShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;
    
    ${shaderIncludeAudioResponseHelper}

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uTexture;
    uniform vec2 texelSize;

    void main () {
        vec3 L = texture2D(uTexture, vL).rgb;
        vec3 R = texture2D(uTexture, vR).rgb;
        vec3 T = texture2D(uTexture, vT).rgb;
        vec3 B = texture2D(uTexture, vB).rgb;
        vec3 C = texture2D(uTexture, vUv).rgb;

        float dx = length(R) - length(L);
        float dy = length(T) - length(B);

        vec3 n = normalize(vec3(dx, dy, length(texelSize)));
        vec3 l = vec3(0.0, 0.0, 1.0);

        float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
        C.rgb *= diffuse;

        float a = max(C.r, max(C.g, C.b));
        gl_FragColor = vec4(mapColor(C), a);
    }
`);

const displayBloomShadingShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;
    
    ${shaderIncludeAudioResponseHelper}

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uTexture;
    uniform sampler2D uBloom;
    uniform sampler2D uDithering;
    uniform vec2 ditherScale;
    uniform vec2 texelSize;

    void main () {
        vec3 L = texture2D(uTexture, vL).rgb;
        vec3 R = texture2D(uTexture, vR).rgb;
        vec3 T = texture2D(uTexture, vT).rgb;
        vec3 B = texture2D(uTexture, vB).rgb;
        vec3 C = texture2D(uTexture, vUv).rgb;

        float dx = length(R) - length(L);
        float dy = length(T) - length(B);

        vec3 n = normalize(vec3(dx, dy, length(texelSize)));
        vec3 l = vec3(0.0, 0.0, 1.0);

        float diffuse = clamp(dot(n, l) + 0.7, 0.7, 1.0);
        C *= diffuse;

        vec3 bloom = texture2D(uBloom, vUv).rgb;
        bloom = pow(bloom.rgb, vec3(1.0 / 2.2));
        C += bloom;

        float a = max(C.r, max(C.g, C.b));
        gl_FragColor = vec4(mapColor(C), a);
    }
`);

const bloomPrefilterShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform vec3 curve;
    uniform float threshold;

    void main () {
        vec3 c = texture2D(uTexture, vUv).rgb;
        float br = max(c.r, max(c.g, c.b));
        float rq = clamp(br - curve.x, 0.0, curve.y);
        rq = curve.z * rq * rq;
        c *= max(rq, br - threshold) / max(br, 0.0001);
        gl_FragColor = vec4(c, 0.0);
    }
`);

const bloomBlurShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uTexture;

    void main () {
        vec4 sum = vec4(0.0);
        sum += texture2D(uTexture, vL);
        sum += texture2D(uTexture, vR);
        sum += texture2D(uTexture, vT);
        sum += texture2D(uTexture, vB);
        sum *= 0.25;
        gl_FragColor = sum;
    }
`);

const bloomFinalShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uTexture;
    uniform float intensity;

    void main () {
        vec4 sum = vec4(0.0);
        sum += texture2D(uTexture, vL);
        sum += texture2D(uTexture, vR);
        sum += texture2D(uTexture, vT);
        sum += texture2D(uTexture, vB);
        sum *= 0.25;
        gl_FragColor = sum * intensity;
    }
`);

const splatShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;

    void main () {
        vec2 p = vUv - point.xy;
        p.x *= aspectRatio;
        vec3 splat = exp(-dot(p, p) / radius) * color;
        vec3 base = texture2D(uTarget, vUv).xyz;
        gl_FragColor = vec4(base + splat, 1.0);
    }
`);

const advectionManualFilteringShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform vec2 dyeTexelSize;
    uniform float dt;
    uniform float dissipation;

    vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
        vec2 st = uv / tsize - 0.5;

        vec2 iuv = floor(st);
        vec2 fuv = fract(st);

        vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
        vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
        vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
        vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);

        return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
    }

    void main () {
        vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
        gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
        gl_FragColor.a = 1.0;
    }
`);

const advectionShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;

    void main () {
        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
        gl_FragColor = dissipation * texture2D(uSource, coord);
        gl_FragColor.a = 1.0;
    }
`);

const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uVelocity;

    void main () {
        float L = texture2D(uVelocity, vL).x;
        float R = texture2D(uVelocity, vR).x;
        float T = texture2D(uVelocity, vT).y;
        float B = texture2D(uVelocity, vB).y;

        vec2 C = texture2D(uVelocity, vUv).xy;
        if (vL.x < 0.0) { L = -C.x; }
        if (vR.x > 1.0) { R = -C.x; }
        if (vT.y > 1.0) { T = -C.y; }
        if (vB.y < 0.0) { B = -C.y; }

        float div = 0.5 * (R - L + T - B);
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
    }
`);

const curlShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uVelocity;

    void main () {
        float L = texture2D(uVelocity, vL).y;
        float R = texture2D(uVelocity, vR).y;
        float T = texture2D(uVelocity, vT).x;
        float B = texture2D(uVelocity, vB).x;
        float vorticity = R - L - T + B;
        gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
    }
`);

const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision highp sampler2D;

    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;

    void main () {
        float L = texture2D(uCurl, vL).x;
        float R = texture2D(uCurl, vR).x;
        float T = texture2D(uCurl, vT).x;
        float B = texture2D(uCurl, vB).x;
        float C = texture2D(uCurl, vUv).x;

        vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
        force /= length(force) + 0.0001;
        force *= curl * C;
        force.y *= -1.0;

        vec2 vel = texture2D(uVelocity, vUv).xy;
        gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
    }
`);

const pressureShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;

    vec2 boundary (vec2 uv) {
        return uv;
        // uncomment if you use wrap or repeat texture mode
        // uv = min(max(uv, 0.0), 1.0);
        // return uv;
    }

    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        float C = texture2D(uPressure, vUv).x;
        float divergence = texture2D(uDivergence, vUv).x;
        float pressure = (L + R + B + T - divergence) * 0.25;
        gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
    }
`);

const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    precision mediump sampler2D;

    varying highp vec2 vUv;
    varying highp vec2 vL;
    varying highp vec2 vR;
    varying highp vec2 vT;
    varying highp vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;

    vec2 boundary (vec2 uv) {
        return uv;
        // uv = min(max(uv, 0.0), 1.0);
        // return uv;
    }

    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        vec2 velocity = texture2D(uVelocity, vUv).xy;
        velocity.xy -= vec2(R - L, T - B);
        gl_FragColor = vec4(velocity, 0.0, 1.0);
    }
`);

const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    return (destination) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
})();

let simWidth;
let simHeight;
let dyeWidth;
let dyeHeight;
let density;
let velocity;
let divergence;
let curl;
let pressure;
let bloom;

let ditheringTexture = createTextureAsync('LDR_RGB1_0.png');

const clearProgram = new GLProgram(baseVertexShader, clearShader);
const colorProgram = new GLProgram(baseVertexShader, colorShader);
const backgroundProgram = new GLProgram(baseVertexShader, backgroundShader);
const displayProgram = new GLProgram(baseVertexShader, displayShader);
const displayBloomProgram = new GLProgram(baseVertexShader, displayBloomShader);
const displayShadingProgram = new GLProgram(baseVertexShader, displayShadingShader);
const displayBloomShadingProgram = new GLProgram(baseVertexShader, displayBloomShadingShader);
const bloomPrefilterProgram = new GLProgram(baseVertexShader, bloomPrefilterShader);
const bloomBlurProgram = new GLProgram(baseVertexShader, bloomBlurShader);
const bloomFinalProgram = new GLProgram(baseVertexShader, bloomFinalShader);
const splatProgram = new GLProgram(baseVertexShader, splatShader);
const advectionProgram = new GLProgram(baseVertexShader, ext.supportLinearFiltering ? advectionShader : advectionManualFilteringShader);
const divergenceProgram = new GLProgram(baseVertexShader, divergenceShader);
const curlProgram = new GLProgram(baseVertexShader, curlShader);
const vorticityProgram = new GLProgram(baseVertexShader, vorticityShader);
const pressureProgram = new GLProgram(baseVertexShader, pressureShader);
const gradientSubtractProgram = new GLProgram(baseVertexShader, gradientSubtractShader);

function initFrameBuffers() {
    let simRes = getResolution(config.SIM_RESOLUTION);
    let dyeRes = getResolution(config.DYE_RESOLUTION);

    simWidth = simRes.width;
    simHeight = simRes.height;
    dyeWidth = dyeRes.width;
    dyeHeight = dyeRes.height;

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const rg = ext.formatRG;
    const r = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    if (density == null)
        density = createDoubleFBO(dyeWidth, dyeHeight, rgba.internalFormat, rgba.format, texType, filtering);
    else
        density = resizeDoubleFBO(density, dyeWidth, dyeHeight, rgba.internalFormat, rgba.format, texType, filtering);

    if (velocity == null)
        velocity = createDoubleFBO(simWidth, simHeight, rg.internalFormat, rg.format, texType, filtering);
    else
        velocity = resizeDoubleFBO(velocity, simWidth, simHeight, rg.internalFormat, rg.format, texType, filtering);

    divergence = createFBO(simWidth, simHeight, r.internalFormat, r.format, texType, gl.NEAREST);
    curl = createFBO(simWidth, simHeight, r.internalFormat, r.format, texType, gl.NEAREST);
    pressure = createDoubleFBO(simWidth, simHeight, r.internalFormat, r.format, texType, gl.NEAREST);

    initBloomFrameBuffers();
}

function initBloomFrameBuffers() {
    let res = getResolution(config.BLOOM_RESOLUTION);

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    bloom = createFBO(res.width, res.height, rgba.internalFormat, rgba.format, texType, filtering);

    bloomFrameBuffers.length = 0;
    for (let i = 0; i < config.BLOOM_ITERATIONS; i++) {
        let width = res.width >> (i + 1);
        let height = res.height >> (i + 1);

        if (width < 2 || height < 2) break;

        let fbo = createFBO(width, height, rgba.internalFormat, rgba.format, texType, filtering);
        bloomFrameBuffers.push(fbo);
    }
}

function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
        texture,
        fbo,
        width: w,
        height: h,
        attach(id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };
}

function createDoubleFBO(w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);

    return {
        get read() {
            return fbo1;
        },
        set read(value) {
            fbo1 = value;
        },
        get write() {
            return fbo2;
        },
        set write(value) {
            fbo2 = value;
        },
        swap() {
            let temp = fbo1;
            fbo1 = fbo2;
            fbo2 = temp;
        }
    }
}

function resizeFBO(target, w, h, internalFormat, format, type, param) {
    let newFBO = createFBO(w, h, internalFormat, format, type, param);
    clearProgram.bind();
    clearProgram.uniform("1i", "uTexture", target.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, 1);
    blit(newFBO.fbo);
    return newFBO;
}

function resizeDoubleFBO(target, w, h, internalFormat, format, type, param) {
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    return target;
}

function createTextureAsync(url) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));

    let obj = {
        texture,
        width: 1,
        height: 1,
        attach(id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };

    let image = new Image();
    image.onload = () => {
        obj.width = image.width;
        obj.height = image.height;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    };
    image.src = url;

    return obj;
}

initFrameBuffers();
// dont think we need this
// multipleSplats(parseInt(Math.random() * 20) + 3);

let lastColorChangeTime = Date.now();

update();

function update() {
    resizeCanvas();
    input();
    if (!config.PAUSED)
        step(0.016);
    render(null);
    requestAnimationFrame(update);
}

function input() {
    if (splatStack.length > 0)
        multipleSplats(splatStack.pop());

    for (let i = 0; i < pointers.length; i++) {
        const p = pointers[i];
        if (p.moved) {
            splat(p.x, p.y, p.dx, p.dy, p.color, config.SPLAT_RADIUS);
            p.moved = false;
        }
    }

    if (lastColorChangeTime + 100 < Date.now()) {
        lastColorChangeTime = Date.now();
        for (let i = 0; i < pointers.length; i++) {
            const p = pointers[i];
            p.color = generateColor(.15);
        }
    }
}

function
step(dt) {
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, simWidth, simHeight);

    curlProgram.bind();
    curlProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    curlProgram.uniform("1i", "uVelocity", velocity.read.attach(0));
    blit(curl.fbo);

    vorticityProgram.bind();
    vorticityProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    vorticityProgram.uniform("1i", "uVelocity", velocity.read.attach(0));
    vorticityProgram.uniform("1i", "uCurl", curl.attach(1));
    vorticityProgram.uniform("1f", "curl", config.CURL);
    vorticityProgram.uniform("1f", "dt", dt);
    blit(velocity.write.fbo);
    velocity.swap();

    divergenceProgram.bind();
    divergenceProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    divergenceProgram.uniform("1i", "uVelocity", velocity.read.attach(0));
    blit(divergence.fbo);

    clearProgram.bind();
    clearProgram.uniform("1i", "uTexture", pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE_DISSIPATION);
    blit(pressure.write.fbo);
    pressure.swap();

    pressureProgram.bind();
    pressureProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    pressureProgram.uniform("1i", "uDivergence", divergence.attach(0));
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
        pressureProgram.uniform("1i", "uPressure", pressure.read.attach(1));
        blit(pressure.write.fbo);
        pressure.swap();
    }

    gradientSubtractProgram.bind();
    gradientSubtractProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    gradientSubtractProgram.uniform("1i", "uPressure", pressure.read.attach(0));
    gradientSubtractProgram.uniform("1i", "uVelocity", velocity.read.attach(1));
    blit(velocity.write.fbo);
    velocity.swap();

    advectionProgram.bind();
    advectionProgram.uniform("2f", "texelSize", 1.0 / simWidth, 1.0 / simHeight);
    if (!ext.supportLinearFiltering)
        advectionProgram.uniform("2f", "dyeTexelSize", 1.0 / simWidth, 1.0 / simHeight);
    let velocityId = velocity.read.attach(0);
    advectionProgram.uniform("1i", "uVelocity", velocityId);
    advectionProgram.uniform("1i", "uSource", velocityId);
    advectionProgram.uniform("1f", "dt", dt);
    advectionProgram.uniform("1f", "dissipation", config.VELOCITY_DISSIPATION);
    blit(velocity.write.fbo);
    velocity.swap();

    gl.viewport(0, 0, dyeWidth, dyeHeight);

    if (!ext.supportLinearFiltering)
        advectionProgram.uniform("2f", "dyeTexelSize", 1.0 / dyeWidth, 1.0 / dyeHeight);
    advectionProgram.uniform("1i", "uVelocity", velocity.read.attach(0));
    advectionProgram.uniform("1i", "uSource", density.read.attach(1));
    advectionProgram.uniform("1f", "dissipation", config.DENSITY_DISSIPATION);
    blit(density.write.fbo);
    density.swap();
}

function render(target) {
    if (config.BLOOM)
        applyBloom(density.read, bloom);

    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.BLEND);

    let width = target == null ? gl.drawingBufferWidth : dyeWidth;
    let height = target == null ? gl.drawingBufferHeight : dyeHeight;

    gl.viewport(0, 0, width, height);

    colorProgram.bind();
    let bc = config.BACK_COLOR;
    gl.uniform4f(colorProgram.uniforms.color, bc.r / 255, bc.g / 255, bc.b / 255, 1);
    blit(target);

    let program;
    if (config.SHADING) {
        program = config.BLOOM ? displayBloomShadingProgram : displayShadingProgram;
    } else {
        program = config.BLOOM ? displayBloomProgram : displayProgram;
    }
    program.bind();
    program.uniform("2f", "texelSize", 1.0 / width, 1.0 / height);
    program.uniform("1i", "uTexture", density.read.attach(0));
    if (config.BLOOM) {
        program.uniform("1i", "uBloom", bloom.attach(1));
        program.uniform("1i", "uDithering", ditheringTexture.attach(2));
        let scale = getTextureScale(ditheringTexture, width, height);
        program.uniform("2f", "ditherScale", scale.x, scale.y);

        // audio uniforms
        program.uniform("1f", "uAudioHigh", audioHigh);
        program.uniform("1f", "uColorAmp", config.AUDIO_COLOR_FILTER_AMP);
        program.uniform("1f", "uColorExp", config.AUDIO_COLOR_FILTER_EXP);
    }

    blit(target);
}

// expects bloomBlurProgram to be bound
function blurTo(last, dest) {
    bloomBlurProgram.uniform("2f", "texelSize", 1.0 / last.width, 1.0 / last.height);
    bloomBlurProgram.uniform("1i", "uTexture", last.attach(0));
    gl.viewport(0, 0, dest.width, dest.height);
    blit(dest.fbo);
}

function applyBloom(source, destination) {
    if (bloomFrameBuffers.length < 2)
        return;

    let last = destination;

    gl.disable(gl.BLEND);
    bloomPrefilterProgram.bind();
    let knee = config.BLOOM_THRESHOLD * config.BLOOM_SOFT_KNEE + 0.0001;
    let curve0 = config.BLOOM_THRESHOLD - knee;
    let curve1 = knee * 2;
    let curve2 = 0.25 / knee;
    gl.uniform3f(bloomPrefilterProgram.uniforms.curve, curve0, curve1, curve2);
    gl.uniform1f(bloomPrefilterProgram.uniforms.threshold, config.BLOOM_THRESHOLD);
    bloomPrefilterProgram.uniform("1i", "uTexture", source.attach(0));
    gl.viewport(0, 0, last.width, last.height);
    blit(last.fbo);

    bloomBlurProgram.bind();
    for (let i = 0; i < bloomFrameBuffers.length; i++) {
        let dest = bloomFrameBuffers[i];
        blurTo(last, dest);
        last = dest;
    }

    gl.blendFunc(gl.ONE, gl.ONE);
    gl.enable(gl.BLEND);

    for (let i = bloomFrameBuffers.length - 2; i >= 0; i--) {
        let baseTex = bloomFrameBuffers[i];
        blurTo(last, baseTex);
        last = baseTex;
    }

    gl.disable(gl.BLEND);
    bloomFinalProgram.bind();
    bloomFinalProgram.uniform("2f", "texelSize", 1.0 / last.width, 1.0 / last.height);
    bloomFinalProgram.uniform("1i", "uTexture", last.attach(0));
    bloomFinalProgram.uniform("1f", "intensity", config.BLOOM_INTENSITY);
    gl.viewport(0, 0, destination.width, destination.height);
    blit(destination.fbo);
}

function splat(x, y, dx, dy, color, radius) {
    gl.viewport(0, 0, simWidth, simHeight);
    splatProgram.bind();
    splatProgram.uniform("1i", "uTarget", velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    splatProgram.uniform("2f", "point", x / canvas.width, 1.0 - y / canvas.height);
    gl.uniform3f(splatProgram.uniforms.color, dx, -dy, 1.0);
    splatProgram.uniform("1f", "radius", radius / 100.0);
    blit(velocity.write.fbo);
    velocity.swap();

    gl.viewport(0, 0, dyeWidth, dyeHeight);
    splatProgram.uniform("1i", "uTarget", density.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, color.r, color.g, color.b);
    blit(density.write.fbo);
    density.swap();
}

function multipleSplats(amount) {
    for (let i = 0; i < amount; i++) {
        const color = generateColor();
        const x = canvas.width * Math.random();
        const y = canvas.height * Math.random();
        const dx = 1000 * (Math.random() - 0.5);
        const dy = 1000 * (Math.random() - 0.5);
        splat(x, y, dx, dy, color, config.SPLAT_RADIUS);
    }
}

function resizeCanvas() {
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        initFrameBuffers();
    }
}

canvas.addEventListener('mousemove', e => {
    if (!config.SHOW_MOUSE_MOVEMENT) return;
    pointers[0].moved = true;
    pointers[0].dx = (e.offsetX - pointers[0].x) * 5.0;
    pointers[0].dy = (e.offsetY - pointers[0].y) * 5.0;
    pointers[0].x = e.offsetX;
    pointers[0].y = e.offsetY;
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        let pointer = pointers[i];
        pointer.moved = pointer.down;
        pointer.dx = (touches[i].pageX - pointer.x) * 8.0;
        pointer.dy = (touches[i].pageY - pointer.y) * 8.0;
        pointer.x = touches[i].pageX;
        pointer.y = touches[i].pageY;
    }
}, false);

canvas.addEventListener('mouseenter', () => {
    pointers[0].down = true;
    pointers[0].color = config.POINTER_COLOR.getRandom();
});

canvas.addEventListener('touchstart', e => {
    if (!config.SPLAT_ON_CLICK) return;
    e.preventDefault();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
        if (i >= pointers.length)
            pointers.push(new pointerPrototype());

        pointers[i].id = touches[i].identifier;
        pointers[i].down = true;
        pointers[i].x = touches[i].pageX;
        pointers[i].y = touches[i].pageY;
        pointers[i].color = config.POINTER_COLOR.getRandom();
    }
});

canvas.addEventListener("mousedown", () => {
    if (!config.SPLAT_ON_CLICK) return;
    multipleSplats(Math.random() * 20 + 5);
});

window.addEventListener('mouseleave', () => {
    pointers[0].down = false;
});

window.addEventListener('touchend', e => {
    const touches = e.changedTouches;
    for (let i = 0; i < touches.length; i++)
        for (let j = 0; j < pointers.length; j++)
            if (touches[i].identifier === pointers[j].id)
                pointers[j].down = false;
});

window.addEventListener('keydown', e => {
    if (e.code === 'KeyP')
        config.PAUSED = !config.PAUSED;
    if (e.key === ' ')
        splatStack.push(Math.random() * 20 + 5);
});

function generateColor(brightness = 1) {
    return HSVtoRGB(Math.random(), 1.0, brightness);
}

function HSVtoRGB(h, s, v) {
    let r, g, b, i, f, p, q, t;
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
    }

    return {
        r,
        g,
        b
    };
}

function getResolution(resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1)
        aspectRatio = 1.0 / aspectRatio;

    let max = Math.round(resolution * aspectRatio);
    let min = Math.round(resolution);

    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
        return {width: max, height: min};
    else
        return {width: min, height: max};
}

function getTextureScale(texture, width, height) {
    return {
        x: width / texture.width,
        y: height / texture.height
    };
}

function rgbToPointerColor(color) {
    let c = color.split(" ");
    // let hue = RGBToHue(c[0], c[1], c[2]);
    // let c2 = HSVtoRGB(hue/360, 1.0, 1.0);
    // c2.r *= 0.15;
    // c2.g *= 0.15;
    // c2.b *= 0.15;
    // return c2;
    return {
        r: c[0] * 0.15,
        g: c[1] * 0.15,
        b: c[2] * 0.15
    }
}
