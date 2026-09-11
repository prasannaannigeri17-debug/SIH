let socket;
let scene, camera, renderer, controls;
let pointCloud, gridGroup, bboxGroup, egoGroup;
let configData = null;
let semanticClasses = {};

// UI State
let isPlaying = true;
let currentFrameIdx = 0;
let totalFrames = 0;
let colorMode = "semantics"; // semantics, elevation, drivability, zones
let latencyChart;

// Zone Definitions matching backend defaults
const ZONES = [
  { name: "A", range: [0, 10],  color: new THREE.Color("#00E5FF") },
  { name: "B", range: [10, 30], color: new THREE.Color("#00E676") },
  { name: "C", range: [30, 60], color: new THREE.Color("#FFEA00") },
  { name: "D", range: [60, 100], color: new THREE.Color("#FF3D00") }
];

async function fetchConfig() {
  const res = await fetch("/api/config");
  const data = await res.json();
  configData = data.config;
  semanticClasses = data.classes;
  buildLegend();
}

function buildLegend() {
  const container = document.getElementById("legend-grid");
  container.innerHTML = "";
  for (const [id, info] of Object.entries(semanticClasses)) {
    const div = document.createElement("div");
    div.className = "legend-item";
    const box = document.createElement("div");
    box.className = "legend-color-box";
    const r = info.color[0], g = info.color[1], b = info.color[2];
    box.style.background = `rgb(${r},${g},${b})`;
    div.appendChild(box);
    div.appendChild(document.createTextNode(info.name));
    container.appendChild(div);
  }
}

function initThreeJS() {
  const canvasContainer = document.getElementById("threejs-canvas");
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#030406");

  camera = new THREE.PerspectiveCamera(60, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 500);
  camera.position.set(-20, -20, 25);
  camera.up.set(0, 0, 1);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  canvasContainer.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(0, 0, 50);
  scene.add(dirLight);

  // Groups
  pointCloud = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 0.1, vertexColors: true, sizeAttenuation: true })
  );
  scene.add(pointCloud);

  gridGroup = new THREE.Group();
  scene.add(gridGroup);

  bboxGroup = new THREE.Group();
  scene.add(bboxGroup);

  egoGroup = new THREE.Group();
  drawEgoVehicle();
  drawZoneRings();
  scene.add(egoGroup);

  window.addEventListener("resize", () => {
    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
  });

  animate();
}

function drawEgoVehicle() {
  const geo = new THREE.BoxGeometry(4.5, 2.0, 1.5);
  const mat = new THREE.MeshLambertMaterial({ color: 0x3B82F6, transparent: true, opacity: 0.8 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, 0, 0.75);
  egoGroup.add(mesh);
  
  // Arrow indicating forward direction (X axis)
  const dir = new THREE.Vector3(1, 0, 0);
  const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,1.5), 4, 0x00E5FF);
  egoGroup.add(arrow);
}

function drawZoneRings() {
  const ringGroup = new THREE.Group();
  ringGroup.name = "zoneRings";
  
  ZONES.forEach(zone => {
    if(zone.range[1] >= 100) return; // don't draw 100m ring, too big
    const geo = new THREE.RingGeometry(zone.range[1] - 0.2, zone.range[1], 64);
    const mat = new THREE.MeshBasicMaterial({ color: zone.color, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.z = -1.0;
    ringGroup.add(mesh);
  });
  egoGroup.add(ringGroup);
}

function setViewMode(mode) {
  document.querySelectorAll(".view-btn").forEach(btn => btn.classList.remove("active"));
  
  if(mode === "top") {
    document.getElementById("btn-view-top").classList.add("active");
    camera.position.set(0, 0, 100);
    controls.target.set(0, 0, 0);
  } else if(mode === "perspective") {
    document.getElementById("btn-view-perspective").classList.add("active");
    camera.position.set(-20, -20, 25);
    controls.target.set(0, 0, 0);
  } else if(mode === "ego") {
    document.getElementById("btn-view-ego").classList.add("active");
    camera.position.set(-15, 0, 8);
    controls.target.set(10, 0, 0);
  }
}

function initChart() {
  const ctx = document.getElementById('latencyChart').getContext('2d');
  latencyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Load', 'Prep', 'Infer', 'Adapt', 'Base'],
      datasets: [{
        label: 'Latency (ms)',
        data: [0, 0, 0, 0, 0],
        backgroundColor: [
          '#64748b', '#3b82f6', '#8b5cf6', '#10b981', '#ef4444'
        ],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: '#2D3340' }, ticks: { color: '#94A3B8', font: {size: 10} } },
        x: { grid: { display: false }, ticks: { color: '#94A3B8', font: {size: 10} } }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10, 13, 18, 0.9)',
          titleColor: '#fff',
          bodyColor: '#fff',
          borderColor: '#2D3340',
          borderWidth: 1
        }
      }
    }
  });
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  
  socket.onopen = () => {
    console.log("WebSocket connected");
    // Ensure server syncs to our default play state
    socket.send(JSON.stringify({ command: isPlaying ? "play" : "pause" }));
  };

  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    updateDashboard(payload);
    render3D(payload);
  };

  socket.onclose = () => {
    console.log("WebSocket closed. Reconnecting in 2s...");
    setTimeout(connectWebSocket, 2000);
  };
}

function updateDashboard(data) {
  const t = data.telemetry;
  
  // Header
  document.getElementById("hud-fps").textContent = t.fps.toFixed(1);
  document.getElementById("hud-latency").textContent = t.total_latency_ms.toFixed(1) + " ms";
  document.getElementById("hud-cell-red").textContent = t.cell_reduction_pct.toFixed(1) + " %";
  
  // Playback Control
  currentFrameIdx = data.frame_id;
  totalFrames = data.total_frames;
  const slider = document.getElementById("frame-slider");
  slider.max = totalFrames - 1;
  slider.value = currentFrameIdx;
  document.getElementById("frame-indicator").textContent = `Frame: ${currentFrameIdx} / ${totalFrames}`;

  // Pill
  document.getElementById("pill-pts").textContent = `${t.point_count.toLocaleString()} Points`;
  document.getElementById("pill-cells").textContent = `${t.adaptive_cells.toLocaleString()} Cells`;
  document.getElementById("pill-time").textContent = `t = ${data.timestamp.toFixed(1)}s`;

  // Live Benchmarks Panel
  document.getElementById("tbl-cell-adapt").textContent = t.adaptive_cells.toLocaleString();
  document.getElementById("tbl-cell-base").textContent = t.baseline_cells.toLocaleString();
  document.getElementById("tbl-cell-red").textContent = `(-${t.cell_reduction_pct.toFixed(1)}%)`;

  document.getElementById("tbl-mem-adapt").textContent = t.adaptive_mem_kb.toFixed(1);
  document.getElementById("tbl-mem-base").textContent = t.baseline_mem_kb.toFixed(1);
  document.getElementById("tbl-mem-red").textContent = `(-${t.mem_reduction_pct.toFixed(1)}%)`;

  const b = t.benchmark;
  document.getElementById("tbl-lat-adapt").textContent = b.adaptive_latency_ms.toFixed(1);
  document.getElementById("tbl-lat-base").textContent = b.baseline_latency_ms.toFixed(1);
  document.getElementById("tbl-lat-speedup").textContent = `(${b.speedup.toFixed(2)}x)`;

  // Resource Bars
  const memAdaptPct = Math.min(100, (t.adaptive_mem_kb / Math.max(1, t.baseline_mem_kb)) * 100);
  document.getElementById("bar-mem-adapt").style.width = memAdaptPct + "%";
  document.getElementById("val-mem-adapt").textContent = t.adaptive_mem_kb.toFixed(1) + " KB";
  document.getElementById("val-mem-base").textContent = t.baseline_mem_kb.toFixed(1) + " KB";

  const cellAdaptPct = Math.min(100, (t.adaptive_cells / Math.max(1, t.baseline_cells)) * 100);
  document.getElementById("bar-cell-adapt").style.width = cellAdaptPct + "%";
  document.getElementById("val-cell-adapt").textContent = t.adaptive_cells.toLocaleString() + " cells";
  document.getElementById("val-cell-base").textContent = t.baseline_cells.toLocaleString() + " cells";

  // Zone Counts
  if (t.zone_distribution) {
    document.getElementById("count-zone-a").textContent = (t.zone_distribution[0] || 0).toLocaleString();
    document.getElementById("count-zone-b").textContent = (t.zone_distribution[1] || 0).toLocaleString();
    document.getElementById("count-zone-c").textContent = (t.zone_distribution[2] || 0).toLocaleString();
    document.getElementById("count-zone-d").textContent = (t.zone_distribution[3] || 0).toLocaleString();
  }

  // Latency Chart
  const sl = t.stage_latencies;
  latencyChart.data.datasets[0].data = [
    sl.load || 0,
    sl.preprocess || 0,
    sl.perception || 0,
    sl.adaptive_grid || 0,
    sl.baseline_grid || 0
  ];
  latencyChart.update();
}

// Simple color map for elevation
function getElevationColor(z) {
  const minZ = -2.0, maxZ = 3.0;
  const t = Math.max(0, Math.min(1, (z - minZ) / (maxZ - minZ)));
  return new THREE.Color().setHSL(0.7 - t * 0.7, 1.0, 0.5);
}

function getZoneColor(cx, cy) {
  const r = Math.sqrt(cx*cx + cy*cy);
  for (const z of ZONES) {
    if (r >= z.range[0] && r < z.range[1]) return z.color;
  }
  return new THREE.Color(0x333333);
}

function render3D(data) {
  // Update Ego
  if (data.ego_pose) {
    const p = data.ego_pose;
    egoGroup.position.set(p[0][3], p[1][3], p[2][3]);
    const r = new THREE.Matrix4().set(
      p[0][0], p[0][1], p[0][2], 0,
      p[1][0], p[1][1], p[1][2], 0,
      p[2][0], p[2][1], p[2][2], 0,
      0,       0,       0,       1
    );
    egoGroup.quaternion.setFromRotationMatrix(r);
  }

  // Points
  if (data.points && document.getElementById("toggle-points").checked) {
    const pts = data.points;
    const count = pts.xyz.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const x = pts.xyz[i][0];
      const y = pts.xyz[i][1];
      const z = pts.xyz[i][2];
      positions[i*3] = x;
      positions[i*3+1] = y;
      positions[i*3+2] = z;

      let color;
      if (colorMode === "elevation") {
        color = getElevationColor(z);
      } else if (colorMode === "zones") {
        color = getZoneColor(x, y);
      } else {
        // semantics
        const sem = pts.sem[i];
        const info = semanticClasses[sem];
        color = info ? new THREE.Color(`rgb(${info.color[0]},${info.color[1]},${info.color[2]})`) : new THREE.Color(0x888888);
      }
      colors[i*3] = color.r;
      colors[i*3+1] = color.g;
      colors[i*3+2] = color.b;
    }

    pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    pointCloud.geometry.attributes.position.needsUpdate = true;
    pointCloud.geometry.attributes.color.needsUpdate = true;
    pointCloud.visible = true;
  } else {
    pointCloud.visible = false;
  }

  // Cells (InstancedMesh)
  gridGroup.clear();
  if (data.cells && document.getElementById("toggle-grid").checked) {
    const cells = data.cells;
    const resGroups = {};

    cells.forEach(c => {
      const res = c.res;
      if (!resGroups[res]) resGroups[res] = [];
      resGroups[res].push(c);
    });

    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxGeoLine = new THREE.EdgesGeometry(boxGeo);

    for (const [resStr, cls] of Object.entries(resGroups)) {
      const res = parseFloat(resStr);
      const mat = new THREE.MeshLambertMaterial({ vertexColors: true, transparent: true, opacity: 0.8 });
      const iMesh = new THREE.InstancedMesh(boxGeo, mat, cls.length);
      
      const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.1 });
      const iLines = new THREE.LineSegments(boxGeoLine, lineMat);
      
      const dummy = new THREE.Object3D();
      
      cls.forEach((c, i) => {
        const h = Math.max(0.05, c.z_max - c.z_min);
        const zCenter = c.z_min + h/2;
        
        dummy.position.set(c.cx, c.cy, zCenter);
        dummy.scale.set(res*0.95, res*0.95, h);
        dummy.updateMatrix();
        iMesh.setMatrixAt(i, dummy.matrix);

        let color;
        if (colorMode === "elevation") {
          color = getElevationColor(c.z_max);
        } else if (colorMode === "zones") {
          color = getZoneColor(c.cx, c.cy);
        } else if (colorMode === "drivability") {
          color = c.is_drivable ? new THREE.Color(0x10B981) : new THREE.Color(0xEF4444);
        } else {
          // Semantics or Dynamic
          if (c.is_dynamic && Object.keys(semanticClasses).length > 0) {
            // Flash dynamic objects orange/red
            color = new THREE.Color(0xFF3D00);
          } else {
            const info = semanticClasses[c.sem];
            color = info ? new THREE.Color(`rgb(${info.color[0]},${info.color[1]},${info.color[2]})`) : new THREE.Color(0x888888);
          }
        }
        iMesh.setColorAt(i, color);
      });
      
      iMesh.instanceMatrix.needsUpdate = true;
      if(iMesh.instanceColor) iMesh.instanceColor.needsUpdate = true;
      gridGroup.add(iMesh);
    }
  }

  // Bounding Boxes
  bboxGroup.clear();
  if (data.bounding_boxes && document.getElementById("toggle-bboxes").checked) {
    data.bounding_boxes.forEach(bb => {
      const w = bb.max[0] - bb.min[0];
      const h = bb.max[1] - bb.min[1];
      const d = bb.max[2] - bb.min[2];
      const cx = bb.min[0] + w/2;
      const cy = bb.min[1] + h/2;
      const cz = bb.min[2] + d/2;

      const geo = new THREE.BoxGeometry(w, h, d);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({ color: 0xFF3D00, linewidth: 2 });
      const mesh = new THREE.LineSegments(edges, mat);
      mesh.position.set(cx, cy, cz);
      bboxGroup.add(mesh);
    });
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function updatePlayPauseBtn() {
  const btn = document.getElementById("btn-play-pause");
  if (isPlaying) {
    btn.innerHTML = `<span class="material-symbols-outlined">pause</span> Pause`;
    btn.classList.add("primary");
  } else {
    btn.innerHTML = `<span class="material-symbols-outlined">play_arrow</span> Play`;
    btn.classList.remove("primary");
  }
}

function setupControls() {
  const btnPlay = document.getElementById("btn-play-pause");
  btnPlay.addEventListener("click", () => {
    isPlaying = !isPlaying;
    updatePlayPauseBtn();
    socket.send(JSON.stringify({ command: isPlaying ? "play" : "pause" }));
  });

  document.getElementById("btn-step").addEventListener("click", () => {
    isPlaying = false;
    updatePlayPauseBtn();
    socket.send(JSON.stringify({ command: "step" }));
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    socket.send(JSON.stringify({ command: "reset" }));
  });

  const slider = document.getElementById("frame-slider");
  let seekTimeout = null;
  slider.addEventListener("input", (e) => {
    isPlaying = false;
    updatePlayPauseBtn();
    document.getElementById("frame-indicator").textContent = `Frame: ${e.target.value} / ${totalFrames}`;
    
    // Throttle seek sending
    if (seekTimeout) clearTimeout(seekTimeout);
    seekTimeout = setTimeout(() => {
      socket.send(JSON.stringify({ command: "seek", frame_index: parseInt(e.target.value) }));
    }, 50);
  });

  document.getElementById("select-aggregation").addEventListener("change", (e) => {
    socket.send(JSON.stringify({ command: "set_aggregation", mode: e.target.value }));
  });

  document.getElementById("select-color-mode").addEventListener("change", (e) => {
    colorMode = e.target.value;
    // Force re-render of current frame by stepping in place (seek to current)
    socket.send(JSON.stringify({ command: "seek", frame_index: currentFrameIdx }));
  });

  // Toggles
  ["grid", "points", "zones", "bboxes", "ego"].forEach(id => {
    document.getElementById(`toggle-${id}`).addEventListener("change", (e) => {
      if (id === "zones") {
        const r = egoGroup.getObjectByName("zoneRings");
        if(r) r.visible = e.target.checked;
      }
      if (id === "ego") egoGroup.visible = e.target.checked;
      
      // Request re-render
      if (!isPlaying) socket.send(JSON.stringify({ command: "seek", frame_index: currentFrameIdx }));
    });
  });

  // View modes
  document.getElementById("btn-view-top").addEventListener("click", () => setViewMode("top"));
  document.getElementById("btn-view-perspective").addEventListener("click", () => setViewMode("perspective"));
  document.getElementById("btn-view-ego").addEventListener("click", () => setViewMode("ego"));
}

window.onload = async () => {
  await fetchConfig();
  initThreeJS();
  initChart();
  setupControls();
  connectWebSocket();
};
