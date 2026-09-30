/**
 * Neural Watershed
 */

(function () {
  // Theme Palettes
  const PALETTES = {
    dark: {
      edgeMinor: 'rgba(0, 191, 165, 0.25)',
      edgeMain: 'rgba(0, 229, 255, 0.6)',
      nodeIdle: 'rgba(0, 191, 165, 0.6)',
      nodeHover: '#ffb74d',
      nodeHoverGlow: 'rgba(255, 183, 77, 0.85)',
      particleBase: 'rgba(0, 229, 255, 0.85)',
      particleRain: 'rgba(120, 255, 245, 0.95)',
      particleGlow: 'rgba(0, 229, 255, 0.6)'
    },
    light: {
      edgeMinor: 'rgba(2, 132, 199, 0.35)',
      edgeMain: 'rgba(3, 105, 161, 0.75)',
      nodeIdle: 'rgba(2, 132, 199, 0.6)',
      nodeHover: '#d97706',
      nodeHoverGlow: 'rgba(217, 119, 6, 0.85)',
      particleBase: 'rgba(2, 132, 199, 0.9)',
      particleRain: 'rgba(30, 64, 175, 0.95)',
      particleGlow: 'rgba(2, 132, 199, 0.5)'
    }
  };

  const rand = (lo, hi) => Math.random() * (hi - lo) + lo;
  const randInt = (lo, hi) => Math.floor(rand(lo, hi));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function isLightMode() {
    return document.body.getAttribute('data-md-color-scheme') === 'default';
  }

  // 2D Segment Intersection Check (Prevents crossing flowlines)
  function ccw(A, B, C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
  }
  function intersect(A, B, C, D) {
    if ((A.x === C.x && A.y === C.y) || (A.x === D.x && A.y === D.y) ||
        (B.x === C.x && B.y === C.y) || (B.x === D.x && B.y === D.y)) {
      return false;
    }
    return (ccw(A, C, D) !== ccw(B, C, D)) && (ccw(A, B, C) !== ccw(A, B, D));
  }

  // =========================================================================
  // 1. HERO WATERSHED ENGINE (HOMEPAGE 100% ORIGINAL v6 CODE & PARAMETERS)
  // =========================================================================
  function initHeroWatershed(canvas) {
    let ctx = canvas.getContext('2d');
    if (!ctx) return null;

    let W, H;
    let nodes = [];
    let edges = [];
    let headwaterNodes = [];
    let catchments = [];
    let particles = [];
    let mouse = { x: -1e4, y: -1e4 };
    let lastRainSpawnTime = 0;
    let animId;
    let visible = false;
    let destroyed = false;

    const P = {
      basinsDesktop: 4,
      basinsMobile: 2,
      baseflowCount: 45,
      stormParticles: 28,
      stormIntervalMs: 4500,
      mouseRadius: 110,
      cloudRainRadius: 75,
      mouseAttraction: 0.3,
      mouseSpeedBoost: 0.85
    };

    function edgeCrossesExisting(p1, p2) {
      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const fromNode = nodes[e.fromId];
        const toNode = nodes[e.toId];
        if (intersect(p1, p2, fromNode, toNode)) {
          return true;
        }
      }
      return false;
    }

    function makeNode(x, y, parentId, order) {
      const nd = {
        id: nodes.length, x, y, parentId,
        streamOrder: order,
        children: [],
        baseRadius: 1.2 + order * 0.55,
        glowIntensity: 0,
        pulsePhase: rand(0, Math.PI * 2)
      };
      nodes.push(nd);
      return nd;
    }

    function buildSinuousEdge(from, to, order) {
      const dx = to.x - from.x, dy = to.y - from.y;
      const dist = Math.hypot(dx, dy) || 1;
      const perpX = -dy / dist, perpY = dx / dist;
      
      const meander = rand(-4.5, 4.5);
      const mx = (from.x + to.x) / 2 + perpX * meander;
      const my = (from.y + to.y) / 2 + perpY * meander;

      const steps = Math.max(8, Math.floor(dist / 7));
      const pts = [];
      for (let i = 0; i <= steps; i++) {
        const t = i / steps, u = 1 - t;
        pts.push({
          x: u * u * from.x + 2 * u * t * mx + t * t * to.x,
          y: u * u * from.y + 2 * u * t * my + t * t * to.y
        });
      }
      return { fromId: from.id, toId: to.id, streamOrder: order, points: pts };
    }

    function tooClose(x, y, minDist) {
      for (let i = nodes.length - 1; i >= Math.max(0, nodes.length - 35); i--) {
        if (Math.hypot(nodes[i].x - x, nodes[i].y - y) < minDist) return true;
      }
      return false;
    }

    function growTributary(parent, angle, order, centerX, halfWidth, basinIds, depth) {
      if (order < 1 || depth > 5) return;

      const baseStep = rand(22, 42);
      const len = baseStep * (0.8 + order * 0.18);
      const wander = rand(-0.25, 0.25);
      const finalAngle = angle + wander;

      let nx = parent.x + Math.cos(finalAngle) * len;
      let ny = parent.y + Math.sin(finalAngle) * len;

      const distFromCenter = nx - centerX;
      if (Math.abs(distFromCenter) > halfWidth) {
        nx = centerX + Math.sign(distFromCenter) * (halfWidth + (Math.abs(distFromCenter) - halfWidth) * 0.25);
      }
      ny = clamp(ny, H * 0.08, H * 0.86);

      if (tooClose(nx, ny, 16)) return;

      const candidatePos = { x: nx, y: ny };
      if (edgeCrossesExisting(candidatePos, parent)) return;

      const nd = makeNode(nx, ny, parent.id, order);
      parent.children.push(nd.id);
      edges.push(buildSinuousEdge(nd, parent, order));
      basinIds.push(nd.id);

      const childCount = order >= 3 ? randInt(2, 3) : (order >= 2 ? randInt(1, 2) : (Math.random() < 0.35 ? 1 : 0));

      if (childCount > 0) {
        const spread = Math.PI * 0.42;
        const startAngle = angle - spread / 2;
        const step = childCount > 1 ? spread / (childCount - 1) : 0;

        for (let c = 0; c < childCount; c++) {
          const ca = childCount === 1 ? angle + rand(-0.12, 0.12) : startAngle + c * step + rand(-0.08, 0.08);
          growTributary(nd, ca, order - 1, centerX, halfWidth, basinIds, depth + 1);
        }
      }
    }

    function buildCatchmentNetwork() {
      nodes = [];
      edges = [];
      headwaterNodes = [];
      catchments = [];
      particles = [];

      const isMobile = W < 768;
      const numBasins = isMobile ? P.basinsMobile : P.basinsDesktop;
      const paddingX = 0.08;
      const usableW = W * (1 - 2 * paddingX);
      const basinW = usableW / numBasins;

      for (let b = 0; b < numBasins; b++) {
        const basinCenterX = W * paddingX + basinW * (b + 0.5);
        const halfW = basinW * 0.44;
        const basinIds = [];

        const outletX = basinCenterX + rand(-halfW * 0.2, halfW * 0.2);
        const outletY = H * 0.86;
        const outlet = makeNode(outletX, outletY, null, 4);
        basinIds.push(outlet.id);

        let cur = outlet;
        const spineSteps = randInt(5, 8);
        const spineDy = (H * 0.68) / spineSteps;

        let curAngle = -Math.PI / 2;
        for (let s = 0; s < spineSteps; s++) {
          const spineMeander = Math.sin(s * 1.1 + b * 1.7) * 20 + rand(-8, 8);
          const targetX = basinCenterX + spineMeander;
          const targetY = outletY - (s + 1) * spineDy;

          const nx = clamp(targetX, basinCenterX - halfW, basinCenterX + halfW);
          const ny = clamp(targetY, H * 0.08, H * 0.92);

          const nextNode = makeNode(nx, ny, cur.id, 4);
          cur.children.push(nextNode.id);
          edges.push(buildSinuousEdge(nextNode, cur, 4));
          basinIds.push(nextNode.id);

          curAngle = Math.atan2(ny - cur.y, nx - cur.x);

          if (s > 0 && s < spineSteps) {
            const tribOrder = s >= 2 ? 3 : 2;
            const leftAngle = curAngle - Math.PI * rand(0.24, 0.38);
            growTributary(cur, leftAngle, tribOrder, basinCenterX, halfW, basinIds, 0);

            const rightAngle = curAngle + Math.PI * rand(0.24, 0.38);
            growTributary(cur, rightAngle, tribOrder, basinCenterX, halfW, basinIds, 0);
          }

          cur = nextNode;
        }

        const fanAngles = [curAngle - 0.42, curAngle, curAngle + 0.42];
        fanAngles.forEach(fa => {
          growTributary(cur, fa, 2, basinCenterX, halfW, basinIds, 0);
        });

        catchments.push(basinIds);
      }

      headwaterNodes = nodes.filter(n => n.children.length === 0);

      for (let i = 0; i < P.baseflowCount; i++) {
        if (headwaterNodes.length === 0) break;
        const hw = headwaterNodes[randInt(0, headwaterNodes.length)];
        const p = spawnParticle(hw, 'baseflow');
        p.idx = rand(0, Math.max(1, p.points.length - 2));
        particles.push(p);
      }
    }

    function buildDownstreamPath(startNode) {
      const pts = [];
      let cur = startNode;
      while (cur.parentId !== null) {
        const parent = nodes[cur.parentId];
        const edge = edges.find(e => e.fromId === cur.id && e.toId === parent.id);
        if (edge) pts.push(...edge.points.slice(0, -1));
        cur = parent;
      }
      if (pts.length === 0) pts.push({ x: startNode.x, y: startNode.y });
      return pts;
    }

    function spawnParticle(startNode, type) {
      const pts = buildDownstreamPath(startNode);
      const isSpecial = (type === 'storm' || type === 'rain');
      return {
        points: pts,
        idx: 0,
        baseSpeed: isSpecial ? rand(0.13, 0.24) : rand(0.035, 0.075),
        size: isSpecial ? rand(1.7, 2.7) : rand(0.9, 1.5),
        alpha: isSpecial ? rand(0.85, 0.98) : rand(0.25, 0.45),
        type,
        born: performance.now(),
        offsetX: 0,
        offsetY: 0
      };
    }

    let lastStormTime = 0;
    function maybeSpawnStorm(now) {
      if (now - lastStormTime < P.stormIntervalMs) return;
      lastStormTime = now;

      if (catchments.length === 0) return;
      const basinIdx = randInt(0, catchments.length);
      const basinSet = new Set(catchments[basinIdx]);
      const basinHW = headwaterNodes.filter(n => basinSet.has(n.id));
      if (basinHW.length === 0) return;

      const count = Math.min(P.stormParticles, basinHW.length * 3);
      for (let i = 0; i < count; i++) {
        const hw = basinHW[randInt(0, basinHW.length)];
        const p = spawnParticle(hw, 'storm');
        p.idx = rand(-3, 0);
        particles.push(p);
      }
    }

    function triggerCloudRain(now) {
      if (now - lastRainSpawnTime < 45) return;
      lastRainSpawnTime = now;

      let nearbyNodes = [];
      nodes.forEach(n => {
        const dm = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        if (dm < P.cloudRainRadius) {
          nearbyNodes.push({ node: n, dist: dm });
        }
      });

      if (nearbyNodes.length > 0) {
        nearbyNodes.sort((a, b) => a.dist - b.dist);
        const targets = nearbyNodes.slice(0, 3);
        targets.forEach(item => {
          const rainP = spawnParticle(item.node, 'rain');
          rainP.idx = 0;
          particles.push(rainP);
        });
      }
    }

    function resize() {
      if (destroyed) return;
      const dpr = window.devicePixelRatio || 1;
      const parent = canvas.parentElement;
      if (!parent) return;
      const r = parent.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return;

      W = r.width; H = r.height;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = W + 'px';
      canvas.style.height = H + 'px';
      ctx.scale(dpr, dpr);
      buildCatchmentNetwork();
    }

    function render(time) {
      if (destroyed || !visible) return;
      ctx.clearRect(0, 0, W, H);
      maybeSpawnStorm(time);

      const palette = isLightMode() ? PALETTES.light : PALETTES.dark;

      edges.forEach(e => {
        if (e.points.length < 2) return;
        const lw = 0.5 + e.streamOrder * 0.55;

        ctx.beginPath();
        ctx.moveTo(e.points[0].x, e.points[0].y);
        for (let i = 1; i < e.points.length; i++) ctx.lineTo(e.points[i].x, e.points[i].y);
        ctx.lineWidth = lw;
        ctx.strokeStyle = e.streamOrder >= 4 ? palette.edgeMain : palette.edgeMinor;
        ctx.stroke();
      });

      const deadIdx = [];
      particles.forEach((p, pi) => {
        if (!p.points || p.points.length < 2) { deadIdx.push(pi); return; }

        const safeIdx = clamp(Math.floor(p.idx), 0, p.points.length - 2);
        const nextI = safeIdx + 1;
        const t = clamp(p.idx - safeIdx, 0, 1);
        const basePt = p.points[safeIdx];
        const nextPt = p.points[nextI];
        let px = basePt.x + (nextPt.x - basePt.x) * t;
        let py = basePt.y + (nextPt.y - basePt.y) * t;

        const dm = Math.hypot(mouse.x - px, mouse.y - py);
        const nearCursor = dm < P.mouseRadius && dm > 0;
        let mouseFactor = 0;

        if (nearCursor) {
          mouseFactor = 1 - dm / P.mouseRadius;
          const attractStrength = P.mouseAttraction * mouseFactor * mouseFactor;
          p.offsetX += ((mouse.x - px) / dm) * attractStrength;
          p.offsetY += ((mouse.y - py) / dm) * attractStrength;
        }

        p.offsetX *= 0.92;
        p.offsetY *= 0.92;
        px += p.offsetX;
        py += p.offsetY;

        const progress = clamp(p.idx / p.points.length, 0, 1);
        const orderBoost = 1 + progress * 1.1;
        const speedMult = nearCursor ? (1 + P.mouseSpeedBoost * mouseFactor) : 1;
        p.idx += p.baseSpeed * orderBoost * speedMult;

        if (p.idx >= p.points.length - 1) {
          if (p.type === 'storm' || p.type === 'rain') { deadIdx.push(pi); return; }
          if (headwaterNodes.length > 0) {
            const hw = headwaterNodes[randInt(0, headwaterNodes.length)];
            Object.assign(p, spawnParticle(hw, 'baseflow'));
          }
        }
        if (p.idx < 0) return;

        let alpha = p.alpha;
        const age = time - p.born;
        if (age < 350) alpha *= age / 350;
        if (p.idx > p.points.length - 6) alpha *= (p.points.length - p.idx) / 6;

        const sz = p.size + mouseFactor * 0.7;
        const isSpecial = (p.type === 'storm' || p.type === 'rain');
        const col = isSpecial ? palette.particleRain : palette.particleBase;

        ctx.beginPath();
        ctx.arc(px, py, sz, 0, Math.PI * 2);
        ctx.fillStyle = col;

        if (isSpecial || mouseFactor > 0.1) {
          ctx.shadowBlur = 7;
          ctx.shadowColor = palette.particleGlow;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      for (let i = deadIdx.length - 1; i >= 0; i--) particles.splice(deadIdx[i], 1);

      let closestNode = null;
      let minDistance = P.mouseRadius;

      nodes.forEach(n => {
        const dm = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        if (dm < minDistance) {
          minDistance = dm;
          closestNode = n;
        }
      });

      nodes.forEach(n => {
        const isTheChosenOne = (n === closestNode);
        const targetGlow = isTheChosenOne ? (1 - minDistance / P.mouseRadius) : 0;
        n.glowIntensity += (targetGlow - n.glowIntensity) * 0.15;

        const breath = Math.sin(time * 0.0015 + n.pulsePhase) * 0.2;
        const r = n.baseRadius * (1 + 0.45 * n.glowIntensity) + breath;
        if (r < 0.8) return;

        ctx.beginPath();
        ctx.arc(n.x, n.y, Math.max(0.8, r), 0, Math.PI * 2);

        if (n.glowIntensity > 0.06) {
          ctx.fillStyle = palette.nodeHover;
          ctx.shadowBlur = 14 * n.glowIntensity;
          ctx.shadowColor = palette.nodeHoverGlow;
        } else {
          ctx.fillStyle = palette.nodeIdle;
          ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animId = requestAnimationFrame(render);
    }

    const parentElem = canvas.parentElement;
    let lastMT = 0;
    function onMouseMove(e) {
      const now = performance.now();
      if (now - lastMT > 16) {
        const r = canvas.getBoundingClientRect();
        mouse.x = e.clientX - r.left;
        mouse.y = e.clientY - r.top;
        lastMT = now;
        triggerCloudRain(now);
      }
    }
    function onMouseLeave() {
      mouse.x = -1e4; mouse.y = -1e4;
    }

    parentElem.addEventListener('mousemove', onMouseMove);
    parentElem.addEventListener('mouseleave', onMouseLeave);

    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        visible = e.isIntersecting;
        if (visible) {
          if (animId) cancelAnimationFrame(animId);
          render(performance.now());
        }
      });
    }, { threshold: 0.1 });
    obs.observe(parentElem);

    resize();

    let resizeTimer;
    function onWindowResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resize, 250);
    }
    window.addEventListener('resize', onWindowResize);

    return {
      destroy: function () {
        destroyed = true;
        if (animId) cancelAnimationFrame(animId);
        obs.disconnect();
        window.removeEventListener('resize', onWindowResize);
        parentElem.removeEventListener('mousemove', onMouseMove);
        parentElem.removeEventListener('mouseleave', onMouseLeave);
      }
    };
  }

  // =========================================================================
  // 2. SIDEBAR MINI-WATERSHED ENGINE (SEAMLESS LEFT BACKGROUND GRAPHIC)
  // =========================================================================
  function initSidebarWatershed(canvas) {
    let ctx = canvas.getContext('2d');
    if (!ctx) return null;

    let W, H;
    let nodes = [];
    let edges = [];
    let headwaterNodes = [];
    let particles = [];
    let mouse = { x: -1e4, y: -1e4 };
    let lastRainSpawnTime = 0;
    let animId;
    let visible = false;
    let destroyed = false;

    const P = {
      baseflowCount: 20,
      stormParticles: 12,
      stormIntervalMs: 3800,
      mouseRadius: 90,
      cloudRainRadius: 70,
      mouseAttraction: 0.3,
      mouseSpeedBoost: 0.85
    };

    function edgeCrossesExisting(p1, p2) {
      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const fromNode = nodes[e.fromId];
        const toNode = nodes[e.toId];
        if (intersect(p1, p2, fromNode, toNode)) {
          return true;
        }
      }
      return false;
    }

    function makeNode(x, y, parentId, order) {
      const nd = {
        id: nodes.length, x, y, parentId,
        streamOrder: order,
        children: [],
        baseRadius: 1.0 + order * 0.5,
        glowIntensity: 0,
        pulsePhase: rand(0, Math.PI * 2)
      };
      nodes.push(nd);
      return nd;
    }

    function buildSinuousEdge(from, to, order) {
      const dx = to.x - from.x, dy = to.y - from.y;
      const dist = Math.hypot(dx, dy) || 1;
      const perpX = -dy / dist, perpY = dx / dist;
      
      const meander = rand(-3.5, 3.5);
      const mx = (from.x + to.x) / 2 + perpX * meander;
      const my = (from.y + to.y) / 2 + perpY * meander;

      const steps = Math.max(7, Math.floor(dist / 6));
      const pts = [];
      for (let i = 0; i <= steps; i++) {
        const t = i / steps, u = 1 - t;
        pts.push({
          x: u * u * from.x + 2 * u * t * mx + t * t * to.x,
          y: u * u * from.y + 2 * u * t * my + t * t * to.y
        });
      }
      return { fromId: from.id, toId: to.id, streamOrder: order, points: pts };
    }

    function tooClose(x, y, minDist) {
      for (let i = nodes.length - 1; i >= Math.max(0, nodes.length - 25); i--) {
        if (Math.hypot(nodes[i].x - x, nodes[i].y - y) < minDist) return true;
      }
      return false;
    }

    function growTributary(parent, angle, order, centerX, halfWidth, depth) {
      if (order < 1 || depth > 4) return;

      const baseStep = rand(18, 32);
      const len = baseStep * (0.8 + order * 0.18);
      const wander = rand(-0.25, 0.25);
      const finalAngle = angle + wander;

      let nx = parent.x + Math.cos(finalAngle) * len;
      let ny = parent.y + Math.sin(finalAngle) * len;

      const distFromCenter = nx - centerX;
      if (Math.abs(distFromCenter) > halfWidth) {
        nx = centerX + Math.sign(distFromCenter) * (halfWidth + (Math.abs(distFromCenter) - halfWidth) * 0.25);
      }
      ny = clamp(ny, H * 0.06, H * 0.90);

      if (tooClose(nx, ny, 12)) return;

      const candidatePos = { x: nx, y: ny };
      if (edgeCrossesExisting(candidatePos, parent)) return;

      const nd = makeNode(nx, ny, parent.id, order);
      parent.children.push(nd.id);
      edges.push(buildSinuousEdge(nd, parent, order));

      const childCount = order >= 3 ? randInt(2, 3) : (order >= 2 ? randInt(1, 2) : (Math.random() < 0.35 ? 1 : 0));

      if (childCount > 0) {
        const spread = Math.PI * 0.42;
        const startAngle = angle - spread / 2;
        const step = childCount > 1 ? spread / (childCount - 1) : 0;

        for (let c = 0; c < childCount; c++) {
          const ca = childCount === 1 ? angle + rand(-0.12, 0.12) : startAngle + c * step + rand(-0.08, 0.08);
          growTributary(nd, ca, order - 1, centerX, halfWidth, depth + 1);
        }
      }
    }

    function buildCatchmentNetwork() {
      nodes = [];
      edges = [];
      headwaterNodes = [];
      particles = [];

      const basinCenterX = W * 0.5;
      const halfW = W * 0.42;

      // Outlet near bottom center
      const outletX = basinCenterX + rand(-halfW * 0.15, halfW * 0.15);
      const outletY = H * 0.90;
      const outlet = makeNode(outletX, outletY, null, 4);

      let cur = outlet;
      const spineSteps = randInt(5, 7);
      const spineDy = (H * 0.72) / spineSteps;

      let curAngle = -Math.PI / 2;
      for (let s = 0; s < spineSteps; s++) {
        const spineMeander = Math.sin(s * 1.2) * 16 + rand(-6, 6);
        const targetX = basinCenterX + spineMeander;
        const targetY = outletY - (s + 1) * spineDy;

        const nx = clamp(targetX, basinCenterX - halfW, basinCenterX + halfW);
        const ny = clamp(targetY, H * 0.08, H * 0.92);

        const nextNode = makeNode(nx, ny, cur.id, 4);
        cur.children.push(nextNode.id);
        edges.push(buildSinuousEdge(nextNode, cur, 4));

        curAngle = Math.atan2(ny - cur.y, nx - cur.x);

        if (s > 0 && s < spineSteps) {
          const tribOrder = s >= 2 ? 3 : 2;
          const leftAngle = curAngle - Math.PI * rand(0.24, 0.38);
          growTributary(cur, leftAngle, tribOrder, basinCenterX, halfW, 0);

          const rightAngle = curAngle + Math.PI * rand(0.24, 0.38);
          growTributary(cur, rightAngle, tribOrder, basinCenterX, halfW, 0);
        }

        cur = nextNode;
      }

      const fanAngles = [curAngle - 0.42, curAngle, curAngle + 0.42];
      fanAngles.forEach(fa => {
        growTributary(cur, fa, 2, basinCenterX, halfW, 0);
      });

      headwaterNodes = nodes.filter(n => n.children.length === 0);

      for (let i = 0; i < P.baseflowCount; i++) {
        if (headwaterNodes.length === 0) break;
        const hw = headwaterNodes[randInt(0, headwaterNodes.length)];
        const p = spawnParticle(hw, 'baseflow');
        p.idx = rand(0, Math.max(1, p.points.length - 2));
        particles.push(p);
      }
    }

    function buildDownstreamPath(startNode) {
      const pts = [];
      let cur = startNode;
      while (cur.parentId !== null) {
        const parent = nodes[cur.parentId];
        const edge = edges.find(e => e.fromId === cur.id && e.toId === parent.id);
        if (edge) pts.push(...edge.points.slice(0, -1));
        cur = parent;
      }
      if (pts.length === 0) pts.push({ x: startNode.x, y: startNode.y });
      return pts;
    }

    function spawnParticle(startNode, type) {
      const pts = buildDownstreamPath(startNode);
      const isSpecial = (type === 'storm' || type === 'rain');
      return {
        points: pts,
        idx: 0,
        baseSpeed: isSpecial ? rand(0.12, 0.22) : rand(0.035, 0.075),
        size: isSpecial ? rand(1.5, 2.4) : rand(0.8, 1.4),
        alpha: isSpecial ? rand(0.85, 0.98) : rand(0.28, 0.48),
        type,
        born: performance.now(),
        offsetX: 0,
        offsetY: 0
      };
    }

    let lastStormTime = 0;
    function maybeSpawnStorm(now) {
      if (now - lastStormTime < P.stormIntervalMs) return;
      lastStormTime = now;

      if (headwaterNodes.length === 0) return;
      const count = Math.min(P.stormParticles, headwaterNodes.length * 3);
      for (let i = 0; i < count; i++) {
        const hw = headwaterNodes[randInt(0, headwaterNodes.length)];
        const p = spawnParticle(hw, 'storm');
        p.idx = rand(-3, 0);
        particles.push(p);
      }
    }

    function triggerCloudRain(now) {
      if (now - lastRainSpawnTime < 45) return;
      lastRainSpawnTime = now;

      let nearbyNodes = [];
      nodes.forEach(n => {
        const dm = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        if (dm < P.cloudRainRadius) {
          nearbyNodes.push({ node: n, dist: dm });
        }
      });

      if (nearbyNodes.length > 0) {
        nearbyNodes.sort((a, b) => a.dist - b.dist);
        const targets = nearbyNodes.slice(0, 3);
        targets.forEach(item => {
          const rainP = spawnParticle(item.node, 'rain');
          rainP.idx = 0;
          particles.push(rainP);
        });
      }
    }

    function resize() {
      if (destroyed) return;
      const dpr = window.devicePixelRatio || 1;
      const parent = canvas.parentElement;
      if (!parent) return;
      const r = parent.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return;

      W = r.width; H = r.height;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = W + 'px';
      canvas.style.height = H + 'px';
      ctx.scale(dpr, dpr);
      buildCatchmentNetwork();
    }

    function render(time) {
      if (destroyed || !visible) return;
      ctx.clearRect(0, 0, W, H);
      maybeSpawnStorm(time);

      const palette = isLightMode() ? PALETTES.light : PALETTES.dark;

      edges.forEach(e => {
        if (e.points.length < 2) return;
        const lw = 0.45 + e.streamOrder * 0.5;

        ctx.beginPath();
        ctx.moveTo(e.points[0].x, e.points[0].y);
        for (let i = 1; i < e.points.length; i++) ctx.lineTo(e.points[i].x, e.points[i].y);
        ctx.lineWidth = lw;
        ctx.strokeStyle = e.streamOrder >= 4 ? palette.edgeMain : palette.edgeMinor;
        ctx.stroke();
      });

      const deadIdx = [];
      particles.forEach((p, pi) => {
        if (!p.points || p.points.length < 2) { deadIdx.push(pi); return; }

        const safeIdx = clamp(Math.floor(p.idx), 0, p.points.length - 2);
        const nextI = safeIdx + 1;
        const t = clamp(p.idx - safeIdx, 0, 1);
        const basePt = p.points[safeIdx];
        const nextPt = p.points[nextI];
        let px = basePt.x + (nextPt.x - basePt.x) * t;
        let py = basePt.y + (nextPt.y - basePt.y) * t;

        const dm = Math.hypot(mouse.x - px, mouse.y - py);
        const nearCursor = dm < P.mouseRadius && dm > 0;
        let mouseFactor = 0;

        if (nearCursor) {
          mouseFactor = 1 - dm / P.mouseRadius;
          const attractStrength = P.mouseAttraction * mouseFactor * mouseFactor;
          p.offsetX += ((mouse.x - px) / dm) * attractStrength;
          p.offsetY += ((mouse.y - py) / dm) * attractStrength;
        }

        p.offsetX *= 0.92;
        p.offsetY *= 0.92;
        px += p.offsetX;
        py += p.offsetY;

        const progress = clamp(p.idx / p.points.length, 0, 1);
        const orderBoost = 1 + progress * 1.1;
        const speedMult = nearCursor ? (1 + P.mouseSpeedBoost * mouseFactor) : 1;
        p.idx += p.baseSpeed * orderBoost * speedMult;

        if (p.idx >= p.points.length - 1) {
          if (p.type === 'storm' || p.type === 'rain') { deadIdx.push(pi); return; }
          if (headwaterNodes.length > 0) {
            const hw = headwaterNodes[randInt(0, headwaterNodes.length)];
            Object.assign(p, spawnParticle(hw, 'baseflow'));
          }
        }
        if (p.idx < 0) return;

        let alpha = p.alpha;
        const age = time - p.born;
        if (age < 350) alpha *= age / 350;
        if (p.idx > p.points.length - 6) alpha *= (p.points.length - p.idx) / 6;

        const sz = p.size + mouseFactor * 0.6;
        const isSpecial = (p.type === 'storm' || p.type === 'rain');
        const col = isSpecial ? palette.particleRain : palette.particleBase;

        ctx.beginPath();
        ctx.arc(px, py, sz, 0, Math.PI * 2);
        ctx.fillStyle = col;

        if (isSpecial || mouseFactor > 0.1) {
          ctx.shadowBlur = 6;
          ctx.shadowColor = palette.particleGlow;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      for (let i = deadIdx.length - 1; i >= 0; i--) particles.splice(deadIdx[i], 1);

      let closestNode = null;
      let minDistance = P.mouseRadius;

      nodes.forEach(n => {
        const dm = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        if (dm < minDistance) {
          minDistance = dm;
          closestNode = n;
        }
      });

      nodes.forEach(n => {
        const isTheChosenOne = (n === closestNode);
        const targetGlow = isTheChosenOne ? (1 - minDistance / P.mouseRadius) : 0;
        n.glowIntensity += (targetGlow - n.glowIntensity) * 0.15;

        const breath = Math.sin(time * 0.0015 + n.pulsePhase) * 0.2;
        const r = n.baseRadius * (1 + 0.45 * n.glowIntensity) + breath;
        if (r < 0.6) return;

        ctx.beginPath();
        ctx.arc(n.x, n.y, Math.max(0.6, r), 0, Math.PI * 2);

        if (n.glowIntensity > 0.06) {
          ctx.fillStyle = palette.nodeHover;
          ctx.shadowBlur = 12 * n.glowIntensity;
          ctx.shadowColor = palette.nodeHoverGlow;
        } else {
          ctx.fillStyle = palette.nodeIdle;
          ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animId = requestAnimationFrame(render);
    }

    const parentElem = canvas.parentElement;
    let lastMT = 0;
    function onMouseMove(e) {
      const now = performance.now();
      if (now - lastMT > 16) {
        const r = canvas.getBoundingClientRect();
        mouse.x = e.clientX - r.left;
        mouse.y = e.clientY - r.top;
        lastMT = now;
        triggerCloudRain(now);
      }
    }
    function onMouseLeave() {
      mouse.x = -1e4; mouse.y = -1e4;
    }

    parentElem.addEventListener('mousemove', onMouseMove);
    parentElem.addEventListener('mouseleave', onMouseLeave);

    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        visible = e.isIntersecting;
        if (visible) {
          if (animId) cancelAnimationFrame(animId);
          render(performance.now());
        }
      });
    }, { threshold: 0.05 });
    obs.observe(parentElem);

    resize();

    let resizeTimer;
    function onWindowResize() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resize, 200);
    }
    window.addEventListener('resize', onWindowResize);

    return {
      destroy: function () {
        destroyed = true;
        if (animId) cancelAnimationFrame(animId);
        obs.disconnect();
        window.removeEventListener('resize', onWindowResize);
        parentElem.removeEventListener('mousemove', onMouseMove);
        parentElem.removeEventListener('mouseleave', onMouseLeave);
      }
    };
  }

  // ── Global Initializer ──
  function setupWatersheds() {
    const heroCanvas = document.getElementById('neural-watershed-canvas');
    if (heroCanvas) {
      if (heroCanvas._watershedInstance) {
        heroCanvas._watershedInstance.destroy();
      }
      heroCanvas._watershedInstance = initHeroWatershed(heroCanvas);
    }

    const sidebarCanvas = document.getElementById('sidebar-watershed-canvas');
    if (sidebarCanvas) {
      if (sidebarCanvas._watershedInstance) {
        sidebarCanvas._watershedInstance.destroy();
      }
      sidebarCanvas._watershedInstance = initSidebarWatershed(sidebarCanvas);
    }
  }

  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      setupWatersheds();
    });
  } else {
    document.addEventListener('DOMContentLoaded', setupWatersheds);
  }
})();
