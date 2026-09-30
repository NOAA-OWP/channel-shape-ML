/**
 * Robust Interactive Diagram Pan, Zoom & Fullscreen Controller
 * Works seamlessly with MkDocs-Material and Mermaid.js SVG rendering.
 */

(function () {
  function attachPanZoomToSvg(svg) {
    if (!svg || svg._panZoomAttached) return;
    svg._panZoomAttached = true;

    const parentPre = svg.closest('.mermaid') || svg.parentElement;
    if (!parentPre) return;

    // Avoid double wrapping
    if (parentPre.closest('.diagram-zoom-wrapper')) return;

    // Create main wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'diagram-zoom-wrapper';

    // Create Viewport
    const viewport = document.createElement('div');
    viewport.className = 'diagram-zoom-viewport';

    // Insert wrapper into DOM
    parentPre.parentNode.insertBefore(wrapper, parentPre);
    wrapper.appendChild(viewport);
    viewport.appendChild(parentPre);

    // Create Floating Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'diagram-zoom-toolbar';
    toolbar.innerHTML = `
      <div class="diagram-zoom-hint">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
        <span>Scroll to Zoom • Drag to Pan</span>
      </div>
      <div class="diagram-zoom-buttons">
        <button type="button" class="diagram-btn zoom-in" title="Zoom In (+)" aria-label="Zoom In">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
        </button>
        <button type="button" class="diagram-btn zoom-out" title="Zoom Out (-)" aria-label="Zoom Out">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 13H5v-2h14v2z"/></svg>
        </button>
        <button type="button" class="diagram-btn zoom-reset" title="Reset (100%)" aria-label="Reset View">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
        </button>
        <button type="button" class="diagram-btn zoom-fullscreen" title="Toggle Fullscreen" aria-label="Toggle Fullscreen">
          <svg class="icon-expand" viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/></svg>
          <svg class="icon-compress" viewBox="0 0 24 24" width="15" height="15" fill="currentColor" style="display:none;"><path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-14v3h3v2h-5V5h2z"/></svg>
        </button>
      </div>
    `;
    wrapper.appendChild(toolbar);

    // Initial SVG styles
    svg.style.maxWidth = 'none';
    svg.style.transition = 'transform 0.05s ease-out';
    svg.style.transformOrigin = 'center center';
    svg.style.display = 'block';

    let scale = 1.0;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX = 0, startY = 0;
    let initialTranslateX = 0, initialTranslateY = 0;

    function updateTransform() {
      svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }

    function setZoom(newScale, focalX, focalY) {
      const clampedScale = Math.min(Math.max(newScale, 0.4), 5.0);
      if (focalX !== undefined && focalY !== undefined) {
        const rect = viewport.getBoundingClientRect();
        const mouseRelX = focalX - rect.left - rect.width / 2;
        const mouseRelY = focalY - rect.top - rect.height / 2;
        
        const ratio = clampedScale / scale;
        translateX -= (mouseRelX - translateX) * (ratio - 1);
        translateY -= (mouseRelY - translateY) * (ratio - 1);
      }
      scale = clampedScale;
      updateTransform();
    }

    function resetView() {
      scale = 1.0;
      translateX = 0;
      translateY = 0;
      svg.style.transition = 'transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1)';
      updateTransform();
      setTimeout(() => {
        svg.style.transition = 'transform 0.05s ease-out';
      }, 300);
    }

    // Wheel zoom
    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      setZoom(scale * zoomFactor, e.clientX, e.clientY);
    }, { passive: false });

    // Drag Panning (Mouse & Pointer)
    viewport.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      initialTranslateX = translateX;
      initialTranslateY = translateY;
      viewport.setPointerCapture(e.pointerId);
      viewport.classList.add('is-grabbing');
    });

    viewport.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      translateX = initialTranslateX + (e.clientX - startX);
      translateY = initialTranslateY + (e.clientY - startY);
      updateTransform();
    });

    function endDrag(e) {
      if (isDragging) {
        isDragging = false;
        viewport.classList.remove('is-grabbing');
        try {
          viewport.releasePointerCapture(e.pointerId);
        } catch (_) {}
      }
    }

    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);

    // Toolbar Buttons
    const btnIn = toolbar.querySelector('.zoom-in');
    const btnOut = toolbar.querySelector('.zoom-out');
    const btnReset = toolbar.querySelector('.zoom-reset');
    const btnFs = toolbar.querySelector('.zoom-fullscreen');
    const iconExpand = toolbar.querySelector('.icon-expand');
    const iconCompress = toolbar.querySelector('.icon-compress');

    btnIn.addEventListener('click', (e) => {
      e.stopPropagation();
      setZoom(scale * 1.25);
    });

    btnOut.addEventListener('click', (e) => {
      e.stopPropagation();
      setZoom(scale * 0.8);
    });

    btnReset.addEventListener('click', (e) => {
      e.stopPropagation();
      resetView();
    });

    btnFs.addEventListener('click', (e) => {
      e.stopPropagation();
      const isFullscreen = wrapper.classList.toggle('is-fullscreen');
      if (isFullscreen) {
        document.body.style.overflow = 'hidden';
        iconExpand.style.display = 'none';
        iconCompress.style.display = 'block';
      } else {
        document.body.style.overflow = '';
        iconExpand.style.display = 'block';
        iconCompress.style.display = 'none';
      }
      resetView();
    });
  }

  // Scan and attach to all rendered Mermaid SVGs
  function scanAndAttach() {
    // Look for all SVG elements created by Mermaid
    const svgs = document.querySelectorAll('.mermaid svg, pre.mermaid svg, [id^="mermaid-"]');
    svgs.forEach(svg => {
      if (svg.tagName.toLowerCase() === 'svg') {
        attachPanZoomToSvg(svg);
      }
    });
  }

  // Global Continuous Observer for asynchronously rendered Mermaid diagrams
  const observer = new MutationObserver((mutations) => {
    let shouldScan = false;
    for (const m of mutations) {
      if (m.addedNodes.length > 0) {
        shouldScan = true;
        break;
      }
    }
    if (shouldScan) {
      scanAndAttach();
    }
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });

  // Initial and lifecycle triggers
  if (typeof document$ !== 'undefined') {
    document$.subscribe(() => {
      scanAndAttach();
      setTimeout(scanAndAttach, 150);
      setTimeout(scanAndAttach, 500);
      setTimeout(scanAndAttach, 1200);
    });
  } else {
    document.addEventListener('DOMContentLoaded', () => {
      scanAndAttach();
      setTimeout(scanAndAttach, 200);
      setTimeout(scanAndAttach, 800);
    });
  }
})();
