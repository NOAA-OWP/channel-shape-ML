/**
 * Model Release Version Selector Handler
 * Persists selected model version across universal pages (Home & References)
 * and dynamically synchronizes navigation tabs, action cards, and mobile drawer.
 */
document.addEventListener("DOMContentLoaded", function () {
    const headerSelect = document.getElementById("header-version-select");
    const drawerSelect = document.getElementById("drawer-version-select");
    const selects = [headerSelect, drawerSelect].filter(Boolean);
    if (selects.length === 0) return;

    function getSiteBase() {
        const homeAnchor = document.querySelector('.md-tabs__link[data-tab="home"]') || document.querySelector('a.md-logo');
        let siteBase = homeAnchor ? homeAnchor.href : window.location.origin + "/";
        if (!siteBase.endsWith("/")) siteBase += "/";
        return siteBase;
    }

    const currentUrl = window.location.href;
    const isV1Url = currentUrl.includes("/v1.0.0/");
    const siteBase = getSiteBase();

    let subPath = currentUrl.startsWith(siteBase) ? currentUrl.substring(siteBase.length) : "";
    subPath = subPath.split("?")[0].split("#")[0].replace(/^index\.html$/, "");
    const isSharedPage = (subPath === "" || subPath === "/" || subPath.startsWith("references"));

    let activeVersion = "v2.0.0";

    if (isV1Url) {
        activeVersion = "v1.0.0";
        localStorage.setItem("ml_web_version", "v1.0.0");
    } else if (isSharedPage) {
        const savedVersion = localStorage.getItem("ml_web_version");
        if (savedVersion === "v1.0.0") {
            activeVersion = "v1.0.0";
        } else {
            activeVersion = "v2.0.0";
        }
    } else {
        activeVersion = "v2.0.0";
        localStorage.setItem("ml_web_version", "v2.0.0");
    }

    // Set dropdown visual values on all selectors
    selects.forEach(s => { s.value = activeVersion; });

    function applyVersionState(version) {
        const tabOverview = document.querySelector('.md-tabs__link[data-tab="overview"]');
        const tabTW = document.querySelector('.md-tabs__link[data-tab="tw"]');
        const tabY = document.querySelector('.md-tabs__link[data-tab="y"]');
        const tabR = document.querySelector('.md-tabs__link[data-tab="r"]');
        const tabN = document.querySelector('.md-tabs__link[data-tab="n"]');

        if (version === "v1.0.0") {
            if (tabOverview) tabOverview.href = siteBase + "v1.0.0/overview/";
            if (tabTW) tabTW.href = siteBase + "v1.0.0/tw/";
            if (tabY) tabY.href = siteBase + "v1.0.0/y/";
            if (tabR) tabR.href = siteBase + "v1.0.0/r/";
            if (tabN) tabN.href = siteBase + "v1.0.0/n/";
        } else {
            if (tabOverview) tabOverview.href = siteBase + "overview/";
            if (tabTW) tabTW.href = siteBase + "tw/";
            if (tabY) tabY.href = siteBase + "y/";
            if (tabR) tabR.href = siteBase + "r/";
            if (tabN) tabN.href = siteBase + "n/";
        }

        const heroCta = document.querySelector('.hero-cta');
        if (heroCta) {
            heroCta.href = (version === "v1.0.0") ? (siteBase + "v1.0.0/overview/") : (siteBase + "overview/");
        }

        const cardLinks = document.querySelectorAll('.pipeline-card a');
        cardLinks.forEach(link => {
            const href = link.getAttribute('href') || "";
            if (version === "v1.0.0") {
                if (href.includes("tw") && !href.includes("v1.0.0")) link.href = siteBase + "v1.0.0/tw/";
                else if (href.includes("y") && !href.includes("v1.0.0")) link.href = siteBase + "v1.0.0/y/";
                else if (href.includes("r") && !href.includes("v1.0.0")) link.href = siteBase + "v1.0.0/r/";
                else if (href.includes("n") && !href.includes("v1.0.0")) link.href = siteBase + "v1.0.0/n/";
            } else {
                if (href.includes("tw")) link.href = siteBase + "tw/";
                else if (href.includes("y")) link.href = siteBase + "y/";
                else if (href.includes("r")) link.href = siteBase + "r/";
                else if (href.includes("n")) link.href = siteBase + "n/";
            }
        });
    }

    if (isSharedPage) {
        applyVersionState(activeVersion);
    }

    function onVersionChanged(targetVersion) {
        localStorage.setItem("ml_web_version", targetVersion);
        selects.forEach(s => { s.value = targetVersion; });

        if (isSharedPage) {
            applyVersionState(targetVersion);
            return;
        }

        if (targetVersion === "v1.0.0" && !isV1Url) {
            if (currentUrl.includes("/tw/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "v1.0.0/tw/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "v1.0.0/tw/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "v1.0.0/tw/xai/";
                else window.location.href = siteBase + "v1.0.0/tw/";
            } else if (currentUrl.includes("/y/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "v1.0.0/y/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "v1.0.0/y/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "v1.0.0/y/xai/";
                else window.location.href = siteBase + "v1.0.0/y/";
            } else if (currentUrl.includes("/r/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "v1.0.0/r/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "v1.0.0/r/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "v1.0.0/r/xai/";
                else window.location.href = siteBase + "v1.0.0/r/";
            } else if (currentUrl.includes("/n/")) {
                window.location.href = siteBase + "v1.0.0/n/";
            } else if (currentUrl.includes("/overview/")) {
                if (currentUrl.includes("data-sources")) window.location.href = siteBase + "v1.0.0/overview/data-sources/";
                else if (currentUrl.includes("feature-engineering") || currentUrl.includes("methods")) window.location.href = siteBase + "v1.0.0/overview/methods/";
                else window.location.href = siteBase + "v1.0.0/overview/";
            } else {
                window.location.href = siteBase + "v1.0.0/overview/";
            }
        } else if (targetVersion === "v2.0.0" && isV1Url) {
            if (currentUrl.includes("/tw/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "tw/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "tw/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "tw/xai/";
                else window.location.href = siteBase + "tw/";
            } else if (currentUrl.includes("/y/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "y/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "y/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "y/xai/";
                else window.location.href = siteBase + "y/";
            } else if (currentUrl.includes("/r/")) {
                if (currentUrl.includes("models")) window.location.href = siteBase + "r/models/";
                else if (currentUrl.includes("skill")) window.location.href = siteBase + "r/skill/";
                else if (currentUrl.includes("xai")) window.location.href = siteBase + "r/xai/";
                else window.location.href = siteBase + "r/";
            } else if (currentUrl.includes("/n/")) {
                window.location.href = siteBase + "n/";
            } else if (currentUrl.includes("/overview/")) {
                if (currentUrl.includes("data-sources")) window.location.href = siteBase + "overview/data-sources/";
                else if (currentUrl.includes("methods")) window.location.href = siteBase + "overview/feature-engineering/";
                else window.location.href = siteBase + "overview/";
            } else {
                window.location.href = siteBase;
            }
        }
    }

    selects.forEach(s => {
        s.addEventListener("change", function(e) {
            onVersionChanged(e.target.value);
        });
    });

    // Drawer Theme Toggle Button Handler
    const drawerThemeBtn = document.getElementById("drawer-theme-toggle");
    if (drawerThemeBtn) {
        drawerThemeBtn.addEventListener("click", function (e) {
            e.preventDefault();
            const currentScheme = document.body.getAttribute("data-md-color-scheme") || "slate";
            const targetId = (currentScheme === "slate") ? "__palette_1" : "__palette_0";
            const targetRadio = document.getElementById(targetId);
            if (targetRadio) {
                targetRadio.click();
            }
        });
    }

    // Drawer Auto-Hide on Hover with grace period & indicator injection safeguard
    function setupDrawerAutoHide() {
        const sidebar = document.querySelector('.md-sidebar--primary');
        if (!sidebar) return;

        // Safeguard: Ensure edge handle exists inside sidebar
        let handle = sidebar.querySelector('.drawer-edge-handle');
        if (!handle) {
            handle = document.createElement('label');
            handle.className = 'drawer-edge-handle';
            handle.setAttribute('for', '__drawer');
            handle.setAttribute('title', 'Open navigation menu');
            handle.setAttribute('aria-label', 'Open navigation menu');
            handle.innerHTML = `
                <span class="drawer-edge-handle-bar"></span>
                <svg class="drawer-edge-handle-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/>
                </svg>
                <span class="drawer-edge-handle-label">Menu</span>
            `;
            sidebar.insertBefore(handle, sidebar.firstChild);
        }

        let hoverZone = sidebar.querySelector('.drawer-edge-hover-zone');
        if (!hoverZone) {
            hoverZone = document.createElement('div');
            hoverZone.className = 'drawer-edge-hover-zone';
            hoverZone.setAttribute('aria-hidden', 'true');
            sidebar.insertBefore(hoverZone, sidebar.firstChild);
        }

        let hideTimeout = null;

        function openDrawer() {
            if (hideTimeout) {
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }
            sidebar.classList.add('drawer-hover-open');
        }

        function closeDrawer() {
            if (hideTimeout) clearTimeout(hideTimeout);
            hideTimeout = setTimeout(() => {
                sidebar.classList.remove('drawer-hover-open');
            }, 260);
        }

        sidebar.addEventListener('mouseenter', openDrawer);
        sidebar.addEventListener('mouseleave', closeDrawer);

        // Close hover drawer if a navigation link inside is clicked
        sidebar.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                sidebar.classList.remove('drawer-hover-open');
            });
        });
    }

    setupDrawerAutoHide();
});
