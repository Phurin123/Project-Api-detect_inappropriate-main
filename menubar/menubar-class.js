(function () {
  const scriptEl =
    document.currentScript ||
    (function () {
      const scripts = document.getElementsByTagName('script');
      for (let i = scripts.length - 1; i >= 0; i -= 1) {
        const src = scripts[i].getAttribute('src') || '';
        if (src.includes('menubar')) {
          return scripts[i];
        }
      }
      return null;
    })();

  const baseUrl = scriptEl ?
    scriptEl.src.substring(0, scriptEl.src.lastIndexOf('/') + 1) :
    '';

  const templateUrl = scriptEl?.dataset.template ?
    new URL(scriptEl.dataset.template, scriptEl.src).toString() :
    `${baseUrl}menubar.html`;

  const stylesheetUrl = scriptEl?.dataset.stylesheet ?
    new URL(scriptEl.dataset.stylesheet, scriptEl.src).toString() :
    `${baseUrl}menubar.css`;

  const logoutRedirect = scriptEl?.dataset.logoutRedirect || '/homepage/index.html';

  let stylesheetAttached = false;

  function ensureStylesheet() {
    if (stylesheetAttached) {
      return;
    }
    const existing = Array.from(
      document.querySelectorAll('link[rel="stylesheet"][href]')
    ).some((link) => link.href === stylesheetUrl);
    if (existing) {
      stylesheetAttached = true;
      return;
    }
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = stylesheetUrl;
    link.dataset.menubarLoaded = 'true';
    document.head.appendChild(link);
    stylesheetAttached = true;
  }

  function resolveTarget(target) {
    if (typeof target === 'string') {
      return document.querySelector(target);
    }
    if (target instanceof HTMLElement) {
      return target;
    }
    return null;
  }

  function toggleDropdown(dropdown, open) {
    const isOpen = open !== undefined ? open : !dropdown.classList.contains('open');
    const button = dropdown.querySelector('.nav-link');
    if (isOpen) {
      dropdown.classList.add('open');
      button?.setAttribute('aria-expanded', 'true');
    } else {
      dropdown.classList.remove('open');
      button?.setAttribute('aria-expanded', 'false');
    }
  }

  function closeAllDropdowns(navEl) {
    navEl.querySelectorAll('.nav-item--dropdown.open').forEach((item) => {
      toggleDropdown(item, false);
    });
  }

  function refreshAuthState() {
    const hasToken = !!localStorage.getItem('token');
    document.body.classList.toggle('is-authenticated', hasToken);

    const loginActions = document.querySelectorAll('[data-auth="login"], [data-auth="signup"]');
    const logoutActions = document.querySelectorAll('[data-auth="logout"]');

    loginActions.forEach((el) => {
      try {
        el.toggleAttribute('hidden', hasToken);
        if (hasToken) {
          el.style.display = 'none';
        } else {
          el.style.display = '';
        }
      } catch (err) {
        console.warn('login action toggle failed', err);
      }
    });

    logoutActions.forEach((el) => {
      try {
        el.toggleAttribute('hidden', !hasToken);
        if (!hasToken) {
          el.style.display = 'none';
        } else {
          el.style.display = '';
        }
      } catch (err) {
        console.warn('logout action toggle failed', err);
      }
    });
  }

  function handleLogoutClick(event) {
    event.preventDefault();
    try {
      localStorage.removeItem('token');
      sessionStorage.removeItem('token');
    } catch (err) {
      console.warn('Unable to clear authentication token:', err);
    }

    refreshAuthState();

    if (typeof window.onNavLogout === 'function') {
      try {
        window.onNavLogout();
      } catch (err) {
        console.error('onNavLogout error', err);
      }
    }

    if (typeof window.logout === 'function') {
      try {
        window.logout();
        return;
      } catch (err) {
        console.error('logout handler error', err);
      }
    }

    window.location.href = logoutRedirect;
  }

  function initialiseNav(navHost) {
    if (!navHost) {
      return;
    }

    document.body.classList.add('has-fixed-nav');
    refreshAuthState();

    const toggleButton = navHost.querySelector('.nav-toggle');
    const linksWrapper = navHost.querySelector('.nav-links-wrapper');
    const logoutButton = navHost.querySelector('[data-auth="logout"]');

    if (toggleButton && linksWrapper) {
      toggleButton.addEventListener('click', () => {
        const isOpen = linksWrapper.classList.toggle('open');
        toggleButton.setAttribute('aria-expanded', String(isOpen));
        if (!isOpen) {
          closeAllDropdowns(navHost);
        }
      });
    }

    navHost.querySelectorAll('.nav-item--dropdown').forEach((item) => {
      const trigger = item.querySelector('.nav-link--button');
      if (!trigger) {
        return;
      }

      trigger.addEventListener('click', (event) => {
        event.preventDefault();
        const alreadyOpen = item.classList.contains('open');
        closeAllDropdowns(navHost);
        toggleDropdown(item, !alreadyOpen);
      });
    });

    document.addEventListener('click', (event) => {
      if (!navHost.contains(event.target)) {
        closeAllDropdowns(navHost);
        linksWrapper?.classList.remove('open');
        toggleButton?.setAttribute('aria-expanded', 'false');
      }
    });

    navHost.querySelectorAll('a.nav-link, .dropdown-item').forEach((anchor) => {
      anchor.addEventListener('click', () => {
        linksWrapper?.classList.remove('open');
        toggleButton?.setAttribute('aria-expanded', 'false');
        closeAllDropdowns(navHost);
      });
    });

    if (logoutButton) {
      logoutButton.addEventListener('click', handleLogoutClick);
    }

    // ป้องกันไม่ให้เข้า Your API Keys ถ้าไม่ล็อกอิน
    const apiKeyLink = navHost.querySelector('a[href="/apikey/view-api-keys.html"]');
    if (apiKeyLink) {
      apiKeyLink.addEventListener('click', (event) => {
        const hasToken = !!localStorage.getItem('token');
        if (!hasToken) {
          event.preventDefault();
          alert('กรุณาเข้าสู่ระบบก่อนเข้าหน้า API Keys');
          window.location.href = '../login-singup/login.html';
        }
      });
    }

  }

  async function loadMenubar(target) {
    const container = resolveTarget(target) || resolveTarget('#menubar');
    if (!container || container.dataset.menubarLoaded === 'true') {
      return;
    }

    ensureStylesheet();

    try {
      const response = await fetch(templateUrl, {
        credentials: 'same-origin'
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch menubar template: ${response.status}`);
      }
      const markup = await response.text();
      container.innerHTML = markup;
      container.dataset.menubarLoaded = 'true';
      const navHost = container.querySelector('[data-menubar]');
      initialiseNav(navHost);
    } catch (error) {
      console.error('Unable to load menubar:', error);
    }

  }

  class Menubar {
    constructor(options = {}) {
      this.target = options.target || '#menubar';
    }

    async load() {
      await loadMenubar(this.target);
    }
  }

  window.Menubar = Menubar;
  window.refreshMenubarAuthState = refreshAuthState;

  document.addEventListener('DOMContentLoaded', () => {
    let host = document.querySelector('#menubar');
    if (!host) {
      host = document.createElement('div');
      host.id = 'menubar';
      // Insert at top of body to ensure visibility on all pages
      if (document.body.firstChild) {
        document.body.insertBefore(host, document.body.firstChild);
      } else {
        document.body.appendChild(host);
      }
    }
    loadMenubar(host);
  });
})();