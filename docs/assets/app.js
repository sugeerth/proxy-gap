/* Shared chrome: theme toggle and active-nav marking. */

const KEY = 'proxygap-theme';

function applyTheme(t) {
  if (t === 'light' || t === 'dark') {
    document.documentElement.setAttribute('data-theme', t);
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
}

function currentTheme() {
  const stored = localStorage.getItem(KEY);
  if (stored) return stored;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function initChrome() {
  applyTheme(localStorage.getItem(KEY));

  const btn = document.querySelector('.theme-toggle');
  if (btn) {
    const paint = () => { btn.textContent = currentTheme() === 'dark' ? '☀' : '☾'; };
    paint();
    btn.addEventListener('click', () => {
      const next = currentTheme() === 'dark' ? 'light' : 'dark';
      localStorage.setItem(KEY, next);
      applyTheme(next);
      paint();
      window.dispatchEvent(new CustomEvent('themechange'));
    });
  }

  const here = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.navlinks a').forEach((a) => {
    if ((a.getAttribute('href') || '').split('/').pop() === here) a.classList.add('active');
  });
}

/* Run a render function now and again whenever the theme flips, so charts that
 * resolved a CSS variable at draw time pick up the new palette. */
export function onThemeChange(fn) {
  window.addEventListener('themechange', fn);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!localStorage.getItem(KEY)) fn();
  });
}
