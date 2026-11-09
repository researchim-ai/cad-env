/* Shared controls for the guide and project notes. No CAD runs or remote services. */
(() => {
  'use strict';
  const menu = document.querySelector('.menu');
  const nav = document.querySelector('#nav');
  const close = () => { nav?.classList.remove('open'); menu?.setAttribute('aria-expanded', 'false'); };
  menu?.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(open));
  });
  nav?.addEventListener('click', event => { if (event.target.closest('a')) close(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && nav?.classList.contains('open')) { close(); menu.focus(); } });
  let timer;
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy)?.textContent;
    if (!text) return;
    const toast = document.querySelector('.toast');
    try {
      await navigator.clipboard.writeText(text);
      if (toast) toast.textContent = window.CAD_I18N.t('copied');
    } catch {
      if (toast) toast.textContent = window.CAD_I18N.t('clipboard');
    }
    toast?.classList.add('visible');
    clearTimeout(timer);
    timer = setTimeout(() => toast?.classList.remove('visible'), 2400);
  }));
})();
