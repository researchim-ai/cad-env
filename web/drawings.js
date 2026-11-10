/* The same TRAIN part in two drawing profiles. All assets are local. */
(() => {
  'use strict';
  const selector = document.getElementById('drawing-part');
  if (!selector) return;
  const formats = [...document.querySelectorAll('[data-drawing-format]')];
  let format = 'iso';
  const names = {
    bearing_cartridge: ['Bearing cartridge', 'Корпус подшипника'],
    adapter_flange: ['Adapter flange', 'Переходный фланец'],
    alignment_fixture: ['Alignment fixture', 'Установочное приспособление']
  };
  function render() {
    const ru = document.documentElement.lang === 'ru';
    const label = format === 'iso' ? 'ISO' : (ru ? 'ГОСТ' : 'GOST');
    const name = names[selector.value][ru ? 1 : 0];
    const base = `assets/drawings/${selector.value}-${format}`;
    const image = document.getElementById('drawing-preview');
    image.src = `${base}-sheet.png`;
    image.alt = `${name} — ${label}`;
    document.getElementById('drawing-image-link').href = `${base}-sheet.png`;
    document.getElementById('drawing-full').href = `${base}-sheet.png`;
    document.getElementById('drawing-pdf').href = `${base}-pdf.pdf`;
    document.getElementById('drawing-stamp').textContent = `${label} · A3`;
    document.getElementById('drawing-caption').textContent = `${name} · ${label}`;
    formats.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.drawingFormat === format)));
  }
  selector.addEventListener('change', render);
  formats.forEach(button => button.addEventListener('click', () => {
    format = button.dataset.drawingFormat;
    render();
  }));
  document.addEventListener('cad-languagechange', render);
  render();
})();
