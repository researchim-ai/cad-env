/* Browse the exact ordered goal images used by the evaluated policy. */
(() => {
  'use strict';
  const viewers = [...document.querySelectorAll('.model-input')];
  if (!viewers.length) return;
  const states = new Map();
  const ru = () => document.documentElement.lang === 'ru';
  const label = standard => standard === 'iso' ? 'ISO' : (ru() ? 'ГОСТ' : 'GOST');

  function render(viewer) {
    const state = states.get(viewer);
    if (!state) return;
    const input = state.example.inputs[state.standard];
    const image = input.images[state.index];
    const url = 'assets/student-solutions/' + image.file;
    viewer.querySelector('.input-image').src = url;
    viewer.querySelector('.input-image').alt = image.caption;
    viewer.querySelector('.input-image-link').href = url;
    viewer.querySelector('.input-original').href = url;
    viewer.querySelector('.input-counter').textContent = `${state.index + 1} / ${input.images.length}`;
    viewer.querySelector('.input-status').textContent = ru()
      ? `${label(state.standard)} · изображений на входе: ${input.images.length} · ${input.task_success ? 'задача решена' : 'задача не решена'}`
      : `${label(state.standard)} · ${input.images.length} input images · ${input.task_success ? 'task solved' : 'task not solved'}`;
    viewer.querySelectorAll('[data-input-standard]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.inputStandard === state.standard));
      button.textContent = label(button.dataset.inputStandard);
    });
    const card = viewer.closest('.student-example');
    card.querySelector('.result-input-origin').textContent = ru()
      ? `Деталь выше построена по ${label(card.dataset.resultStandard)}`
      : `Part above built from ${label(card.dataset.resultStandard)}`;
    const strip = viewer.querySelector('.input-thumbnails');
    const scroll = strip.scrollLeft;
    strip.replaceChildren(...input.images.map((item, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', `${ru() ? 'Изображение' : 'Image'} ${index + 1}: ${item.caption}`);
      button.setAttribute('aria-pressed', String(index === state.index));
      const thumb = document.createElement('img');
      thumb.src = 'assets/student-solutions/' + item.file;
      thumb.alt = ''; thumb.loading = 'lazy';
      button.append(thumb);
      button.addEventListener('click', () => { state.index = index; render(viewer); });
      return button;
    }));
    strip.scrollLeft = scroll;
  }

  fetch('assets/student-solutions/manifest.json', { cache: 'no-store' })
    .then(response => { if (!response.ok) throw new Error('Input manifest unavailable'); return response.json(); })
    .then(manifest => {
      viewers.forEach(viewer => {
        const example = manifest.examples.find(e => e.id === viewer.dataset.example);
        if (!example) throw new Error('Unknown example');
        const state = { example, standard: viewer.dataset.standard, index: 0 };
        states.set(viewer, state);
        viewer.querySelectorAll('[data-input-standard]').forEach(button => button.addEventListener('click', () => {
          state.standard = button.dataset.inputStandard; state.index = 0; render(viewer);
        }));
        viewer.querySelectorAll('[data-input-direction]').forEach(button => button.addEventListener('click', () => {
          const count = example.inputs[state.standard].images.length;
          state.index = (state.index + Number(button.dataset.inputDirection) + count) % count;
          render(viewer);
        }));
        render(viewer);
      });
    }).catch(error => {
      viewers.forEach(viewer => {
        viewer.querySelector('.input-status').textContent = ru() ? 'Не удалось загрузить список изображений.' : 'Could not load the input image list.';
      });
      console.error(error);
    });
  document.addEventListener('cad-languagechange', () => viewers.forEach(render));
})();
